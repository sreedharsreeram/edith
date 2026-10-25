"""Gemini adapter against a fake SDK: tool calls, history format, fallback, failures."""

import asyncio

import httpx
import pytest
from google.genai import errors, types
from helpers import TINY_JPEG

from edith.agent.harness import REFUSAL_TEXT, BrainError, Exchange, Tool, ToolResult
from edith.models import gemini as gemini_module
from edith.models.gemini import Gemini

TOOL = Tool(
    name="walk_around",
    description="walk",
    parameters={"type": "object", "properties": {"instruction": {"type": "string"}}},
    run=None,
)


def api_error(code: int, message: str = "x") -> errors.APIError:
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


def text_chunk(text: str, **part_fields) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text, **part_fields)]))]
    )


def call_chunk(name: str, args: dict, call_id: str | None = None, signature: bytes | None = None):
    part = types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args), thought_signature=signature)
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))])


def blocked_chunk() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)])


class FakeModels:
    """Stands in for client.aio.models. Each request pops the next reply: chunks, an error, or a delay (float)."""

    def __init__(self, replies: dict[str, list]) -> None:
        self.replies = replies
        self.requests: list[dict] = []

    async def generate_content_stream(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        reply = self.replies[model].pop(0)
        if isinstance(reply, Exception):
            raise reply

        async def stream():
            for item in reply:
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, float):
                    await asyncio.sleep(item)
                    continue
                yield item

        return stream()

    async def get(self, *, model):
        raise OSError("offline")


class Harness:
    def __init__(self, replies: dict[str, list], history=None) -> None:
        self.api = FakeModels(replies)
        self.gemini = Gemini(models=list(replies), client=_client(self.api))
        self.conversation = self.gemini.start(
            system="sys", history=history or [], photo=TINY_JPEG, question="what is it?", tools=[TOOL]
        )
        self.streamed: list[str] = []

    def step(self, allow_tools=True):
        async def on_text(text):
            self.streamed.append(text)

        return asyncio.run(self.conversation.step(on_text, allow_tools))


def _client(models):
    class Aio:
        pass

    class Client:
        aio = Aio()

    Client.aio.models = models
    return Client()


def test_first_request_has_photo_question_history_and_tools():
    h = Harness({"m": [[text_chunk("Hi")]]}, history=[Exchange("earlier q", "earlier a")])
    h.step()
    request = h.api.requests[0]
    user_earlier, model_earlier, now = request["contents"]
    assert (user_earlier.role, user_earlier.parts[0].text) == ("user", "earlier q")
    assert (model_earlier.role, model_earlier.parts[0].text) == ("model", "earlier a")
    assert now.parts[0].inline_data.data == TINY_JPEG
    assert now.parts[1].text == "what is it?"
    config = request["config"]
    assert config.system_instruction == "sys"
    assert config.tools[0].function_declarations[0].name == "walk_around"
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO
    assert config.automatic_function_calling.disable is True  # the SDK must never run tools behind our back


def test_streams_text_and_merges_it_in_history():
    h = Harness({"m": [[text_chunk("A red "), text_chunk("apple.")]]})
    assert h.step() == []
    assert h.streamed == ["A red ", "apple."]
    assert [p.text for p in h.conversation.contents[-1].parts] == ["A red apple."]


def test_thoughts_are_not_shown_on_the_glasses():
    h = Harness({"m": [[text_chunk("thinking about it", thought=True), text_chunk("Answer.")]]})
    h.step()
    assert h.streamed == ["Answer."]


def test_tool_call_round_trip_keeps_ids_and_thought_signature():
    h = Harness({"m": [[call_chunk("walk_around", {"instruction": "go"}, "fc-1", b"sig")], [text_chunk("A Jaguar.")]]})
    [c] = h.step()
    assert (c.id, c.name, c.args) == ("fc-1", "walk_around", {"instruction": "go"})

    h.conversation.add_results([c], [ToolResult("8 views", images=[b"jpeg1", b"jpeg2"])])
    h.step()

    second = h.api.requests[1]["contents"]
    model_turn, results_turn = second[-2], second[-1]
    assert model_turn.parts[0].thought_signature == b"sig"  # Gemini 3 rejects the turn if this is dropped
    response = results_turn.parts[0].function_response
    assert (response.id, response.name, response.response) == ("fc-1", "walk_around", {"result": "8 views"})
    assert [p.inline_data.data for p in response.parts] == [b"jpeg1", b"jpeg2"]


def test_calls_without_ids_get_our_own_and_none_goes_back():
    h = Harness({"m": [[call_chunk("walk_around", {})], [text_chunk("ok")]]})
    [c] = h.step()
    assert c.id
    h.conversation.add_results([c], [ToolResult("failed", is_error=True)])
    response = h.conversation.contents[-1].parts[0].function_response
    assert response.id is None
    assert response.response == {"error": "failed"}


def test_last_step_turns_tools_off():
    h = Harness({"m": [[text_chunk("ok")]]})
    h.step(allow_tools=False)
    assert h.api.requests[0]["config"].tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE


def test_safety_block_says_so():
    h = Harness({"m": [[blocked_chunk()]]})
    assert h.step() == []
    assert h.streamed == [REFUSAL_TEXT]


class TestFallback:
    @pytest.mark.parametrize("code", [429, 500, 503, 504])
    def test_busy_model_falls_back(self, code):
        h = Harness({"new": [api_error(code)], "old": [[text_chunk("It says STOP.")]]})
        h.step()
        assert [r["model"] for r in h.api.requests] == ["new", "old"]
        assert h.conversation.model_name == "old"

    def test_hanging_model_falls_back(self, monkeypatch):
        monkeypatch.setattr(gemini_module, "FIRST_OUTPUT_TIMEOUT_S", 0.05)
        h = Harness({"hangs": [[5.0]], "old": [[text_chunk("ok")]]})
        h.step()
        assert h.streamed == ["ok"]

    def test_slow_answer_is_not_cut_off_once_started(self, monkeypatch):
        monkeypatch.setattr(gemini_module, "FIRST_OUTPUT_TIMEOUT_S", 0.05)
        h = Harness({"m": [[text_chunk("It says "), 0.1, text_chunk("STOP.")]]})
        h.step()
        assert h.streamed == ["It says ", "STOP."]

    def test_never_splices_two_answers_together(self):
        h = Harness({"new": [[text_chunk("It says "), api_error(503)]], "old": [[text_chunk("STOP.")]]})
        with pytest.raises(BrainError):
            h.step()
        assert [r["model"] for r in h.api.requests] == ["new"]

    def test_later_steps_stay_on_the_same_model(self):
        """Thought signatures belong to the model that made them, so no switching mid-turn."""
        h = Harness({"new": [api_error(503)], "old": [[call_chunk("walk_around", {}, "c1")], api_error(503)]})
        [c] = h.step()
        h.conversation.add_results([c], [ToolResult("ok")])
        with pytest.raises(BrainError):
            h.step()
        assert [r["model"] for r in h.api.requests] == ["new", "old", "old"]

    def test_busy_model_is_skipped_next_turn(self):
        h = Harness({"new": [api_error(503)], "old": [[text_chunk("one")], [text_chunk("two")]]})
        h.step()
        second = h.gemini.start(system="s", history=[], photo=TINY_JPEG, question="q", tools=[])

        async def ignore(text):
            pass

        asyncio.run(second.step(ignore, True))
        assert [r["model"] for r in h.api.requests] == ["new", "old", "old"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (api_error(400, "API_KEY_INVALID"), "Server API key is invalid."),
        (api_error(400, "bad schema"), "AI service error. Try again."),  # not every 400 is a bad key
        (api_error(403), "Server API key is invalid."),
        (api_error(429, "rate limited"), "Busy right now. Try again."),
        (api_error(429, "You exceeded your current quota, please check your plan"), "AI quota used up. Check API billing."),
        (httpx.ConnectError("refused"), "Can't reach the AI. Check WiFi."),
        (OSError("no route"), "Can't reach the AI. Check WiFi."),
    ],
)
def test_errors_become_friendly_messages(error, message):
    h = Harness({"m": [error]})
    with pytest.raises(BrainError) as e:
        h.step()
    assert e.value.user_message == message


def test_timeout_mid_answer():
    h = Harness({"m": [[text_chunk("It says "), httpx.ReadTimeout("x")]]})
    with pytest.raises(BrainError, match="slow right now"):
        h.step()


def test_warm_never_raises():
    asyncio.run(Harness({"m": []}).gemini.warm())


def test_model_list_from_env(monkeypatch):
    monkeypatch.setenv("EDITH_GEMINI_MODELS", "x-model, y-model")
    assert Gemini(client=object()).models == ["x-model", "y-model"]

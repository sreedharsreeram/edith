"""Claude adapter against a fake HTTP API: request shape, tool calls, results, refusals, errors."""

import asyncio
import base64
import json

import anthropic
import httpx2
import pytest
from helpers import TINY_JPEG

from edith.agent.harness import REFUSAL_TEXT, BrainError, Exchange, Tool, ToolResult
from edith.models.claude import Claude

TOOL = Tool(
    name="walk_around",
    description="walk",
    parameters={"type": "object", "properties": {"instruction": {"type": "string"}}},
    run=None,
)


def sse(*blocks, stop_reason="end_turn") -> str:
    """A streamed Messages API response. blocks: strings (text) or (id, name, args) tuples (tool calls)."""
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5-5",
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0},
    }})]
    for i, block in enumerate(blocks):
        if isinstance(block, str):
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block}
        else:
            call_id, name, args = block
            start = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(args)}
        events += [
            ("content_block_start", {"type": "content_block_start", "index": i, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


class Harness:
    def __init__(self, *replies, history=None, tools=(TOOL,)) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []
        self.headers: list = []
        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
        )
        self.claude = Claude(model="claude-opus-5-5", client=client)
        self.conversation = self.claude.start(
            system="sys", history=history or [], photo=TINY_JPEG, question="what is it?", tools=list(tools)
        )
        self.streamed: list[str] = []

    def handle(self, request):
        self.bodies.append(json.loads(request.content))
        self.headers.append(request.headers)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx2.Response):
            return reply
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=reply)

    def step(self, allow_tools=True):
        async def on_text(text):
            self.streamed.append(text)

        return asyncio.run(self.conversation.step(on_text, allow_tools))


def test_request_shape():
    h = Harness(sse("Hi"), history=[Exchange("earlier q", "earlier a")])
    h.step()
    body = h.bodies[0]
    assert (body["model"], body["system"], body["stream"], body["fallbacks"]) == ("claude-opus-5-5", "sys", True, "default")
    assert "server-side-fallback-2026-07-01" in h.headers[0]["anthropic-beta"]
    assert body["tools"] == [{"name": "walk_around", "description": "walk", "input_schema": TOOL.parameters}]
    assert body["tool_choice"] == {"type": "auto"}
    earlier_q, earlier_a, now = body["messages"]
    assert (earlier_q["content"], earlier_a["content"]) == ("earlier q", "earlier a")
    image, question = now["content"]
    assert image["source"]["data"] == base64.standard_b64encode(TINY_JPEG).decode()
    assert question == {"type": "text", "text": "what is it?"}


def test_no_tools_means_no_tool_fields():
    h = Harness(sse("Hi"), tools=())
    h.step()
    assert "tools" not in h.bodies[0] and "tool_choice" not in h.bodies[0]


def test_streams_text():
    h = Harness(sse("A red apple."))
    assert h.step() == []
    assert h.streamed == ["A red apple."]


def test_tool_call_round_trip():
    h = Harness(sse(("toolu_1", "walk_around", {"instruction": "go"}), stop_reason="tool_use"), sse("A Jaguar."))
    [c] = h.step()
    assert (c.id, c.name, c.args) == ("toolu_1", "walk_around", {"instruction": "go"})

    h.conversation.add_results([c], [ToolResult("8 views", images=[b"jpeg1"])])
    h.step()
    assistant, results = h.bodies[1]["messages"][-2:]
    assert assistant["role"] == "assistant" and assistant["content"][0]["type"] == "tool_use"
    [result] = results["content"]
    assert (result["type"], result["tool_use_id"], result["is_error"]) == ("tool_result", "toolu_1", False)
    text, image = result["content"]
    assert text == {"type": "text", "text": "8 views"}
    assert image["source"]["data"] == base64.standard_b64encode(b"jpeg1").decode()


def test_parallel_results_go_in_one_message():
    h = Harness(sse(("t1", "walk_around", {}), ("t2", "walk_around", {}), stop_reason="tool_use"), sse("ok"))
    calls = h.step()
    h.conversation.add_results(calls, [ToolResult("a"), ToolResult("", is_error=True)])
    h.step()
    results = h.bodies[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    assert results[1]["content"][0]["text"] == "(no output)"  # empty text blocks are rejected by the API


def test_last_step_turns_tools_off():
    h = Harness(sse("ok"))
    h.step(allow_tools=False)
    assert h.bodies[0]["tool_choice"] == {"type": "none"}


def test_refusal_says_so():
    h = Harness(sse(stop_reason="refusal"))
    assert h.step() == []
    assert h.streamed == [REFUSAL_TEXT]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (httpx2.Response(401, json={"type": "error", "error": {"type": "x", "message": "x"}}), "Server API key is invalid."),
        (httpx2.Response(429, json={"type": "error", "error": {"type": "x", "message": "x"}}), "Busy right now. Try again."),
        (httpx2.Response(529, json={"type": "error", "error": {"type": "x", "message": "x"}}), "AI service error. Try again."),
        (httpx2.ConnectError("no route"), "Can't reach the AI. Check WiFi."),
        (httpx2.ReadTimeout("slow"), "AI is slow right now. Try again."),
    ],
)
def test_errors_become_friendly_messages(reply, message):
    h = Harness(reply)
    with pytest.raises(BrainError) as e:
        h.step()
    assert e.value.user_message == message


def test_warm_never_raises():
    asyncio.run(Harness(httpx2.ConnectError("offline")).claude.warm())

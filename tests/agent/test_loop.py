"""The agent loop with a scripted model: tool calling, failures, limits, safety."""

import asyncio
import time

import pytest
from helpers import FakeDevice, ScriptedModel, Step, call, jpeg

from server.agent.core import NO_ANSWER_TEXT, Agent, BrainError, Exchange, Tool, ToolError, ToolResult
from server.agent.trace import Trace

PHOTO = jpeg("red")
NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}


def tool(name="lookup", run=None, **kwargs) -> Tool:
    async def default(ctx, args):
        return f"{name} result"

    params = kwargs.pop("parameters", NO_ARGS)
    return Tool(name=name, description=f"{name} tool", parameters=params, run=run or default, **kwargs)


def run(agent: Agent, device=None, history=None, question="what is this?") -> tuple[str, list[str], Trace]:
    streamed: list[str] = []
    trace = Trace()

    async def on_text(text):
        streamed.append(text)

    answer = asyncio.run(
        agent.run(
            question=question,
            photo=PHOTO,
            history=history or [],
            device=device or FakeDevice(),
            on_text=on_text,
            trace=trace,
        )
    )
    return answer, streamed, trace


def test_answers_directly_without_tools():
    model = ScriptedModel(Step(text=["A red ", "apple."]))
    answer, streamed, _ = run(Agent(model, [tool()], "sys"))
    assert answer == "A red apple."
    assert streamed == ["A red ", "apple."]  # streamed as it was written, not all at the end
    assert model.conversations[0].allow_tools == [True]


def test_model_sees_photo_question_history_tools_and_system_prompt():
    model = ScriptedModel(Step(text=["ok"]))
    history = [Exchange("what car?", "A Jaguar E-Type.")]
    run(Agent(model, [tool("a"), tool("b")], "be brief"), history=history, question="what year?")
    c = model.conversations[0]
    assert (c.system, c.photo, c.question, c.history) == ("be brief", PHOTO, "what year?", history)
    assert [t.name for t in c.tools] == ["a", "b"]


def test_tool_result_goes_back_to_the_model():
    model = ScriptedModel(Step(calls=[call("lookup")]), Step(text=["Found it."]))
    answer, _, trace = run(Agent(model, [tool()], "sys"))
    assert answer == "Found it."
    [[(c, result)]] = model.conversations[0].results
    assert (c.name, result.text, result.is_error) == ("lookup", "lookup result", False)
    assert [e["kind"] for e in trace.events] == ["model", "tool", "model"]


def test_parallel_tool_calls_run_concurrently():
    async def slow(ctx, args):
        await asyncio.sleep(0.2)
        return "done"

    model = ScriptedModel(Step(calls=[call("a"), call("b"), call("c")]), Step(text=["ok"]))
    tools = [tool("a", slow), tool("b", slow), tool("c", slow)]
    started = time.perf_counter()
    run(Agent(model, tools, "sys"))
    assert time.perf_counter() - started < 0.5  # 3 x 0.2s in parallel, not 0.6s in a row
    [results] = model.conversations[0].results
    assert [c.name for c, _ in results] == ["a", "b", "c"]  # results stay matched to their calls


def test_tool_images_reach_the_model():
    async def photo(ctx, args):
        return ToolResult("two views", images=[b"img1", b"img2"])

    model = ScriptedModel(Step(calls=[call("look")]), Step(text=["ok"]))
    run(Agent(model, [tool("look", photo)], "sys"))
    [[(_, result)]] = model.conversations[0].results
    assert result.images == [b"img1", b"img2"]


class TestToolFailuresGoBackToTheModel:
    """A broken tool must never crash the turn; the model hears what happened and can recover."""

    def result_for(self, tools, calls) -> ToolResult:
        model = ScriptedModel(Step(calls=calls), Step(text=["recovered"]))
        answer, _, _ = run(Agent(model, tools, "sys"))
        assert answer == "recovered"
        [[(_, result)]] = model.conversations[0].results
        assert result.is_error
        return result

    def test_unknown_tool(self):
        assert "no tool called 'nope'" in self.result_for([tool()], [call("nope")]).text

    def test_invalid_arguments(self):
        t = tool(parameters={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
        assert "Invalid arguments" in self.result_for([t], [call("lookup", n="five")]).text
        assert "Invalid arguments" in self.result_for([t], [call("lookup")]).text

    def test_expected_tool_error(self):
        async def fails(ctx, args):
            raise ToolError("No signal in the parking garage.")

        assert self.result_for([tool(run=fails)], [call("lookup")]).text == "No signal in the parking garage."

    def test_crash_does_not_leak_details(self):
        async def crashes(ctx, args):
            raise RuntimeError("password=hunter2")

        result = self.result_for([tool(run=crashes)], [call("lookup")])
        assert "hunter2" not in result.text
        assert result.text == "lookup failed unexpectedly."

    def test_timeout(self):
        async def hangs(ctx, args):
            await asyncio.sleep(10)

        result = self.result_for([tool(run=hangs, timeout_s=0.05)], [call("lookup")])
        assert "took too long" in result.text


def test_huge_tool_output_is_truncated():
    async def huge(ctx, args):
        return "x" * 100_000

    model = ScriptedModel(Step(calls=[call("lookup")]), Step(text=["ok"]))
    run(Agent(model, [tool(run=huge)], "sys"))
    [[(_, result)]] = model.conversations[0].results
    assert len(result.text) < 8_100
    assert result.text.endswith("[truncated]")


def test_last_step_has_tools_turned_off_so_it_must_answer():
    model = ScriptedModel(*[Step(calls=[call("lookup")]) for _ in range(3)], Step(text=["Final answer."]))
    answer, _, _ = run(Agent(model, [tool()], "sys", max_steps=4))
    assert answer == "Final answer."
    assert model.conversations[0].allow_tools == [True, True, True, False]


def test_tools_requested_on_the_last_step_are_not_run():
    ran = []

    async def record(ctx, args):
        ran.append(1)
        return "ok"

    model = ScriptedModel(Step(calls=[call("lookup")]), Step(calls=[call("lookup")]))
    answer, _, _ = run(Agent(model, [tool(run=record)], "sys", max_steps=2))
    assert ran == [1]
    assert answer == NO_ANSWER_TEXT


def test_empty_answer_says_so():
    answer, streamed, _ = run(Agent(ScriptedModel(Step(text=[])), [], "sys"))
    assert answer == NO_ANSWER_TEXT
    assert streamed == [NO_ANSWER_TEXT]


def test_model_failure_propagates():
    model = ScriptedModel(Step(error=BrainError("Busy")))
    with pytest.raises(BrainError):
        run(Agent(model, [], "sys"))


class TestConfirmation:
    """Tools with side effects need a button press. Nothing in a photo or email can skip it."""

    def setup_method(self):
        self.ran = []

        async def delete(ctx, args):
            self.ran.append(args)
            return "deleted"

        self.tool = tool("delete", delete, confirm=lambda args: "Delete everything?")
        self.model = ScriptedModel(Step(calls=[call("delete")]), Step(text=["ok"]))

    def test_runs_when_wearer_approves(self):
        device = FakeDevice(confirm=True)
        run(Agent(self.model, [self.tool], "sys"), device=device)
        assert device.confirms == ["Delete everything?"]
        assert self.ran == [{}]

    def test_does_not_run_when_wearer_declines(self):
        device = FakeDevice(confirm=False)
        run(Agent(self.model, [self.tool], "sys"), device=device)
        assert self.ran == []
        [[(_, result)]] = self.model.conversations[0].results
        assert result.is_error and "declined" in result.text

    def test_invalid_arguments_are_rejected_before_asking(self):
        t = tool("delete", confirm=lambda a: "?", parameters={"type": "object", "required": ["id"]})
        device = FakeDevice()
        model = ScriptedModel(Step(calls=[call("delete")]), Step(text=["ok"]))
        run(Agent(model, [t], "sys"), device=device)
        assert device.confirms == []  # don't bother the wearer with a request that can't run


def test_status_is_shown_while_a_tool_runs():
    device = FakeDevice()
    model = ScriptedModel(Step(calls=[call("lookup")]), Step(text=["ok"]))
    run(Agent(model, [tool(status="Checking email...")], "sys"), device=device)
    assert device.statuses == ["Checking email..."]


def test_bad_tool_definitions_fail_at_startup():
    with pytest.raises(ValueError, match="duplicate"):
        Agent(ScriptedModel(), [tool("a"), tool("a")], "sys")
    with pytest.raises(Exception, match="is not valid"):
        Agent(ScriptedModel(), [tool(parameters={"type": "not-a-type"})], "sys")


def test_trace_records_tool_calls_without_storing_their_output():
    async def secret(ctx, args):
        return "the email says: meet at 5pm"

    model = ScriptedModel(Step(calls=[call("lookup", query="sarah")]), Step(text=["ok"]))
    _, _, trace = run(Agent(model, [tool(run=secret, parameters={"type": "object"})], "sys"))
    tool_event = trace.events[1]
    assert (tool_event["name"], tool_event["args"], tool_event["error"]) == ("lookup", '{"query": "sarah"}', None)
    assert "5pm" not in str(trace.events)

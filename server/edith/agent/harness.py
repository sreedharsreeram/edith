"""The agent harness: a model, a set of tools, and the loop between them.

One turn: the model sees the photo and the question, may call tools (several at once), sees
their results, and repeats until it answers. Provider details (Gemini, Claude) live in
edith/models/; this file never imports either SDK.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import jsonschema

from .trace import Trace

log = logging.getLogger("edith")

REFUSAL_TEXT = "Can't help with that."
NO_ANSWER_TEXT = "Sorry, I couldn't work that out."
MAX_STEPS = 5  # model calls per turn; the last one has tools switched off so it has to answer
MAX_RESULT_CHARS = 8_000  # one tool can't flood the model's context (or the bill)


class BrainError(Exception):
    """A failure with a short message that's safe to show on the glasses."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class ToolError(Exception):
    """An expected tool failure. The message goes to the model, which can explain it or try something else."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    text: str
    images: list[bytes] = field(default_factory=list)  # JPEGs, e.g. frames from a walk-around
    is_error: bool = False


@dataclass(frozen=True)
class Exchange:
    """An earlier question and answer, so follow-ups like "when was it made?" make sense."""

    question: str
    answer: str


class Device(Protocol):
    """What tools can ask the glasses to do."""

    async def status(self, text: str) -> None: ...

    async def capture(self, *, instruction: str, countdown_s: int, seconds: float, fps: float) -> list[bytes]: ...

    async def confirm(self, question: str) -> bool: ...


@dataclass
class ToolContext:
    device: Device
    photo: bytes  # what the wearer was looking at when they pressed the button


@dataclass(frozen=True)
class Tool:
    name: str
    description: str  # the model decides when to use the tool from this, so write it for the model
    parameters: dict  # JSON Schema for the arguments; validated before run() is called
    run: Callable[[ToolContext, dict], Awaitable[ToolResult | str]]
    status: str | None = None  # shown on the glasses while it runs, e.g. "Checking email..."
    # For tools with side effects: builds the question the wearer must approve with a button press.
    # Text in photos, emails or web pages can try to steer the model; this makes sure it can't act alone.
    confirm: Callable[[dict], str] | None = None
    timeout_s: float = 15


OnText = Callable[[str], Awaitable[None]]


class Conversation(Protocol):
    """One turn's dialogue with a model. Adapters keep the provider's own message format inside."""

    model_name: str

    async def step(self, on_text: OnText, allow_tools: bool) -> list[ToolCall]:
        """Run the model once, streaming its text to on_text. Returns the tools it wants to call."""
        ...

    def add_results(self, calls: list[ToolCall], results: list[ToolResult]) -> None: ...


class Model(Protocol):
    name: str

    async def warm(self) -> None: ...

    def start(
        self, *, system: str, history: list[Exchange], photo: bytes, question: str, tools: list[Tool]
    ) -> Conversation: ...


class Agent:
    def __init__(self, model: Model, tools: list[Tool], system: str, max_steps: int = MAX_STEPS) -> None:
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tool names: {names}")
        for tool in tools:
            jsonschema.Draft202012Validator.check_schema(tool.parameters)  # fail at startup, not mid-question
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.system = system
        self.max_steps = max_steps

    async def run(
        self,
        *,
        question: str,
        photo: bytes,
        history: list[Exchange],
        device: Device,
        on_text: OnText,
        trace: Trace,
    ) -> str:
        conversation = self.model.start(
            system=self.system, history=history, photo=photo, question=question, tools=list(self.tools.values())
        )
        context = ToolContext(device=device, photo=photo)
        answer: list[str] = []

        async def emit(text: str) -> None:
            answer.append(text)
            await on_text(text)

        for step in range(1, self.max_steps + 1):
            allow_tools = step < self.max_steps
            started = time.perf_counter()
            calls = await conversation.step(emit, allow_tools=allow_tools)
            trace.event(
                "model",
                step=step,
                model=conversation.model_name,
                ms=_ms_since(started),
                tool_calls=[c.name for c in calls],
            )
            if not calls or not allow_tools:  # done (a model that calls tools anyway on the last step is ignored)
                break
            results = await asyncio.gather(*(self._call(c, context, trace) for c in calls))
            conversation.add_results(calls, list(results))

        if not "".join(answer).strip():
            await emit(NO_ANSWER_TEXT)
        return "".join(answer).strip()

    async def _call(self, call: ToolCall, context: ToolContext, trace: Trace) -> ToolResult:
        started = time.perf_counter()
        result = await self._execute(call, context)
        trace.event(
            "tool",
            name=call.name,
            args=_preview(call.args),
            ms=_ms_since(started),
            error=result.text if result.is_error else None,
            images=len(result.images),
        )
        if result.is_error:
            log.warning("tool %s: %s", call.name, result.text)
        return result

    async def _execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(f"There is no tool called {call.name!r}.", is_error=True)
        try:
            jsonschema.validate(call.args, tool.parameters)
        except jsonschema.ValidationError as e:
            return ToolResult(f"Invalid arguments: {e.message}", is_error=True)

        if tool.confirm and not await context.device.confirm(tool.confirm(call.args)):
            return ToolResult("The wearer declined, so nothing was done. Tell them it was cancelled.", is_error=True)
        if tool.status:
            await context.device.status(tool.status)

        try:
            async with asyncio.timeout(tool.timeout_s):
                result = await tool.run(context, call.args)
        except TimeoutError:
            return ToolResult(f"{call.name} took too long and was stopped.", is_error=True)
        except ToolError as e:
            return ToolResult(str(e), is_error=True)
        except Exception:
            log.exception("tool %s crashed", call.name)
            return ToolResult(f"{call.name} failed unexpectedly.", is_error=True)

        if isinstance(result, str):
            result = ToolResult(result)
        if len(result.text) > MAX_RESULT_CHARS:
            result.text = result.text[:MAX_RESULT_CHARS] + "\n[truncated]"
        return result


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _preview(args: dict, limit: int = 200) -> str:
    text = json.dumps(args, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."

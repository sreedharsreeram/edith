"""Fakes and helpers shared by the tests."""

import asyncio
import io
import json
from dataclasses import dataclass, field

from PIL import Image

from edith import protocol
from edith.agent.harness import Exchange, Tool, ToolCall, ToolResult

# Passes the JPEG magic-byte check; fakes never decode it.
TINY_JPEG = protocol.JPEG_MAGIC + b"\xff\xe0" + bytes(100) + b"\xff\xd9"


def speech(seconds: float = 1.0) -> bytes:
    """Stand-in PCM audio (fake transcribers don't listen to it)."""
    return b"\x01\x00" * int(protocol.SAMPLE_RATE * seconds)


def jpeg(color="gray", size=(64, 48)) -> bytes:
    """A real, decodable JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


class FakeTranscriber:
    def __init__(self, text: str = "what is this", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[bytes] = []

    def __call__(self, pcm: bytes) -> str:
        self.calls.append(pcm)
        if self.error:
            raise self.error
        return self.text


@dataclass
class Step:
    """One scripted model step: stream some text, then ask for some tools (or fail)."""

    text: list[str] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    delay: float = 0.0  # before each text chunk
    error: Exception | None = None


def call(name: str, call_id: str | None = None, **args) -> ToolCall:
    return ToolCall(id=call_id or f"{name}-1", name=name, args=args)


class ScriptedModel:
    """A model that follows a script, shared across turns: each step() takes the next Step."""

    name = "scripted"

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.conversations: list[ScriptedConversation] = []
        self.warmed = 0
        self.cancelled = 0

    async def warm(self) -> None:
        self.warmed += 1

    def start(self, *, system: str, history: list[Exchange], photo: bytes, question: str, tools: list[Tool]):
        conversation = ScriptedConversation(self, system, history, photo, question, tools)
        self.conversations.append(conversation)
        return conversation


class ScriptedConversation:
    model_name = "scripted"

    def __init__(self, model, system, history, photo, question, tools) -> None:
        self.model = model
        self.system, self.history, self.photo, self.question, self.tools = system, history, photo, question, tools
        self.allow_tools: list[bool] = []
        self.results: list[list[tuple[ToolCall, ToolResult]]] = []

    async def step(self, on_text, allow_tools: bool) -> list[ToolCall]:
        self.allow_tools.append(allow_tools)
        step = self.model.steps.pop(0) if self.model.steps else Step(text=["(script ran out)"])
        try:
            for chunk in step.text:
                if step.delay:
                    await asyncio.sleep(step.delay)
                await on_text(chunk)
            if step.error:
                raise step.error
        except asyncio.CancelledError:
            self.model.cancelled += 1
            raise
        return list(step.calls)

    def add_results(self, calls, results) -> None:
        self.results.append(list(zip(calls, results, strict=True)))


class FakeDevice:
    """Stands in for the glasses when testing the agent loop without a WebSocket."""

    def __init__(self, frames: list[bytes] | None = None, confirm: bool = True) -> None:
        self.frames = frames if frames is not None else [jpeg()]
        self.confirm_answer = confirm
        self.statuses: list[str] = []
        self.captures: list[dict] = []
        self.confirms: list[str] = []

    async def status(self, text: str) -> None:
        self.statuses.append(text)

    async def capture(self, **request) -> list[bytes]:
        self.captures.append(request)
        return list(self.frames)

    async def confirm(self, question: str) -> bool:
        self.confirms.append(question)
        return self.confirm_answer


def send_turn(ws, image: bytes | None = TINY_JPEG, audio: bytes = b"", chunk: int = 3200) -> None:
    """Do what the glasses do for one question: start, photo, audio chunks, end."""
    ws.send_text(json.dumps({"type": "start"}))
    if image is not None:
        ws.send_bytes(bytes([protocol.IMAGE]) + image)
    for i in range(0, len(audio), chunk):
        ws.send_bytes(bytes([protocol.AUDIO]) + audio[i : i + chunk])
    ws.send_text(json.dumps({"type": "end"}))


def receive_turn(ws) -> list[dict]:
    """Collect server messages until the answer finishes (done) or fails (error)."""
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["type"] in ("done", "error"):
            return msgs


def answer_text(msgs: list[dict]) -> str:
    return "".join(m["text"] for m in msgs if m["type"] == "delta")

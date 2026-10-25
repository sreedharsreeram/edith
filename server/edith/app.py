"""EDITH server: one WebSocket per pair of glasses. See PROTOCOL.md.

Run: uv run uvicorn edith.app:create_app --factory --host 0.0.0.0 --port 8000
"""

import asyncio
import hmac
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, status

from . import protocol
from .agent.harness import Agent, BrainError, Exchange, Model, Tool
from .agent.prompt import SYSTEM_PROMPT
from .agent.trace import Trace
from .device import DeviceLink

log = logging.getLogger("edith")

DEFAULT_QUESTION = "What am I looking at?"
MIN_SPEECH_BYTES = protocol.SAMPLE_RATE * 2 * 3 // 10  # under 0.3s of audio is a tap, not a question
HISTORY_TURNS = 4  # earlier exchanges the model sees, for follow-up questions (text only, no old photos)

Transcribe = Callable[[bytes], str]


def error(message: str) -> dict:
    return {"type": "error", "message": message}


def default_model() -> Model:
    """EDITH_BRAIN=claude|gemini picks explicitly; otherwise use whichever API key is set."""
    choice = os.environ.get("EDITH_BRAIN")
    if choice is None:
        has_gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        choice = "gemini" if has_gemini and not os.environ.get("ANTHROPIC_API_KEY") else "claude"
    if choice == "gemini":
        from .models.gemini import Gemini

        model = Gemini()
    elif choice == "claude":
        from .models.claude import Claude

        model = Claude()
    else:
        raise SystemExit(f"EDITH_BRAIN must be 'claude' or 'gemini', not {choice!r}")
    log.info("using %s (%s)", choice, model.name)
    return model


def create_app(
    transcribe: Transcribe | None = None,
    model: Model | None = None,
    tools: list[Tool] | None = None,
    token: str | None = None,
    data_dir: Path | None = None,
) -> FastAPI:
    """Build the server. Tests pass fakes; uvicorn calls this with no arguments to get the real thing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = data_dir or Path(os.environ.get("EDITH_DATA_DIR", "data"))
    if transcribe is None:
        from .stt import WhisperTranscriber

        transcribe = WhisperTranscriber()
    if tools is None:
        from .tools import default_tools

        tools = default_tools(data_dir)
    agent = Agent(model or default_model(), tools, SYSTEM_PROMPT)
    if token is None:
        token = os.environ.get("EDITH_TOKEN", "")
    if not token:
        log.warning("EDITH_TOKEN is not set: anyone on your network can use this server (and your API credits)")
    log.info("tools: %s", ", ".join(agent.tools))

    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.websocket("/ws")
    async def glasses(ws: WebSocket) -> None:
        if token and not hmac.compare_digest(ws.query_params.get("token", "").encode(), token.encode()):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            log.warning("rejected connection with a bad token")
            return
        await ws.accept()
        log.info("glasses connected")
        session = Session(ws, transcribe, agent, data_dir / "traces")
        try:
            await session.run()
        finally:
            session.cancel_turn()
            log.info("glasses disconnected")

    return app


class Session:
    """One connected pair of glasses. A turn = start, photo, audio..., end -> streamed answer."""

    def __init__(self, ws: WebSocket, transcribe: Transcribe, agent: Agent, trace_dir: Path) -> None:
        self.ws = ws
        self.transcribe = transcribe
        self.agent = agent
        self.trace_dir = trace_dir
        self.device = DeviceLink(self.send)
        self.history: deque[Exchange] = deque(maxlen=HISTORY_TURNS)
        self.image: bytes | None = None
        self.audio = bytearray()
        self.turn: asyncio.Task | None = None
        self.warming: asyncio.Task | None = None
        self.send_lock = asyncio.Lock()

    async def run(self) -> None:
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            try:
                frame = protocol.parse(msg)
            except protocol.ProtocolError as e:
                log.warning("bad frame: %s", e)
                await self.send(error(str(e)))
                continue
            self.handle(frame)

    def handle(self, frame: protocol.Frame) -> None:
        match frame:
            case protocol.Start():
                self.cancel_turn()  # pressing the button again means "never mind, new question"
                self.image, self.audio = None, bytearray()
                self.warming = asyncio.create_task(self.agent.model.warm())  # handshake while the wearer talks
            case protocol.Image(jpeg):
                self.image = jpeg
            case protocol.Audio(pcm):
                room = protocol.MAX_AUDIO_BYTES - len(self.audio)
                self.audio += pcm[:room]  # past the cap, answer using the first 30s
            case protocol.End():
                if self.turn and not self.turn.done():
                    return  # duplicate end; the answer is already on its way
                self.turn = asyncio.create_task(self.answer(self.image, bytes(self.audio)))
            case protocol.BurstFrame(jpeg):
                self.device.on_frame(jpeg)
            case protocol.CaptureDone(capture_id):
                self.device.on_capture_done(capture_id)
            case protocol.ConfirmResult(confirm_id, ok):
                self.device.on_confirm_result(confirm_id, ok)

    def cancel_turn(self) -> None:
        if self.turn and not self.turn.done():
            self.turn.cancel()
            log.info("turn cancelled")

    async def send(self, msg: dict) -> None:
        async with self.send_lock:
            try:
                await self.ws.send_json(msg)
            except Exception:  # noqa: BLE001, S110
                pass  # connection is gone; run() sees the disconnect and cleans up

    async def answer(self, image: bytes | None, audio: bytes) -> None:
        if image is None:
            await self.send(error("No photo received. Try again."))
            return

        trace = Trace()
        started = time.perf_counter()
        timings: dict[str, int] = {}
        question = answer = ""
        outcome = "cancelled"

        def ms() -> int:
            return round((time.perf_counter() - started) * 1000)

        async def on_text(text: str) -> None:
            timings.setdefault("first_word", ms())
            await self.send({"type": "delta", "text": text})

        try:
            if len(audio) >= MIN_SPEECH_BYTES:
                question = await asyncio.to_thread(self.transcribe, audio)
            question = question or DEFAULT_QUESTION
            timings["stt"] = ms()
            await self.send({"type": "transcript", "text": question})
            log.info("[%s] heard: %r", trace.id, question)

            answer = await self.agent.run(
                question=question,
                photo=image,
                history=list(self.history),
                device=self.device,
                on_text=on_text,
                trace=trace,
            )
            timings["total"] = ms()
            self.history.append(Exchange(question, answer))
            outcome = "ok"
            await self.send({"type": "done", "timings": timings})
            log.info("[%s] answer: %r  %s", trace.id, answer, timings)
        except BrainError as e:
            outcome = f"error: {e.user_message}"
            log.warning("brain error: %s (%r)", e.user_message, e.__cause__)
            await self.send(error(e.user_message))
        except Exception:
            outcome = "crashed"
            log.exception("turn failed")
            await self.send(error("Something went wrong."))
        finally:
            trace.write(self.trace_dir, question=question, answer=answer, outcome=outcome, timings=timings)

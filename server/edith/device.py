"""The server's side of asking the glasses to do something: show a status, record a burst, confirm.

Requests go out as JSON; replies come back through Session.run(), which hands them to on_*().
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable

from .agent.harness import ToolError

log = logging.getLogger("edith")

CAPTURE_GRACE_S = 5  # upload slack after the glasses stop recording
CONFIRM_TIMEOUT_S = 10  # no press in time means no
MAX_BURST_FRAMES = 60


class DeviceLink:
    def __init__(self, send: Callable[[dict], Awaitable[None]]) -> None:
        self.send = send
        self.lock = asyncio.Lock()  # one camera, one screen: one capture or confirmation at a time
        self._capture: tuple[str, list[bytes], asyncio.Future] | None = None
        self._confirm: tuple[str, asyncio.Future] | None = None

    async def status(self, text: str) -> None:
        await self.send({"type": "status", "text": text})

    async def capture(self, *, instruction: str, countdown_s: int, seconds: float, fps: float) -> list[bytes]:
        async with self.lock:
            capture_id = uuid.uuid4().hex[:8]
            frames: list[bytes] = []
            done = asyncio.get_running_loop().create_future()
            self._capture = (capture_id, frames, done)
            try:
                await self.send({
                    "type": "capture",
                    "id": capture_id,
                    "text": instruction,
                    "countdown": countdown_s,
                    "seconds": seconds,
                    "fps": fps,
                })
                try:
                    await asyncio.wait_for(done, countdown_s + seconds + CAPTURE_GRACE_S)
                except TimeoutError:
                    log.warning("capture %s: glasses never said done; using the %d frames received", capture_id, len(frames))
            finally:
                self._capture = None
            if not frames:
                raise ToolError("The glasses didn't send any frames. The camera may be unavailable.")
            return frames

    async def confirm(self, question: str) -> bool:
        async with self.lock:
            confirm_id = uuid.uuid4().hex[:8]
            answer = asyncio.get_running_loop().create_future()
            self._confirm = (confirm_id, answer)
            try:
                await self.send({"type": "confirm", "id": confirm_id, "text": question, "timeout": CONFIRM_TIMEOUT_S})
                return await asyncio.wait_for(answer, CONFIRM_TIMEOUT_S + 1)
            except TimeoutError:
                return False
            finally:
                self._confirm = None

    def on_frame(self, jpeg: bytes) -> None:
        if self._capture is None:
            return  # a late frame from a capture that already ended
        _, frames, _ = self._capture
        if len(frames) < MAX_BURST_FRAMES:
            frames.append(jpeg)

    def on_capture_done(self, capture_id: str) -> None:
        if self._capture and self._capture[0] == capture_id and not self._capture[2].done():
            self._capture[2].set_result(None)

    def on_confirm_result(self, confirm_id: str, ok: bool) -> None:
        if self._confirm and self._confirm[0] == confirm_id and not self._confirm[1].done():
            self._confirm[1].set_result(ok)

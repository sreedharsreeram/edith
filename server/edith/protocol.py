"""Wire format between the glasses and the server."""

import json
from dataclasses import dataclass

IMAGE, AUDIO, BURST = 0x01, 0x02, 0x03  # BURST: a frame of a capture the server asked for
SAMPLE_RATE = 16_000
JPEG_MAGIC = b"\xff\xd8"
MAX_IMAGE_BYTES = 1_000_000  # a VGA JPEG is ~30-80KB; anything near 1MB is a bug or abuse
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * 30  # 30s of 16-bit mono


class ProtocolError(Exception):
    pass


@dataclass(frozen=True)
class Start:
    pass


@dataclass(frozen=True)
class End:
    pass


@dataclass(frozen=True)
class Image:
    jpeg: bytes


@dataclass(frozen=True)
class Audio:
    pcm: bytes


@dataclass(frozen=True)
class BurstFrame:
    jpeg: bytes


@dataclass(frozen=True)
class CaptureDone:
    id: str


@dataclass(frozen=True)
class ConfirmResult:
    id: str
    ok: bool


Frame = Start | End | Image | Audio | BurstFrame | CaptureDone | ConfirmResult


def parse(msg: dict) -> Frame:
    """Turn a raw ASGI websocket message into a Frame, or raise ProtocolError."""
    data = msg.get("bytes")
    if data is not None:
        if not data:
            raise ProtocolError("empty binary frame")
        tag, payload = data[0], data[1:]
        if tag in (IMAGE, BURST):
            if len(payload) > MAX_IMAGE_BYTES:
                raise ProtocolError("photo too large")
            if not payload.startswith(JPEG_MAGIC):
                raise ProtocolError("photo is not a JPEG")
            return Image(payload) if tag == IMAGE else BurstFrame(payload)
        if tag == AUDIO:
            return Audio(payload)
        raise ProtocolError(f"unknown frame tag 0x{tag:02x}")

    try:
        event = json.loads(msg.get("text") or "")
    except ValueError:
        raise ProtocolError("text frame is not JSON") from None
    kind = event.get("type") if isinstance(event, dict) else None
    if kind == "start":
        return Start()
    if kind == "end":
        return End()
    if kind == "capture_done" and isinstance(event.get("id"), str):
        return CaptureDone(event["id"])
    if kind == "confirm_result" and isinstance(event.get("id"), str) and isinstance(event.get("ok"), bool):
        return ConfirmResult(event["id"], event["ok"])
    if kind in ("capture_done", "confirm_result"):
        raise ProtocolError(f"malformed {kind}")
    raise ProtocolError(f"unknown event type {kind!r}")

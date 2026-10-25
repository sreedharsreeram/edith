import json

import pytest
from helpers import TINY_JPEG

from server import protocol
from server.protocol import Audio, End, Image, ProtocolError, Start, parse


def text(obj) -> dict:
    return {"type": "websocket.receive", "text": json.dumps(obj)}


def binary(data: bytes) -> dict:
    return {"type": "websocket.receive", "bytes": data}


def test_control_events():
    assert parse(text({"type": "start"})) == Start()
    assert parse(text({"type": "end"})) == End()


def test_image_and_audio():
    assert parse(binary(b"\x01" + TINY_JPEG)) == Image(TINY_JPEG)
    assert parse(binary(b"\x02" + b"\x00\x01")) == Audio(b"\x00\x01")
    assert parse(binary(b"\x02")) == Audio(b"")  # empty chunk is harmless


@pytest.mark.parametrize(
    ("msg", "reason"),
    [
        (binary(b""), "empty binary frame"),
        (binary(b"\x09abc"), "unknown frame tag 0x09"),
        (binary(b"\x01not a jpeg"), "not a JPEG"),
        (binary(b"\x01" + protocol.JPEG_MAGIC + bytes(protocol.MAX_IMAGE_BYTES)), "too large"),
        ({"type": "websocket.receive", "text": "{oops"}, "not JSON"),
        ({"type": "websocket.receive", "text": None}, "not JSON"),
        (text(["start"]), "unknown event type None"),
        (text({"type": "reboot"}), "unknown event type 'reboot'"),
        (text({}), "unknown event type None"),
    ],
)
def test_rejects_bad_frames(msg, reason):
    with pytest.raises(ProtocolError, match=reason):
        parse(msg)


def test_device_replies():
    from server.protocol import BurstFrame, CaptureDone, ConfirmResult

    assert parse(binary(b"\x03" + TINY_JPEG)) == BurstFrame(TINY_JPEG)
    assert parse(text({"type": "capture_done", "id": "c1"})) == CaptureDone("c1")
    assert parse(text({"type": "confirm_result", "id": "k1", "ok": False})) == ConfirmResult("k1", False)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "capture_done"},
        {"type": "capture_done", "id": 5},
        {"type": "confirm_result", "id": "k1"},
        {"type": "confirm_result", "id": "k1", "ok": "yes"},  # must be a real boolean, not truthy text
    ],
)
def test_malformed_device_replies(event):
    with pytest.raises(ProtocolError, match="malformed"):
        parse(text(event))


def test_burst_frame_must_be_jpeg():
    with pytest.raises(ProtocolError, match="not a JPEG"):
        parse(binary(b"\x03not a jpeg"))

"""Server-initiated device requests (capture bursts, confirmations) over a real WebSocket."""

import json

import pytest
from helpers import ScriptedModel, Step, call, jpeg, receive_turn, send_turn, speech

from server import device, protocol
from server.agent.core import Tool, ToolContext, ToolResult

BURST = bytes([protocol.BURST])


async def capture_tool(ctx: ToolContext, args: dict) -> ToolResult:
    frames = await ctx.device.capture(instruction="Walk around it", countdown_s=0, seconds=1, fps=2)
    return ToolResult(f"{len(frames)} frames", images=frames)


async def confirm_tool(ctx: ToolContext, args: dict) -> str:
    return "approved" if await ctx.device.confirm("Really?") else "declined"


CAPTURE = Tool(name="capture", description="d", parameters={"type": "object"}, run=capture_tool, timeout_s=30)
CONFIRM = Tool(name="ask", description="d", parameters={"type": "object"}, run=confirm_tool)


def receive_until(ws, kind: str) -> dict:
    while (msg := ws.receive_json())["type"] != kind:
        pass
    return msg


def tool_results(model: ScriptedModel):
    return [result for step in model.conversations[0].results for _, result in step]


def test_capture_burst_reaches_the_tool(make_client):
    model = ScriptedModel(Step(calls=[call("capture")]), Step(text=["Nice car."]))
    frames = [jpeg("red"), jpeg("blue"), jpeg("green")]
    with make_client(model, [CAPTURE]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        request = receive_until(ws, "capture")
        assert (request["text"], request["countdown"], request["seconds"], request["fps"]) == ("Walk around it", 0, 1, 2)
        for frame in frames:
            ws.send_bytes(BURST + frame)
        ws.send_text(json.dumps({"type": "capture_done", "id": request["id"]}))
        msgs = receive_turn(ws)
    assert msgs[-1]["type"] == "done"
    [result] = tool_results(model)
    assert result.images == frames


def test_capture_uses_frames_received_if_glasses_never_say_done(make_client, monkeypatch):
    monkeypatch.setattr(device, "CAPTURE_GRACE_S", 0.2)
    model = ScriptedModel(Step(calls=[call("capture")]), Step(text=["ok"]))
    with make_client(model, [CAPTURE]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_until(ws, "capture")
        ws.send_bytes(BURST + jpeg())
        receive_turn(ws)
    [result] = tool_results(model)
    assert len(result.images) == 1


def test_capture_with_no_frames_is_a_tool_error(make_client, monkeypatch):
    monkeypatch.setattr(device, "CAPTURE_GRACE_S", 0.1)
    model = ScriptedModel(Step(calls=[call("capture")]), Step(text=["Camera's not working."]))
    with make_client(model, [CAPTURE]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_until(ws, "capture")
        msgs = receive_turn(ws)
    assert msgs[-1]["type"] == "done"  # the turn survives; the model explains
    [result] = tool_results(model)
    assert result.is_error and "didn't send any frames" in result.text


def test_frames_for_a_finished_capture_are_ignored(make_client):
    model = ScriptedModel(Step(calls=[call("capture")]), Step(calls=[call("capture")]), Step(text=["ok"]))
    with make_client(model, [CAPTURE]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        first = receive_until(ws, "capture")
        ws.send_bytes(BURST + jpeg("red"))
        ws.send_text(json.dumps({"type": "capture_done", "id": first["id"]}))
        second = receive_until(ws, "capture")
        ws.send_text(json.dumps({"type": "capture_done", "id": first["id"]}))  # stale: must not end the second
        ws.send_bytes(BURST + jpeg("blue"))
        ws.send_text(json.dumps({"type": "capture_done", "id": second["id"]}))
        receive_turn(ws)
    first_result, second_result = tool_results(model)
    assert first_result.images == [jpeg("red")]
    assert second_result.images == [jpeg("blue")]


def test_burst_is_capped(make_client):
    model = ScriptedModel(Step(calls=[call("capture")]), Step(text=["ok"]))
    with make_client(model, [CAPTURE]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        request = receive_until(ws, "capture")
        for _ in range(device.MAX_BURST_FRAMES + 10):
            ws.send_bytes(BURST + jpeg())
        ws.send_text(json.dumps({"type": "capture_done", "id": request["id"]}))
        receive_turn(ws)
    [result] = tool_results(model)
    assert len(result.images) == device.MAX_BURST_FRAMES


@pytest.mark.parametrize(("ok", "expected"), [(True, "approved"), (False, "declined")])
def test_confirm_round_trip(make_client, ok, expected):
    model = ScriptedModel(Step(calls=[call("ask")]), Step(text=["ok"]))
    with make_client(model, [CONFIRM]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        request = receive_until(ws, "confirm")
        assert request["text"] == "Really?"
        ws.send_text(json.dumps({"type": "confirm_result", "id": request["id"], "ok": ok}))
        receive_turn(ws)
    [result] = tool_results(model)
    assert result.text == expected


def test_no_answer_to_confirm_means_no(make_client, monkeypatch):
    monkeypatch.setattr(device, "CONFIRM_TIMEOUT_S", 0.1)
    model = ScriptedModel(Step(calls=[call("ask")]), Step(text=["ok"]))
    with make_client(model, [CONFIRM]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_until(ws, "confirm")
        receive_turn(ws)
    [result] = tool_results(model)
    assert result.text == "declined"


def test_confirm_for_a_different_request_is_ignored(make_client, monkeypatch):
    monkeypatch.setattr(device, "CONFIRM_TIMEOUT_S", 0.2)
    model = ScriptedModel(Step(calls=[call("ask")]), Step(text=["ok"]))
    with make_client(model, [CONFIRM]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_until(ws, "confirm")
        ws.send_text(json.dumps({"type": "confirm_result", "id": "forged", "ok": True}))
        receive_turn(ws)
    [result] = tool_results(model)
    assert result.text == "declined"


def test_status_is_sent_to_the_glasses(make_client):
    async def noop(ctx, args):
        return "ok"

    t = Tool(name="email", description="d", parameters={"type": "object"}, run=noop, status="Checking email...")
    model = ScriptedModel(Step(calls=[call("email")]), Step(text=["ok"]))
    with make_client(model, [t]).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        msgs = receive_turn(ws)
    assert {"type": "status", "text": "Checking email..."} in msgs

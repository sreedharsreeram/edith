"""The server end to end over a real WebSocket, with fake Whisper and Claude."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from helpers import TINY_JPEG, FakeTranscriber, ScriptedModel, Step, answer_text, receive_turn, send_turn, speech
from starlette.websockets import WebSocketDisconnect

from server import protocol
from server.agent.core import BrainError
from server.app import DEFAULT_QUESTION, create_app


def test_full_turn(client, stt, model):
    audio = speech(1.5)
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=audio)
        msgs = receive_turn(ws)

    assert msgs[0] == {"type": "transcript", "text": "what is this"}
    assert answer_text(msgs) == "A red apple."
    assert msgs[-1]["type"] == "done"
    assert set(msgs[-1]["timings"]) == {"stt", "first_word", "total"}
    assert stt.calls == [audio]  # every chunk arrived, in order
    conversation = model.conversations[0]
    assert (conversation.photo, conversation.question) == (TINY_JPEG, "what is this")


def test_button_press_warms_the_api_connection(client, model):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_turn(ws)
    assert model.warmed == 1


@pytest.mark.parametrize("audio", [b"", speech(0.1)])
def test_quick_tap_asks_the_default_question(client, stt, model, audio):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=audio)
        msgs = receive_turn(ws)
    assert msgs[0]["text"] == DEFAULT_QUESTION
    assert stt.calls == []  # too short to be worth transcribing
    assert model.conversations[0].question == DEFAULT_QUESTION


def test_silence_falls_back_to_default_question(client, stt):
    stt.text = ""
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        msgs = receive_turn(ws)
    assert msgs[0]["text"] == DEFAULT_QUESTION


def test_missing_photo_is_an_error_and_connection_survives(client):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, image=None, audio=speech())
        assert receive_turn(ws) == [{"type": "error", "message": "No photo received. Try again."}]
        send_turn(ws, audio=speech())
        assert receive_turn(ws)[-1]["type"] == "done"


def test_bad_frame_is_reported_and_connection_survives(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{garbage")
        assert ws.receive_json() == {"type": "error", "message": "text frame is not JSON"}
        ws.send_bytes(b"\x01not a jpeg")
        assert ws.receive_json()["type"] == "error"
        send_turn(ws, audio=speech())
        assert receive_turn(ws)[-1]["type"] == "done"


def test_photo_from_previous_turn_is_not_reused(client):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_turn(ws)
        send_turn(ws, image=None, audio=speech())
        assert receive_turn(ws)[0]["type"] == "error"


def test_audio_is_capped(client, stt):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech(35), chunk=64_000)
        receive_turn(ws)
    assert len(stt.calls[0]) == protocol.MAX_AUDIO_BYTES


def test_odd_sized_audio_chunks_are_reassembled_exactly(client, stt):
    audio = speech(1)
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=audio, chunk=333)
        receive_turn(ws)
    assert stt.calls == [audio]


def test_pressing_again_cancels_the_current_answer(make_client):
    model = ScriptedModel(Step(text=["word "] * 50, delay=0.05), Step(text=["second answer"]))
    with make_client(model).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        assert ws.receive_json()["type"] == "transcript"
        assert ws.receive_json()["type"] == "delta"  # first answer is mid-stream...
        send_turn(ws, audio=speech())  # ...when the wearer asks something new
        msgs = receive_turn(ws)

    assert model.cancelled == 1
    assert msgs[-1]["type"] == "done"
    assert msgs[-2] == {"type": "delta", "text": "second answer"}
    assert sum(m["type"] == "done" for m in msgs) == 1  # the cancelled answer never "finished"


def test_duplicate_end_is_ignored(make_client):
    model = ScriptedModel(Step(text=["one"], delay=0.05), Step(text=["two"]))
    with make_client(model).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        ws.send_text(json.dumps({"type": "end"}))
        receive_turn(ws)
        send_turn(ws, audio=speech())
        assert ws.receive_json()["type"] == "transcript"  # not a second answer to the first question
    assert len(model.conversations) == 2


def test_brain_failure_shows_friendly_message(make_client):
    model = ScriptedModel(Step(error=BrainError("Busy right now. Try again.")), Step(text=["ok"]))
    with make_client(model).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        assert receive_turn(ws)[-1] == {"type": "error", "message": "Busy right now. Try again."}
        send_turn(ws, audio=speech())
        assert receive_turn(ws)[-1]["type"] == "done"


def test_unexpected_failure_does_not_leak_details(model, tmp_path):
    stt = FakeTranscriber(error=RuntimeError("secret internal path /Users/x"))
    with TestClient(create_app(transcribe=stt, model=model, tools=[], token="", data_dir=tmp_path)) as client:
        with client.websocket_connect("/ws") as ws:
            send_turn(ws, audio=speech())
            assert receive_turn(ws)[-1] == {"type": "error", "message": "Something went wrong."}


def test_disconnect_mid_answer_cancels_the_work(make_client):
    model = ScriptedModel(Step(text=["word "] * 100, delay=0.05))
    with make_client(model).websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        ws.receive_json()
        ws.receive_json()
    deadline = time.monotonic() + 3
    while not model.cancelled and time.monotonic() < deadline:
        time.sleep(0.02)
    assert model.cancelled == 1  # we stopped paying for tokens nobody will see


def test_follow_up_questions_see_earlier_answers(client, stt, model):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_turn(ws)
        stt.text = "when was it made"
        send_turn(ws, audio=speech())
        receive_turn(ws)
    first, second = model.conversations
    assert first.history == []
    assert [(e.question, e.answer) for e in second.history] == [("what is this", "A red apple.")]


def test_history_is_capped(client, model):
    with client.websocket_connect("/ws") as ws:
        for _ in range(6):
            send_turn(ws, audio=speech())
            receive_turn(ws)
    assert len(model.conversations[-1].history) == 4


def test_every_turn_is_traced(client, tmp_path):
    with client.websocket_connect("/ws") as ws:
        send_turn(ws, audio=speech())
        receive_turn(ws)
    [trace_file] = (tmp_path / "traces").iterdir()
    [line] = trace_file.read_text().splitlines()
    trace = json.loads(line)
    assert (trace["question"], trace["answer"], trace["outcome"]) == ("what is this", "A red apple.", "ok")
    assert [e["kind"] for e in trace["events"]] == ["model"]


class TestAuth:
    @pytest.fixture
    def client(self, make_client, model):
        return make_client(model, token="s3cret")

    @pytest.mark.parametrize("path", ["/ws", "/ws?token=wrong", "/ws?token=", "/ws?token=s3cret%C3%A9"])
    def test_rejects_bad_tokens(self, client, path):
        with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect(path):
            pass
        assert e.value.code == 1008

    def test_accepts_the_right_token(self, client):
        with client.websocket_connect("/ws?token=s3cret") as ws:
            send_turn(ws, audio=speech())
            assert receive_turn(ws)[-1]["type"] == "done"


def test_health(client):
    assert client.get("/health").json() == {"ok": True}

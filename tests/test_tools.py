"""The built-in tools, run through the real agent loop with a scripted model."""

import asyncio

import pytest
from helpers import FakeDevice, ScriptedModel, Step, call, jpeg

from server.agent.core import Agent, ToolError
from server.agent.prompt import SYSTEM_PROMPT
from server.agent.trace import Trace
from server.tools import default_tools
from server.tools.camera import MAX_VIEWS
from server.tools.notes import NoteStore


def run_tools(tmp_path, *calls, device=None) -> list:
    """Run one turn where the model makes these calls in order, then return each tool result."""
    model = ScriptedModel(*[Step(calls=[c]) for c in calls], Step(text=["ok"]))
    agent = Agent(model, default_tools(tmp_path), SYSTEM_PROMPT, max_steps=len(calls) + 1)

    async def ignore(text):
        pass

    asyncio.run(
        agent.run(question="q", photo=jpeg(), history=[], device=device or FakeDevice(), on_text=ignore, trace=Trace())
    )
    return [result for step in model.conversations[0].results for _, result in step]


class TestNotes:
    def test_remember_then_recall(self, tmp_path):
        saved, notes = run_tools(tmp_path, call("remember", text="Parked on level 3, spot B12"), call("recall"))
        assert saved.text == "Saved as note #1."
        assert "#1" in notes.text and "level 3, spot B12" in notes.text

    def test_notes_survive_a_restart(self, tmp_path):
        run_tools(tmp_path, call("remember", text="Locker 42"))
        [notes] = run_tools(tmp_path, call("recall"))  # fresh tools, same data dir
        assert "Locker 42" in notes.text

    def test_recall_lists_newest_first(self, tmp_path):
        results = run_tools(tmp_path, call("remember", text="old"), call("remember", text="new"), call("recall"))
        assert results[-1].text.index("new") < results[-1].text.index("old")

    def test_recall_when_empty(self, tmp_path):
        [notes] = run_tools(tmp_path, call("recall"))
        assert notes.text == "There are no saved notes."

    def test_forget_asks_first_and_shows_the_note(self, tmp_path):
        device = FakeDevice(confirm=True)
        _, deleted, notes = run_tools(
            tmp_path, call("remember", text="Gate code 1234"), call("forget", id=1), call("recall"), device=device
        )
        assert device.confirms == ["Delete note: Gate code 1234?"]
        assert deleted.text == "Deleted note #1."
        assert notes.text == "There are no saved notes."

    def test_forget_declined_keeps_the_note(self, tmp_path):
        _, declined, notes = run_tools(
            tmp_path, call("remember", text="keep me"), call("forget", id=1), call("recall"), device=FakeDevice(confirm=False)
        )
        assert declined.is_error
        assert "keep me" in notes.text

    def test_forget_missing_note(self, tmp_path):
        [result] = run_tools(tmp_path, call("forget", id=9))
        assert result.is_error and "no note #9" in result.text

    def test_corrupt_notes_file_is_not_overwritten(self, tmp_path):
        (tmp_path / "notes.json").write_text("{not json")
        [result] = run_tools(tmp_path, call("remember", text="new"))
        assert result.is_error and "unreadable" in result.text
        assert (tmp_path / "notes.json").read_text() == "{not json"  # the user's data is still there to fix

    def test_overlong_note_is_rejected(self, tmp_path):
        [result] = run_tools(tmp_path, call("remember", text="x" * 501))
        assert result.is_error and "Invalid arguments" in result.text

    def test_store_rejects_non_list_file(self, tmp_path):
        (tmp_path / "notes.json").write_text('{"a": 1}')
        with pytest.raises(ToolError):
            NoteStore(tmp_path / "notes.json").load()


class TestCamera:
    def test_walk_around_sends_the_best_views(self, tmp_path):
        frames = [jpeg((i * 9 % 255, i * 37 % 255, i * 71 % 255)) for i in range(20)]
        device = FakeDevice(frames=frames)
        [result] = run_tools(tmp_path, call("walk_around", instruction="Walk around the car", seconds=10), device=device)
        assert device.captures == [{"instruction": "Walk around the car", "countdown_s": 3, "seconds": 10, "fps": 2}]
        assert len(result.images) == MAX_VIEWS
        assert "20 frames captured" in result.text

    def test_walk_around_defaults_its_length(self, tmp_path):
        device = FakeDevice()
        run_tools(tmp_path, call("walk_around", instruction="Walk around it"), device=device)
        assert device.captures[0]["seconds"] == 12

    def test_walk_around_rejects_silly_lengths(self, tmp_path):
        [result] = run_tools(tmp_path, call("walk_around", instruction="go", seconds=600))
        assert result.is_error

    def test_instruction_must_fit_the_screen(self, tmp_path):
        [result] = run_tools(tmp_path, call("take_photo", instruction="x" * 41))
        assert result.is_error

    def test_take_photo_returns_one_image(self, tmp_path):
        device = FakeDevice(frames=[jpeg("red"), jpeg("blue")])
        [result] = run_tools(tmp_path, call("take_photo", instruction="Look at the plaque"), device=device)
        assert len(result.images) == 1


def test_get_time(tmp_path):
    [result] = run_tools(tmp_path, call("get_time"))
    assert any(day in result.text for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

"""The wearer's personal notes: "remember I parked on level 3", "where did I park?"."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from ..agent.core import Tool, ToolContext, ToolError

log = logging.getLogger("edith")

MAX_NOTES_RETURNED = 50


class NoteStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict]:
        try:
            notes = json.loads(self.path.read_text())
        except FileNotFoundError:
            return []
        except (ValueError, OSError) as e:
            raise ToolError("The notes file is unreadable, so notes are unavailable right now.") from e
        if not isinstance(notes, list):
            raise ToolError("The notes file is unreadable, so notes are unavailable right now.")
        return notes

    def save(self, notes: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(notes, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)  # atomic: a crash mid-write can't corrupt the notes

    def add(self, text: str) -> dict:
        notes = self.load()
        note = {
            "id": max((n["id"] for n in notes), default=0) + 1,
            "text": text,
            "at": datetime.now().astimezone().isoformat(timespec="minutes"),
        }
        self.save([*notes, note])
        return note

    def get(self, note_id: int) -> dict | None:
        return next((n for n in self.load() if n["id"] == note_id), None)

    def remove(self, note_id: int) -> dict | None:
        notes = self.load()
        note = next((n for n in notes if n["id"] == note_id), None)
        if note:
            self.save([n for n in notes if n["id"] != note_id])
        return note


def note_tools(store: NoteStore) -> list[Tool]:
    async def remember(ctx: ToolContext, args: dict) -> str:
        note = store.add(args["text"])
        return f"Saved as note #{note['id']}."

    async def recall(ctx: ToolContext, args: dict) -> str:
        notes = store.load()[-MAX_NOTES_RETURNED:][::-1]
        if not notes:
            return "There are no saved notes."
        return "Saved notes, newest first:\n" + "\n".join(f"#{n['id']} ({n['at']}): {n['text']}" for n in notes)

    async def forget(ctx: ToolContext, args: dict) -> str:
        note = store.remove(args["id"])
        if note is None:
            raise ToolError(f"There's no note #{args['id']}. Use recall to see the notes.")
        return f"Deleted note #{note['id']}."

    def confirm_forget(args: dict) -> str:
        try:
            note = store.get(args["id"])
        except ToolError:
            note = None
        return f"Delete note: {note['text'][:40]}?" if note else f"Delete note #{args['id']}?"

    return [
        Tool(
            name="remember",
            description="Save a note for the wearer, e.g. where they parked or a name they want to remember.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 500}},
                "required": ["text"],
                "additionalProperties": False,
            },
            run=remember,
            status="Saving note...",
        ),
        Tool(
            name="recall",
            description="Get the wearer's saved notes, newest first, to answer questions like 'where did I park?'.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            run=recall,
            status="Checking notes...",
        ),
        Tool(
            name="forget",
            description="Delete one saved note by its number. Use recall first to find the number.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "integer", "minimum": 1}},
                "required": ["id"],
                "additionalProperties": False,
            },
            run=forget,
            confirm=confirm_forget,
        ),
    ]

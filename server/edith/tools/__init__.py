"""Every tool EDITH can use. To add a skill: write a Tool, add it to default_tools(), add tests."""

from pathlib import Path

from ..agent.harness import Tool
from .camera import CAMERA_TOOLS
from .clock import CLOCK_TOOLS
from .notes import NoteStore, note_tools


def default_tools(data_dir: Path) -> list[Tool]:
    return [*CAMERA_TOOLS, *note_tools(NoteStore(data_dir / "notes.json")), *CLOCK_TOOLS]

"""What the OLED shows. No pygame here, so it's testable; the firmware will mirror this logic."""

import textwrap

COLS, ROWS = 21, 8  # what a 128x64 OLED fits with a 6x8 pixel font


def wrap(text: str) -> list[str]:
    return [row for para in text.split("\n") for row in (textwrap.wrap(para, COLS) or [""])]


class Screen:
    """The question on top, then either a status line (while EDITH works) or the streaming answer."""

    def __init__(self) -> None:
        self.question = ""
        self.status = "EDITH ready\nHold SPACE to talk"
        self.answer = ""

    @property
    def text(self) -> str:
        top = f"> {self.question}\n" if self.question else ""
        return top + (self.answer or self.status)

    def show(self, status: str) -> None:
        """A local status ("Listening...") that starts a fresh screen."""
        self.question, self.answer, self.status = "", "", status

    def handle(self, msg: dict) -> dict | None:
        """Apply a server message. Returns the timings when an answer finishes."""
        match msg.get("type"):
            case "transcript":
                self.question, self.answer, self.status = msg["text"], "", "Thinking..."
            case "status":
                self.status = msg["text"]
            case "delta":
                self.answer += msg["text"]
            case "error":
                self.question, self.answer, self.status = "", "", f"ERROR: {msg['message']}"
            case "done":
                return msg["timings"]
        return None

    def lines(self) -> list[str]:
        return wrap(self.text)[-ROWS:]  # auto-scroll: keep the newest lines visible

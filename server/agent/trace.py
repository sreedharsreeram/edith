"""A record of one turn: every model step and tool call, with timings.

Written to data/traces/YYYY-MM-DD.jsonl (one line per turn) so you can see why the agent did
what it did. Arguments are truncated and images, emails and other tool output are never stored.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger("edith")


class Trace:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.at = datetime.now().astimezone()
        self.started = time.perf_counter()
        self.events: list[dict] = []

    def event(self, kind: str, **fields) -> None:
        self.events.append({"t_ms": round((time.perf_counter() - self.started) * 1000), "kind": kind, **fields})

    def write(self, directory: Path, **summary) -> None:
        line = {"id": self.id, "at": self.at.isoformat(timespec="seconds"), **summary, "events": self.events}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{self.at:%Y-%m-%d}.jsonl").open("a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("couldn't write trace: %s", e)  # never fail a turn over logging

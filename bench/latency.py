"""Latency benchmark against a running server: where does the time go?

Usage: uv run python bench/latency.py [--url ws://localhost:8000/ws] [--runs 10] [--image photo.jpg]
Costs real API money per run (a fraction of a cent each).
"""

import argparse
import io
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 3200  # 100ms of 16 kHz 16-bit audio


def send_turn(ws, image: bytes, audio: bytes) -> None:
    ws.send(json.dumps({"type": "start"}))
    ws.send(b"\x01" + image)
    for i in range(0, len(audio), CHUNK):
        ws.send(b"\x02" + audio[i : i + CHUNK])
    ws.send(json.dumps({"type": "end"}))


def receive_turn(ws) -> list[dict]:
    msgs = [json.loads(ws.recv())]
    while msgs[-1]["type"] not in ("done", "error"):
        msgs.append(json.loads(ws.recv()))
    return msgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--image", type=Path, help="JPEG to ask about (default: a plain gray frame)")
    parser.add_argument("--audio", default="plant", help="fixture name in tests/fixtures/audio")
    args = parser.parse_args()

    if args.image:
        image = args.image.read_bytes()
    else:
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), "gray").save(buf, "JPEG")
        image = buf.getvalue()
    with wave.open(str(ROOT / "tests" / "fixtures" / "audio" / f"{args.audio}.wav")) as w:
        audio = w.readframes(w.getnframes())

    token = os.environ.get("EDITH_TOKEN")
    url = f"{args.url}?token={quote(token)}" if token else args.url
    results: dict[str, list[int]] = {"stt": [], "first_word": [], "total": [], "client_total": []}
    with connect(url, max_size=None) as ws:
        for i in range(args.runs):
            sent = time.perf_counter()
            send_turn(ws, image=image, audio=audio)
            msgs = receive_turn(ws)
            if msgs[-1]["type"] == "error":
                sys.exit(f"run {i}: {msgs[-1]['message']}")
            for key, value in msgs[-1]["timings"].items():
                results[key].append(value)
            results["client_total"].append(round((time.perf_counter() - sent) * 1000))
            print(f"run {i + 1}/{args.runs}: {msgs[-1]['timings']}")

    print(f"\n{'stage':<14}{'p50':>8}{'p95':>8}{'max':>8}   (ms, measured from button release)")
    for key, values in results.items():
        values.sort()
        p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
        print(f"{key:<14}{statistics.median(values):>8.0f}{p95:>8}{values[-1]:>8}")


if __name__ == "__main__":
    main()

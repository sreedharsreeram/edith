"""Virtual EDITH glasses: behaves like the ESP32 firmware will, minus the soldering.

Runs real turns against the real server: takes a photo, "holds the button" while a spoken question
plays into the mic at real-time speed, then draws the streamed answer on a terminal OLED. When EDITH
asks to record a walk-around, the frames come from a video, a folder of photos, or the webcam; when
it asks for confirmation, you (or --confirm) answer. WiFi can be degraded to see how latency holds up.

Examples (server must be running):
  uv run python -m sim.device --say "what am I looking at"              # webcam, macOS voice
  uv run python -m sim.device --photo desk.jpg --say "what is this" --wifi bad
  uv run python -m sim.device --video car.mp4 --say "tell me about this car"
  uv run python -m sim.device --frames views/ --say "what's the serial number?"
  uv run python -m sim.device --say "remember I parked on level 3" --say "where did I park?"
  uv run python -m sim.device --photo desk.jpg                           # quick tap, no question
"""

import argparse
import io
import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps
from websockets.sync.client import connect

from sim.screen import COLS, Screen

IMAGE, AUDIO, BURST = b"\x01", b"\x02", b"\x03"
SAMPLE_RATE = 16_000
BYTES_PER_SEC = SAMPLE_RATE * 2
CHUNK = BYTES_PER_SEC // 10  # 100ms of audio: the mic hands the firmware one buffer at a time

# One-way latency (s) and throughput (bytes/s). ESP32 WiFi is a few Mbit/s at best in practice.
WIFI = {
    "perfect": (0.0, None),
    "good": (0.005, 1_000_000),
    "ok": (0.030, 250_000),
    "bad": (0.150, 40_000),
}


def sleep_until(t: float) -> None:
    if (delay := t - time.perf_counter()) > 0:
        time.sleep(delay)


class Link:
    """The WebSocket, through simulated WiFi: frames wait their turn on the air, then the latency."""

    def __init__(self, url: str, latency: float, rate: float | None) -> None:
        self.ws = connect(url, max_size=None)
        self.latency, self.rate = latency, rate
        self.outbox: queue.Queue = queue.Queue()
        self.inbox: queue.Queue = queue.Queue()
        threading.Thread(target=self._uplink, daemon=True).start()
        threading.Thread(target=self._downlink, daemon=True).start()

    def send(self, frame: bytes | str) -> None:
        self.outbox.put((time.perf_counter(), frame))

    def receive(self) -> dict:
        arrived, msg = self.inbox.get()
        sleep_until(arrived + self.latency)
        return msg

    def _uplink(self) -> None:
        air_free = 0.0  # when the radio finishes sending what's already queued
        while True:
            queued, frame = self.outbox.get()
            air_free = max(air_free, queued) + (len(frame) / self.rate if self.rate else 0)
            sleep_until(air_free + self.latency)
            self.ws.send(frame)

    def _downlink(self) -> None:
        for raw in self.ws:
            self.inbox.put((time.perf_counter(), json.loads(raw)))  # timestamp now; latency applied on read


class TerminalOLED:
    """Draws the 128x64 OLED's 21x8 characters as a box in the terminal, redrawn in place."""

    HEIGHT = 11  # box + status line

    def __init__(self) -> None:
        self.drawn = False
        self.live = sys.stdout.isatty()  # piped to a file/log: no redraw tricks, only print changes
        self.last = None

    def draw(self, screen: Screen, status: str) -> None:
        if not self.live:
            frame = (tuple(screen.lines()), status.split()[0])
            if frame != self.last:
                print("\n".join(["+" + "-" * COLS + "+", *(f"|{line:<{COLS}}|" for line in screen.lines()), status]))
            self.last = frame
            return
        if self.drawn:
            sys.stdout.write(f"\033[{self.HEIGHT}F")  # cursor back to the top of the box
        lines = screen.lines() + [""] * (8 - len(screen.lines()))
        out = ["┌" + "─" * COLS + "┐"]
        out += [f"│\033[96m{line:<{COLS}}\033[0m│" for line in lines]
        out += ["└" + "─" * COLS + "┘", f"\033[2K {status}"]
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        self.drawn = True


def to_vga_jpeg(data: bytes, name: str = "image") -> bytes:
    """A 640x480 JPEG, like the ESP32 camera in VGA mode."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001 (any decode failure means "not an image")
        sys.exit(f"can't read {name} as an image (iPhone HEIC? convert: sips -s format jpeg in.heic --out out.jpg)")
    buf = io.BytesIO()
    ImageOps.fit(img, (640, 480)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


class StillSource:
    """The same photo for everything (fine for questions that don't need a walk-around)."""

    def __init__(self, path: Path) -> None:
        self.jpeg = to_vga_jpeg(path.read_bytes(), str(path))

    def frame(self, progress: float = 0.0) -> bytes:
        return self.jpeg


class FolderSource:
    """Photos in a folder, in name order, played across the walk-around (first one is the initial photo)."""

    def __init__(self, folder: Path) -> None:
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not paths:
            sys.exit(f"no .jpg/.png files in {folder}")
        self.frames = [to_vga_jpeg(p.read_bytes(), str(p)) for p in paths]

    def frame(self, progress: float = 0.0) -> bytes:
        return self.frames[min(int(progress * len(self.frames)), len(self.frames) - 1)]


class VideoSource:
    """A video of walking around something, stretched across the walk-around (first frame is the initial photo)."""

    def __init__(self, path: Path) -> None:
        import cv2  # only needed here; heavy import

        self.cv2 = cv2
        self.video = cv2.VideoCapture(str(path))
        self.count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.video.isOpened() or self.count < 1:
            sys.exit(f"can't read {path} as a video")

    def frame(self, progress: float = 0.0) -> bytes:
        self.video.set(self.cv2.CAP_PROP_POS_FRAMES, min(int(progress * self.count), self.count - 1))
        ok, img = self.video.read()
        if not ok:
            sys.exit("video read failed")
        ok, data = self.cv2.imencode(".jpg", img)
        return to_vga_jpeg(data.tobytes())


class WebcamSource:
    """Live webcam frames, from camera.py in its own process (OpenCV and pygame clash in one process)."""

    def __init__(self) -> None:
        self.latest: bytes | None = None
        self.proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("camera.py"))], stdout=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True).start()
        deadline = time.monotonic() + 10
        while self.latest is None:
            if time.monotonic() > deadline or self.proc.poll() is not None:
                sys.exit("webcam failed; pass --photo, --frames or --video instead")
            time.sleep(0.05)
        time.sleep(0.5)  # let exposure settle

    def _read(self) -> None:
        while header := self.proc.stdout.read(4):
            self.latest = self.proc.stdout.read(struct.unpack(">I", header)[0])

    def frame(self, progress: float = 0.0) -> bytes:
        return self.latest


def load_audio(say: str | None, wav: Path | None) -> bytes:
    if say:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q.wav"
            subprocess.run(["say", "-o", str(out), f"--data-format=LEI16@{SAMPLE_RATE}", say], check=True)
            return load_audio(None, out)
    if wav:
        with wave.open(str(wav)) as w:
            if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (SAMPLE_RATE, 2, 1):
                sys.exit("WAV must be 16 kHz, 16-bit, mono (or use --say)")
            return w.readframes(w.getnframes())
    return b""  # quick tap


def capture(link: Link, screen: Screen, oled: TerminalOLED, source, request: dict) -> None:
    """Do what the firmware will: count down, record frames at the requested rate, upload as we go."""
    for n in range(request["countdown"], 0, -1):
        screen.status = f"{request['text']}\nStarting in {n}..."
        oled.draw(screen, "COUNTDOWN")
        time.sleep(1)
    started, interval, seconds = time.perf_counter(), 1 / request["fps"], request["seconds"]
    k = 0
    while (t := k * interval) < seconds:
        sleep_until(started + t)
        link.send(BURST + source.frame(t / seconds))
        screen.status = f"{request['text']}\nRecording... {seconds - t:.0f}s"
        oled.draw(screen, f"RECORDING  frame {k + 1}")
        k += 1
    link.send(json.dumps({"type": "capture_done", "id": request["id"]}))
    screen.status = "Thinking..."
    oled.draw(screen, "THINKING")


def confirm(link: Link, screen: Screen, oled: TerminalOLED, request: dict, policy: str) -> None:
    screen.status = f"{request['text']}\nPress to confirm"
    oled.draw(screen, "CONFIRM?")
    if policy == "ask" and sys.stdin.isatty():
        ok = input(f"  {request['text']} [y/N] ").strip().lower() in ("y", "yes")
    else:
        ok = policy == "yes"
    link.send(json.dumps({"type": "confirm_result", "id": request["id"], "ok": ok}))
    screen.status = "Thinking..."


def run_turn(link: Link, source, audio: bytes, confirm_policy: str) -> dict:
    screen, oled = Screen(), TerminalOLED()

    pressed = time.perf_counter()  # button down: photo goes out immediately, mic starts
    link.send(json.dumps({"type": "start"}))
    link.send(IMAGE + source.frame(0.0))
    screen.show("Listening...")
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i : i + CHUNK]
        sleep_until(pressed + (i + len(chunk)) / BYTES_PER_SEC)  # a buffer is ready once it's been spoken
        link.send(AUDIO + chunk)
        oled.draw(screen, f"LISTENING  {time.perf_counter() - pressed:.1f}s")

    released = time.perf_counter()  # button up: everything from here is what the wearer waits for
    link.send(json.dumps({"type": "end"}))
    screen.show("Thinking...")
    oled.draw(screen, "THINKING")

    first_word = None
    while True:
        msg = link.receive()
        if msg["type"] == "capture":
            capture(link, screen, oled, source, msg)
            continue
        if msg["type"] == "confirm":
            confirm(link, screen, oled, msg, confirm_policy)
            continue
        server_timings = screen.handle(msg)
        elapsed = time.perf_counter() - released
        if msg["type"] == "delta" and first_word is None:
            first_word = elapsed
        if msg["type"] == "error":
            oled.draw(screen, f"ERROR after {elapsed:.2f}s")
            return {"error": msg["message"]}
        oled.draw(screen, "DONE" if server_timings else f"ANSWERING  {elapsed:.1f}s")
        if server_timings:
            return {"first_word": first_word, "total": elapsed, "server": server_timings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--photo", type=Path, help="image to 'see' (default: the webcam)")
    view.add_argument("--frames", type=Path, help="folder of photos to walk around (first is the initial photo)")
    view.add_argument("--video", type=Path, help="video of walking around something")
    question = parser.add_mutually_exclusive_group()
    question.add_argument("--say", action="append", help="question to speak (repeat for follow-ups, same session)")
    question.add_argument("--wav", type=Path, help="16 kHz mono WAV of you asking a question")
    parser.add_argument("--confirm", choices=["ask", "yes", "no"], default="ask", help="answer to confirmations")
    parser.add_argument("--wifi", choices=WIFI, default="good")
    args = parser.parse_args()

    if args.photo:
        source = StillSource(args.photo)
    elif args.frames:
        source = FolderSource(args.frames)
    elif args.video:
        source = VideoSource(args.video)
    else:
        source = WebcamSource()
    questions = [load_audio(q, None) for q in args.say] if args.say else [load_audio(None, args.wav)]

    token = os.environ.get("EDITH_TOKEN")
    try:
        link = Link(f"{args.url}?token={quote(token)}" if token else args.url, *WIFI[args.wifi])
    except OSError:
        sys.exit(f"can't reach the server at {args.url} (is it running?)")

    for i, audio in enumerate(questions):
        if i:
            print()
        print(f"question {len(audio) / BYTES_PER_SEC:.1f}s of audio | wifi {args.wifi}\n")
        result = run_turn(link, source, audio, args.confirm)
        if "error" in result:
            sys.exit(1)
        s = result["server"]
        print(f"\nwearer waited: first word {result['first_word']:.2f}s, full answer {result['total']:.2f}s")
        print(f"server side:   stt {s['stt']}ms, first word {s.get('first_word', '-')}ms, total {s['total']}ms")


if __name__ == "__main__":
    main()

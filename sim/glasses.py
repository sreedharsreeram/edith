"""Simulated EDITH glasses: webcam = camera, laptop mic = mic, a window = the OLED.

Hold SPACE to talk (it's the button). Speaks PROTOCOL.md, exactly like the real firmware will.
Usage: uv run python -m sim.glasses [ws://host:8000/ws]   (set EDITH_TOKEN if the server uses one)
"""

import io
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pygame
import sounddevice as sd
from websockets.sync.client import connect

from sim.screen import Screen

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
TOKEN = os.environ.get("EDITH_TOKEN", "")
IMAGE, AUDIO, BURST = b"\x01", b"\x02", b"\x03"
SAMPLE_RATE = 16_000
OLED_W, OLED_H = 128, 64
SCALE = 4


class Glasses:
    def __init__(self) -> None:
        self.ws = connect(f"{URL}?token={quote(TOKEN)}" if TOKEN else URL, max_size=None)
        self.send_lock = threading.Lock()  # mic, capture and UI threads all send
        self.latest_jpeg: bytes | None = None
        self.mic_chunks: queue.Queue[bytes] = queue.Queue()
        self.talking = False
        self.pending_confirm: dict | None = None
        self.screen = Screen()
        threading.Thread(target=self._read_camera, daemon=True).start()
        threading.Thread(target=self._read_server, daemon=True).start()

    def send(self, frame: bytes | str) -> None:
        with self.send_lock:
            self.ws.send(frame)

    def _read_camera(self) -> None:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("camera.py"))], stdout=subprocess.PIPE)
        while header := proc.stdout.read(4):
            (size,) = struct.unpack(">I", header)
            self.latest_jpeg = proc.stdout.read(size)

    def _read_server(self) -> None:
        for raw in self.ws:
            msg = json.loads(raw)
            if msg["type"] == "capture":
                threading.Thread(target=self._capture, args=(msg,), daemon=True).start()
            elif msg["type"] == "confirm":
                self.pending_confirm = msg
                self.screen.status = f"{msg['text']}\nSPACE = yes"
                threading.Timer(msg["timeout"], self._confirm_expired, args=(msg,)).start()
            elif t := self.screen.handle(msg):
                print(f"stt {t['stt']}ms | first word {t.get('first_word', '-')}ms | total {t['total']}ms")

    def _capture(self, request: dict) -> None:
        """Count down, then stream webcam frames at the requested rate (this is what the firmware will do)."""
        for n in range(request["countdown"], 0, -1):
            self.screen.status = f"{request['text']}\nStarting in {n}..."
            time.sleep(1)
        started = time.monotonic()
        while (elapsed := time.monotonic() - started) < request["seconds"]:
            self.screen.status = f"{request['text']}\nRecording... {request['seconds'] - elapsed:.0f}s"
            if self.latest_jpeg:
                self.send(BURST + self.latest_jpeg)
            time.sleep(1 / request["fps"])
        self.send(json.dumps({"type": "capture_done", "id": request["id"]}))
        self.screen.status = "Thinking..."

    def _confirm_expired(self, request: dict) -> None:
        if self.pending_confirm is request:  # the server already treats no answer as "no"
            self.pending_confirm = None
            self.screen.status = "Cancelled"

    def _on_mic(self, indata, frames, time, status) -> None:  # runs on the audio thread
        if self.talking:
            self.mic_chunks.put(bytes(indata))

    def press(self) -> None:
        if request := self.pending_confirm:  # a press while EDITH asks "are you sure?" means yes
            self.pending_confirm = None
            self.send(json.dumps({"type": "confirm_result", "id": request["id"], "ok": True}))
            self.screen.status = "Thinking..."
            return
        if self.latest_jpeg is None:
            self.screen.show("Camera not ready...")
            return
        self.talking = True
        self.screen.show("Listening...")
        self.send(json.dumps({"type": "start"}))
        self.send(IMAGE + self.latest_jpeg)  # photo at press time, uploads while you talk

    def release(self) -> None:
        if not self.talking:
            return
        self.talking = False
        self.flush_mic()
        self.send(json.dumps({"type": "end"}))
        self.screen.show("Thinking...")

    def flush_mic(self) -> None:
        chunk = b""
        while not self.mic_chunks.empty():
            chunk += self.mic_chunks.get()
        if chunk:
            self.send(AUDIO + chunk)


def draw_oled(font: pygame.font.Font, screen: Screen) -> pygame.Surface:
    oled = pygame.Surface((OLED_W, OLED_H))
    for i, line in enumerate(screen.lines()):
        oled.blit(font.render(line, False, (120, 220, 255)), (1, i * 8))
    return pygame.transform.scale(oled, (OLED_W * SCALE, OLED_H * SCALE))  # chunky pixels, like the real thing


def draw_camera(jpeg: bytes | None) -> pygame.Surface | None:
    if jpeg is None:
        return None
    return pygame.transform.smoothscale(pygame.image.load(io.BytesIO(jpeg), "frame.jpg"), (320, 240))


def main() -> None:
    glasses = Glasses()
    pygame.init()
    window = pygame.display.set_mode((OLED_W * SCALE + 340, max(OLED_H * SCALE, 240) + 20))
    pygame.display.set_caption("EDITH simulator — hold SPACE to talk")
    font = pygame.font.SysFont("Menlo", 8)
    clock = pygame.time.Clock()

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=glasses._on_mic):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not event.repeat:
                    glasses.press()
                if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                    glasses.release()

            if glasses.talking:
                glasses.flush_mic()  # stream audio as you speak, not all at once at the end

            window.fill((20, 20, 20))
            if cam := draw_camera(glasses.latest_jpeg):
                window.blit(cam, (10, 10))
            window.blit(draw_oled(font, glasses.screen), (340, 10))
            pygame.display.flip()
            clock.tick(30)


if __name__ == "__main__":
    main()

# EDITH

Open-source AI glasses you can build for under $150: an ESP32 with a camera, a tiny display in
front of your eye, and an agent on your laptop that sees what you see. Hold the button, ask a
question, and the answer appears in your field of view.

```
glasses (ESP32, or a simulator)  ──WebSocket──▶  server  ──▶  Whisper (local speech-to-text)
     camera · mic · OLED · button                  │
                                                   └──▶  agent  ──▶  Gemini or Claude
                                                           │
                                                           └──▶  tools: walk around an object,
                                                                 take a photo, notes, time
```

- **Hardware:** Seeed XIAO ESP32-S3 Sense, 0.96" OLED, beam splitter + lens, LiPo battery.
- **Protocol:** the glasses and server talk over one WebSocket, including server requests like "record a
  walk-around" and "confirm this" (see `server/edith/protocol.py`).
- **Safety:** tools with side effects need a physical button press to confirm, so text in a photo or a
  web page can't make EDITH act on its own.

```
firmware/   ESP32 code for the glasses
server/     the Python server: agent, tools, model adapters, simulators, AWS deploy
```

## Run it

Needs [uv](https://docs.astral.sh/uv/) and a Gemini or Anthropic API key. Everything below runs from `server/`.

```sh
cd server
export GEMINI_API_KEY=...                # from aistudio.google.com (or ANTHROPIC_API_KEY)
uv run uvicorn edith.app:create_app --factory --host 0.0.0.0 --port 8000
```

Then, in a second terminal, pick a pair of simulated glasses.

**Virtual device** (scripted; behaves like the ESP32: real-time mic, simulated WiFi, terminal OLED):

```sh
uv run python -m sim.device --say "what am I looking at"                  # webcam + macOS voice
uv run python -m sim.device --photo desk.jpg --say "what is this" --wifi bad
uv run python -m sim.device --video car.mp4 --say "tell me about this car"  # walk-around from a video
uv run python -m sim.device --say "remember I parked on level 3" --say "where did I park?"
```

**Interactive** (webcam, your real voice, and a fake OLED window; hold SPACE to talk):

```sh
uv run python -m sim.glasses
```

## How the agent works

`server/edith/agent/harness.py` runs the loop: the model sees the photo and question, may call tools (in
parallel), sees their results, and repeats, for up to 5 steps. The last step has tools switched off
so it must answer. Tool arguments are validated against their JSON Schema, every tool has a timeout,
and failures go back to the model as errors instead of crashing the turn.

Every turn is written to `data/traces/` (one JSON line per turn: each model step and tool call, with
timings). Tool output and images are never stored there.

Adding a skill means writing a `Tool` in `server/edith/tools/` and listing it in `default_tools()`.

## Settings

Environment variables on the server:

- `EDITH_BRAIN`: `gemini` or `claude` (default: whichever API key is set)
- `EDITH_GEMINI_MODELS`: comma-separated, tried in order when one is busy
  (default `gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash`)
- `EDITH_MODEL`: Claude model (default `claude-opus-5-5`)
- `EDITH_WHISPER`: speech-to-text model (default `base.en`)
- `EDITH_TOKEN`: shared secret the glasses must send (strongly recommended)
- `EDITH_DATA_DIR`: where notes and traces live (default `data/`)

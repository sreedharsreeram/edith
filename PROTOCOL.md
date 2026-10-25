# Glasses <-> Server protocol

One WebSocket, kept open: `ws://<laptop-ip>:8000/ws` (add `?token=...` if the server sets `EDITH_TOKEN`).
The simulators and the real ESP32 firmware all speak exactly this, so the server never knows the difference.

## A normal question (glasses -> server)

| Frame | When | Contents |
|---|---|---|
| text `{"type":"start"}` | button pressed | starts a turn (and cancels any answer in progress) |
| binary `0x01` + JPEG | right after `start` | photo of what you're looking at |
| binary `0x02` + PCM | while button held | audio chunk: 16 kHz, 16-bit little-endian, mono |
| text `{"type":"end"}` | button released | server starts answering |

A quick tap with no speech means "What am I looking at?".

## The answer (server -> glasses)

| Frame | Meaning |
|---|---|
| `{"type":"transcript","text":"..."}` | what Whisper heard |
| `{"type":"status","text":"..."}` | what EDITH is doing ("Checking notes..."); replaced by the answer once it starts |
| `{"type":"delta","text":"..."}` | next chunk of the answer (append it to the screen) |
| `{"type":"done","timings":{...}}` | answer finished; timings in ms |
| `{"type":"error","message":"..."}` | something went wrong; show it |

## Requests from the server, mid-answer

EDITH's tools can ask the glasses to do things. At most one request is active at a time.

**Capture a burst** (e.g. "walk around the car"):

| Direction | Frame |
|---|---|
| server -> glasses | `{"type":"capture","id":"c1","text":"Walk slowly around it","countdown":3,"seconds":12,"fps":2}` |
| glasses | show `text`, count down `countdown` seconds, then record `seconds` at `fps` |
| glasses -> server | binary `0x03` + JPEG for each frame, sent as it's taken |
| glasses -> server | `{"type":"capture_done","id":"c1"}` |

The server waits `countdown + seconds + 5s` at most, then uses whatever frames arrived.

**Confirm an action** (anything with side effects, e.g. deleting a note):

| Direction | Frame |
|---|---|
| server -> glasses | `{"type":"confirm","id":"k1","text":"Delete note: Gate code?","timeout":10}` |
| glasses | show `text`; a button press within `timeout` seconds means yes |
| glasses -> server | `{"type":"confirm_result","id":"k1","ok":true}` |

While a confirmation is showing, a button press answers it; it does **not** send `start`.
No reply within the timeout counts as no.

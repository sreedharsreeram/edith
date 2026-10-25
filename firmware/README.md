# Firmware

Code for the glasses: a Seeed XIAO ESP32-S3 Sense (camera, mic, WiFi) driving a 0.96" SSD1306 OLED
and a push button. It speaks the same WebSocket protocol (`server/edith/protocol.py`) as the simulators in
`server/sim/`, so the server can't tell the difference.

Not written yet. Planned pieces:

- WiFi + WebSocket connection to the server (with the `EDITH_TOKEN`)
- Button: start/end a turn, answer confirmations
- Camera: VGA JPEG on button press; bursts when the server asks for a capture
- Mic: 16 kHz PCM streamed in 100 ms chunks while the button is held
- OLED: transcript, status and streamed answer, mirrored for the beam splitter

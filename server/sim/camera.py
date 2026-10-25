"""Webcam feeder, run as its own process by glasses.py (OpenCV and pygame clash in one process).

Writes frames to stdout as: 4-byte big-endian length + JPEG bytes. 640x480 matches the ESP32's VGA mode.
"""

import struct
import sys

import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    sys.exit("could not open webcam (check System Settings > Privacy > Camera for your terminal)")

while True:
    ok, frame = cam.read()
    if not ok:
        continue
    frame = cv2.resize(frame, (640, 480))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = jpeg.tobytes()
    sys.stdout.buffer.write(struct.pack(">I", len(data)) + data)
    sys.stdout.buffer.flush()

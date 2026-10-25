"""Speech-to-text with a local Whisper model, loaded once and kept warm."""

import os

import numpy as np
from faster_whisper import WhisperModel

from .protocol import SAMPLE_RATE


class WhisperTranscriber:
    def __init__(self, model: str | None = None) -> None:
        # base.en is fast on Apple Silicon; try small.en if transcripts are sloppy.
        name = model or os.environ.get("EDITH_WHISPER", "base.en")
        self.model = WhisperModel(name, device="cpu", compute_type="int8")
        # The first call is slow (lazy init), so pay that at startup instead of on the first question.
        self(np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes())

    def __call__(self, pcm16: bytes) -> str:
        pcm16 = pcm16[: len(pcm16) // 2 * 2]  # an odd byte count (dropped byte on the wire) must not crash decoding
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        # beam_size=1 is greedy decoding (fastest); vad_filter trims silence so there's less to decode
        # and stops Whisper from hallucinating words in quiet audio.
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

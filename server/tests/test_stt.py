"""Real Whisper on recorded questions: accuracy and speed. Slow: loads the model."""

import json
import re
import statistics
import time
import wave
from pathlib import Path

import pytest

from edith.protocol import SAMPLE_RATE

pytestmark = pytest.mark.slow

AUDIO = Path(__file__).parent / "fixtures" / "audio"
TRANSCRIPTS = json.loads((AUDIO / "transcripts.json").read_text())


@pytest.fixture(scope="module")
def transcribe():
    from edith.stt import WhisperTranscriber

    return WhisperTranscriber()


def pcm(name: str) -> bytes:
    with wave.open(str(AUDIO / f"{name}.wav")) as w:
        assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (SAMPLE_RATE, 2, 1)
        return w.readframes(w.getnframes())


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).split()


def word_error_rate(expected: str, actual: str) -> float:
    ref, hyp = words(expected), words(actual)
    dist = list(range(len(hyp) + 1))  # edit distance over words
    for i, r in enumerate(ref, 1):
        prev, dist[0] = dist[0], i
        for j, h in enumerate(hyp, 1):
            prev, dist[j] = dist[j], min(dist[j] + 1, dist[j - 1] + 1, prev + (r != h))
    return dist[-1] / len(ref)


@pytest.mark.parametrize("name", sorted(TRANSCRIPTS))
def test_transcribes_questions_accurately(transcribe, name):
    assert word_error_rate(TRANSCRIPTS[name], transcribe(pcm(name))) <= 0.15


def test_fast_enough(transcribe):
    audio = pcm("plant")
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        transcribe(audio)
        runs.append(time.perf_counter() - start)
    assert statistics.median(runs) < 1.0, f"median {statistics.median(runs):.2f}s"


def test_silence_is_empty(transcribe):
    assert transcribe(bytes(SAMPLE_RATE * 2 * 2)) == ""


def test_odd_byte_count_does_not_crash(transcribe):
    transcribe(pcm("looking") + b"\x00")

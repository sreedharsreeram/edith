"""Picking the best frames from a walk-around burst: drop the blurry ones, keep the most different views."""

import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError

log = logging.getLogger("edith")

BLUR_CUTOFF = 0.4  # frames below this fraction of the median sharpness are motion-blurred


def select_views(frames: list[bytes], k: int) -> list[bytes]:
    """Up to k frames, as sharp and as different from each other as possible, in the order they were taken."""
    decoded = [(i, gray) for i, frame in enumerate(frames) if (gray := _gray(frame)) is not None]
    if len(decoded) <= k:
        return [frames[i] for i, _ in decoded]

    sharpness = {i: _sharpness(gray) for i, gray in decoded}
    cutoff = BLUR_CUTOFF * float(np.median(list(sharpness.values())))
    sharp = [(i, g) for i, g in decoded if sharpness[i] >= cutoff]
    candidates = sharp if len(sharp) >= k else decoded  # never return fewer views just to avoid blur

    # Farthest-point sampling: start from the sharpest frame, then keep adding whichever frame
    # differs most from everything picked so far. Spreads the picks around the object.
    thumbs = {i: _thumbnail(g) for i, g in candidates}
    picked = [max(candidates, key=lambda c: sharpness[c[0]])[0]]
    distance = {i: _distance(thumbs[i], thumbs[picked[0]]) for i, _ in candidates}
    while len(picked) < k:
        next_i = max((i for i, _ in candidates if i not in picked), key=lambda i: (distance[i], sharpness[i]))
        picked.append(next_i)
        for i, _ in candidates:
            distance[i] = min(distance[i], _distance(thumbs[i], thumbs[next_i]))
    return [frames[i] for i in sorted(picked)]


def _gray(jpeg: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(jpeg)).convert("L").resize((160, 120))
    except (UnidentifiedImageError, OSError):
        log.warning("skipping a frame that isn't a readable image")
        return None
    return np.asarray(img, dtype=np.float32)


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high when there are crisp edges, low when blurred."""
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def _thumbnail(gray: np.ndarray) -> np.ndarray:
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((16, 12)), dtype=np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)  # ignore overall brightness changes


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())

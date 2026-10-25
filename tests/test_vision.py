import io

import numpy as np
from PIL import Image, ImageFilter

from server.vision import select_views


def view(seed: int, blur: float = 0) -> bytes:
    """A distinct, detailed 'view': random blocks, so sharpness and difference are measurable."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
    img = Image.fromarray(blocks).resize((320, 240), Image.NEAREST)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_fewer_frames_than_asked_returns_them_all():
    frames = [view(1), view(2)]
    assert select_views(frames, 5) == frames


def test_drops_blurry_frames():
    sharp = [view(i) for i in range(6)]
    blurry = [view(100 + i, blur=6) for i in range(3)]
    picked = select_views(sharp[:3] + blurry + sharp[3:], 6)
    assert set(picked) == set(sharp)


def test_prefers_different_views_over_duplicates():
    a, b, c = view(1), view(2), view(3)
    frames = [a, a, a, a, b, b, b, c]  # standing still for a while, then moving
    assert sorted(select_views(frames, 3)) == sorted([a, b, c])


def test_keeps_time_order():
    frames = [view(i) for i in range(10)]
    picked = select_views(frames, 4)
    assert picked == sorted(picked, key=frames.index)


def test_skips_unreadable_frames():
    frames = [view(1), b"\xff\xd8 corrupt", view(2)]
    assert select_views(frames, 5) == [frames[0], frames[2]]


def test_all_blurry_still_returns_views():
    frames = [view(i, blur=6) for i in range(6)]
    assert len(select_views(frames, 3)) == 3

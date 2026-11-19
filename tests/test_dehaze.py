import cv2
import numpy as np

from src.dehaze import DarkChannelDehazer
from src.hazing import add_fog
from src.metrics import psnr


def _street_like_scene(h=180, w=320):
    """Sky-to-road gradient with dark vehicle-like blobs (DCP assumes dark pixels exist)."""
    y = np.linspace(0, 1, h)[:, None, None]
    base = np.array([0.55, 0.35, 0.2]) * (1 - y) + np.array([0.15, 0.15, 0.15]) * y
    base = np.broadcast_to(base, (h, w, 3)).astype(np.float32).copy()
    rng = np.random.default_rng(3)
    for _ in range(12):
        x0 = int(rng.integers(0, w - 40))
        y0 = int(rng.integers(h // 2, h - 20))
        color = tuple(float(v) for v in rng.uniform(0.02, 0.15, 3))
        cv2.rectangle(base, (x0, y0), (x0 + 30, y0 + 15), color, -1)
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def test_dehaze_improves_psnr_on_dark_scene():
    clear = _street_like_scene()
    hazy = add_fog(clear, beta=0.8, airlight=0.92)
    out = DarkChannelDehazer().dehaze(hazy)
    assert out.shape == hazy.shape and out.dtype == np.uint8
    assert psnr(out, clear) > psnr(hazy, clear)

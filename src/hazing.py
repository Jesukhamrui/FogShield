"""Synthetic fog generation using the atmospheric scattering model.

    I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))

J is the clear scene, I the hazy observation, A the airlight, t the transmission
and d the scene depth. When no depth map is available (e.g. plain Cityscapes
images), a ground-plane proxy is used: pixels nearer the top of the image are
treated as farther away.
"""
import numpy as np


def ground_plane_depth(h: int, w: int, max_depth: float = 3.0) -> np.ndarray:
    """Proxy depth map in [0, max_depth]: bottom rows are near, top rows far."""
    y = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    depth = np.repeat(y, w, axis=1)
    return depth * max_depth


def add_fog(img_bgr: np.ndarray, beta: float = 1.0, airlight: float = 0.9,
            depth: np.ndarray | None = None) -> np.ndarray:
    """Return a hazy uint8 BGR image generated from a clear image."""
    img = img_bgr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if depth is None:
        depth = ground_plane_depth(h, w)
    t = np.exp(-beta * depth)[..., None]
    out = img * t + airlight * (1.0 - t)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

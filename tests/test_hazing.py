import numpy as np

from src.hazing import add_fog, ground_plane_depth


def test_fog_preserves_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    out = add_fog(img, beta=1.0)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_fog_reduces_contrast():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32] = 0
    img[32:] = 255
    out = add_fog(img, beta=1.2, airlight=0.9)
    assert out.std() < img.std()


def test_zero_beta_returns_clear_image():
    img = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    out = add_fog(img, beta=0.0)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_depth_proxy_bottom_is_near():
    d = ground_plane_depth(10, 10)
    assert d[-1, 0] < d[0, 0]

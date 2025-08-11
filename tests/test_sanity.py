import numpy as np
from cv_starter.utils import normalize_image


def test_normalize_image():
    arr = np.array([0, 127, 255], dtype=np.uint8)
    out = normalize_image(arr)
    assert out.dtype == np.float32
    assert out.min() == 0.0
    assert np.isclose(out.max(), 1.0)

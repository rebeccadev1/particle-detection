import numpy as np

from src.capture.preview import display_frame


def test_display_frame_converts_bgr_and_shrinks() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[..., 0] = 255
    shown = display_frame(image, max_side=50)
    assert shown.shape == (25, 50, 3)
    assert int(shown[0, 0, 0]) == 0
    assert int(shown[0, 0, 2]) == 255


def test_display_frame_leaves_a_small_gray_image() -> None:
    image = np.full((20, 30), 40, dtype=np.uint8)
    shown = display_frame(image, max_side=960)
    assert shown.shape == (20, 30)
    assert int(shown[0, 0]) == 40

"""Depth averaged over frames (camera.median_depth, 13 Oct 2026)."""
import numpy as np

from fyp.shared.hardware.camera import median_depth


def test_one_frame_is_returned_unchanged():
    d = np.full((2, 2), 1.2, dtype=np.float32)
    assert median_depth([d]) is d


def test_each_pixel_is_the_median_of_its_valid_readings():
    frames = [np.full((1, 1), v, dtype=np.float32) for v in (1.20, 1.22, 1.50, 1.21, 1.19)]
    assert median_depth(frames)[0, 0] == np.float32(1.21)   # 1.50 is outvoted


def test_a_dropout_is_left_out_not_counted_as_zero():
    """Two of five frames with no reading must not pull the pixel towards the camera."""
    frames = [np.full((1, 1), v, dtype=np.float32) for v in (1.2, 0.0, 1.2, 0.0, 1.2)]
    assert median_depth(frames)[0, 0] == np.float32(1.2)


def test_a_pixel_never_seen_stays_invalid():
    frames = [np.zeros((1, 1), dtype=np.float32) for _ in range(5)]
    assert median_depth(frames)[0, 0] == 0.0

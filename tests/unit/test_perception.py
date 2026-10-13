"""Unit tests for the perception maths: filters, depth sampling, and localising.

Everything here runs on synthetic arrays. The detector itself is not tested,
because OWLv2 needs torch, which lives in the lerobot venv rather than this one,
so the model stays an integration concern. Everything AROUND the detector is
plain arithmetic and is covered below.

THE AXIS CONVENTION IS TESTED HERE, and it is the reason this file exists at
all. `pixel_to_camera` used to negate y and z, which is the OpenGL convention
MuJoCo uses, while the RealSense reports depth forward-positive with y down, in
the OpenCV convention its own intrinsics assume. Nothing caught it, because
`camera_to_pixel` negated them back and the round trip stayed exact. The tests
below pin the signs directly, so it cannot drift back.
"""

from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_a.perception.filters import (apply_filters, iou, nms,
                                                   to_json_records, top1_per_query)
from fyp.architecture_a.perception.localiser import (find_duplicates, locate,
                                                     nearest_truth)
from fyp.architecture_a.perception.pixel_to_3d import (camera_to_pixel, depth_at,
                                                       intrinsics_from_fovy,
                                                       pixel_to_camera,
                                                       smooth_depth,
                                                       surface_to_centroid)

INTR = intrinsics_from_fovy(60.0, 640, 480)


# -------------------------------------------------------------------- iou

def test_identical_boxes_overlap_completely():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_disjoint_boxes_do_not_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


def test_touching_boxes_do_not_overlap():
    """Edge case. A shared edge has zero area, and a negative width would be worse."""
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == pytest.approx(0.0)


def test_half_overlap_is_a_third():
    """Two equal boxes sharing half their area: intersection 1, union 3."""
    assert iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)


# -------------------------------------------------------------------- nms

def test_nms_keeps_the_higher_score_of_an_overlapping_pair():
    dets = [("cube", 0.9, [0, 0, 10, 10]), ("cube", 0.5, [1, 1, 11, 11])]
    assert nms(dets, iou_thresh=0.5) == [dets[0]]


def test_nms_keeps_both_when_they_do_not_overlap():
    dets = [("cube", 0.9, [0, 0, 10, 10]), ("cube", 0.5, [50, 50, 60, 60])]
    assert len(nms(dets, iou_thresh=0.5)) == 2


def test_nms_is_class_agnostic_by_default():
    """
    OWLv2 fires two different labels on one object regularly. Suppressing only
    within a label would keep both and the planner would be told there are two
    objects where there is one.
    """
    dets = [("cube", 0.9, [0, 0, 10, 10]), ("block", 0.5, [1, 1, 11, 11])]
    assert nms(dets, iou_thresh=0.5) == [dets[0]]


def test_per_query_nms_keeps_both_labels():
    dets = [("cube", 0.9, [0, 0, 10, 10]), ("block", 0.5, [1, 1, 11, 11])]
    assert len(nms(dets, iou_thresh=0.5, per_query=True)) == 2


# ------------------------------------------------------------------ top1

def test_top1_keeps_the_best_of_each_label():
    dets = [("cube", 0.4, [0, 0, 1, 1]), ("cube", 0.9, [5, 5, 6, 6]),
            ("tray", 0.7, [9, 9, 10, 10])]
    kept = top1_per_query(dets)
    assert sorted(d[1] for d in kept) == [0.7, 0.9]


# ---------------------------------------------------------- apply_filters

def test_the_filter_chain_reports_what_each_stage_removed():
    """
    The counts are how a bad detection run is diagnosed. Everything dropped at
    the threshold is a different problem from everything dropped by NMS.
    """
    dets = [("cube", 0.9, [0, 0, 10, 10]),
            ("cube", 0.8, [1, 1, 11, 11]),
            ("cube", 0.1, [50, 50, 60, 60])]

    kept, (n_raw, n_thresh, n_nms, n_final) = apply_filters(dets, threshold=0.3)
    assert (n_raw, n_thresh, n_nms, n_final) == (3, 2, 1, 1)
    assert kept[0][1] == 0.9


def test_the_threshold_is_inclusive():
    dets = [("cube", 0.3, [0, 0, 10, 10])]
    kept, _ = apply_filters(dets, threshold=0.3)
    assert len(kept) == 1


def test_json_records_carry_the_box_centre():
    """The centre is the only part the localiser reads, so it has to be right."""
    record = to_json_records([("cube", 0.912345, [0, 0, 10, 20])])[0]
    assert record["center_uv"] == [5.0, 10.0]
    assert record["score"] == 0.9123


# ------------------------------------------------------------- depth_at

def test_depth_at_takes_the_median_of_its_patch():
    """
    A single pixel of depth is speckly. The median over a patch is what makes a
    reading usable, and one wild pixel must not move it.
    """
    depth = np.full((20, 20), 1.0, dtype=float)
    depth[10, 10] = 9.0
    assert depth_at(depth, 10, 10, radius=2) == pytest.approx(1.0)


def test_depth_at_ignores_holes():
    """Zero means no reading on a RealSense, and averaging it in halves the range."""
    depth = np.zeros((20, 20), dtype=float)
    depth[9:12, 9:12] = 2.0
    assert depth_at(depth, 10, 10, radius=2) == pytest.approx(2.0)


def test_depth_at_rejects_readings_beyond_the_maximum():
    depth = np.full((20, 20), 5.0, dtype=float)
    assert np.isnan(depth_at(depth, 10, 10, radius=1, max_depth=2.0))


def test_depth_at_returns_nan_when_the_patch_is_empty():
    assert np.isnan(depth_at(np.zeros((20, 20)), 10, 10, radius=2))


def test_smooth_depth_matches_depth_at_at_every_pixel():
    """
    The saved depth image claims to show what the pipeline sees. That is only
    true if it agrees with depth_at everywhere, including at the borders, in
    holes, and where a reading is past the maximum.
    """
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.4, 2.5, (24, 32))
    depth[rng.random(depth.shape) < 0.2] = 0.0
    depth[5:12, 8:16] = 0.0          # a patch with no readings at all
    smooth = smooth_depth(depth, radius=2, max_depth=2.0)
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            want = depth_at(depth, u, v, radius=2, max_depth=2.0)
            if np.isnan(want):
                assert np.isnan(smooth[v, u]), (u, v)
            else:
                assert smooth[v, u] == pytest.approx(want, abs=1e-6), (u, v)


def test_depth_at_outside_the_image_is_nan_rather_than_an_exception():
    """
    A detection can sit on the edge, and a crash there would take down a whole
    trial for one bad box.
    """
    depth = np.ones((20, 20), dtype=float)
    assert np.isnan(depth_at(depth, 99, 10))
    assert np.isnan(depth_at(depth, 10, -1))


def test_depth_at_clamps_its_patch_at_the_border():
    depth = np.ones((20, 20), dtype=float)
    assert depth_at(depth, 0, 0, radius=3) == pytest.approx(1.0)


# ------------------------------------------------------- pinhole geometry

def test_pixel_and_camera_are_inverses():
    """
    Necessary and NOT sufficient. This passed for months while both functions
    shared the wrong convention, which is exactly why the sign tests below exist.
    """
    u, v, z = 200.0, 150.0, 0.8
    p = pixel_to_camera(u, v, z, INTR)
    u2, v2, z2 = camera_to_pixel(p, INTR)

    assert (u2, v2) == pytest.approx((u, v))
    assert z2 == pytest.approx(z)


def test_the_principal_point_maps_to_the_optical_axis():
    p = pixel_to_camera(INTR.cx, INTR.cy, 1.0, INTR)
    assert p[0] == pytest.approx(0.0)
    assert p[1] == pytest.approx(0.0)


def test_a_point_behind_the_camera_is_refused():
    with pytest.raises(ValueError, match="behind the camera"):
        camera_to_pixel(np.array([0.0, 0.0, 0.0]), INTR)


def test_the_camera_frame_is_opencv_not_opengl():
    """
    The regression test for A12. z must be POSITIVE going away from the camera,
    and y must increase DOWN the image, because that is what pyrealsense2
    deprojects into and what the SDK intrinsics assume.

    Getting this wrong is not a small error. With a MEASURED hand-eye transform,
    which is what the no-marker decision leaves, nothing absorbs the flip and
    every point lands mirrored and behind the camera.
    """
    p = pixel_to_camera(INTR.cx, INTR.cy, 0.8, INTR)
    assert p[2] == pytest.approx(0.8), "z must be positive depth, forward from the camera"

    below = pixel_to_camera(INTR.cx, INTR.cy + 100, 0.8, INTR)
    assert below[1] > 0, "a pixel lower in the image must have a larger y"

    right = pixel_to_camera(INTR.cx + 100, INTR.cy, 0.8, INTR)
    assert right[0] > 0, "a pixel right of centre must have a larger x"


def test_surface_to_centroid_pushes_the_point_away_from_the_camera():
    """
    The depth reading lands on the near face, and a grasp wants the middle, so
    the point moves further away. Away is +z in OpenCV axes.
    """
    p = np.array([0.1, 0.2, 0.5])
    out = surface_to_centroid(p, 0.02)
    assert out[2] > p[2]
    # The push is 0.02 along the ray, so z grows by 0.02 * cos(angle off axis).
    assert out[2] == pytest.approx(0.5 + 0.02 * 0.5 / np.linalg.norm(p))


def test_depth_scales_the_offset_linearly():
    """Twice as far away, twice the lateral offset for the same pixel."""
    near = pixel_to_camera(300.0, 200.0, 1.0, INTR)
    far = pixel_to_camera(300.0, 200.0, 2.0, INTR)
    assert far[0] == pytest.approx(near[0] * 2)


def test_intrinsics_from_fovy_puts_the_centre_in_the_middle():
    intr = intrinsics_from_fovy(60.0, 640, 480)
    assert intr.cx == pytest.approx(319.5)
    assert intr.cy == pytest.approx(239.5)
    assert intr.fx == intr.fy


@pytest.mark.parametrize("bad", [0.0, 180.0, -10.0, 200.0])
def test_an_impossible_field_of_view_is_refused(bad):
    with pytest.raises(ValueError, match="fovy"):
        intrinsics_from_fovy(bad, 640, 480)


def test_surface_to_centroid_moves_along_the_ray():
    """Off axis, the centre of a round object is further along the SAME ray."""
    p = np.array([0.1, 0.2, 0.5])
    out = surface_to_centroid(p, 0.02)
    assert np.linalg.norm(np.cross(out, p)) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(p) + 0.02)


def test_surface_to_centroid_does_not_modify_its_input():
    p = np.array([0.1, 0.2, 0.5])
    surface_to_centroid(p, 0.02)
    assert p[2] == 0.5


# ----------------------------------------------------------- the localiser

def a_depth_map(value: float = 0.8) -> np.ndarray:
    return np.full((480, 640), value, dtype=float)


def test_locate_attaches_a_3d_point_to_each_detection():
    dets = [{"query": "cube", "score": 0.9, "center_uv": [320.0, 240.0]}]
    located = locate(dets, a_depth_map(), INTR)

    assert len(located) == 1
    assert located[0].query == "cube"
    assert located[0].valid
    assert located[0].depth_m == pytest.approx(0.8)


def test_a_detection_with_no_depth_is_kept_but_marked_invalid():
    """
    Dropped detections vanish silently. A detection with no depth is evidence:
    it usually means the object is dark, shiny, or closer than the sensor
    minimum, and that is worth seeing rather than losing.
    """
    dets = [{"query": "cube", "score": 0.9, "center_uv": [320.0, 240.0]}]
    located = locate(dets, np.zeros((480, 640)), INTR)

    assert len(located) == 1
    assert not located[0].valid
    assert located[0].p_cam is None


def test_a_hole_at_the_centre_falls_back_to_the_middle_of_the_box():
    """
    13 Oct 2026: glare on a white lid left the centre patch empty and the box
    was dropped every time, although most of its middle had depth.
    """
    depth = a_depth_map()
    depth[235:246, 315:326] = 0.0                    # the hole under the centre patch
    box = [300.0, 220.0, 340.0, 260.0]
    located = locate([{"query": "box", "center_uv": [320.0, 240.0], "box_xyxy": box}],
                     depth, INTR, max_centre_offset=0.0)
    assert located[0].valid
    assert located[0].depth_from == "box"
    assert located[0].depth_m == pytest.approx(0.8)


def test_a_detection_without_a_box_is_not_pushed():
    located = locate([{"query": "cube", "center_uv": [320.0, 240.0]}], a_depth_map(), INTR)
    assert located[0].p_cam[2] == pytest.approx(0.8)
    assert located[0].centre_offset_m == 0.0


def test_a_boxed_detection_is_pushed_back_by_half_its_size():
    # 20 px at 0.8 m is 20 * 0.8 / fx metres wide, so the centre is half that further.
    box = [310.0, 230.0, 330.0, 250.0]
    located = locate([{"query": "ball", "center_uv": [320.0, 240.0], "box_xyxy": box}],
                     a_depth_map(), INTR, max_centre_offset=1.0)
    half = 0.5 * 20 * 0.8 / INTR.fx
    assert located[0].centre_offset_m == pytest.approx(half)
    assert np.linalg.norm(located[0].p_cam) == pytest.approx(0.8 + half, abs=1e-3)


def test_a_flat_object_uses_its_smaller_side():
    box = [220.0, 235.0, 420.0, 245.0]          # 200 px wide, 10 px tall
    located = locate([{"query": "tray", "center_uv": [320.0, 240.0], "box_xyxy": box}],
                     a_depth_map(), INTR, max_centre_offset=1.0)
    assert located[0].centre_offset_m == pytest.approx(0.5 * 10 * 0.8 / INTR.fy)


def test_the_push_is_capped():
    box = [220.0, 140.0, 420.0, 340.0]          # 200 px, about 0.39 m at 0.8 m
    located = locate([{"query": "box", "center_uv": [320.0, 240.0], "box_xyxy": box}],
                     a_depth_map(), INTR, max_centre_offset=0.04)
    assert located[0].centre_offset_m == pytest.approx(0.04)


def test_a_missing_score_does_not_crash_the_localiser():
    located = locate([{"query": "cube", "center_uv": [320.0, 240.0]}],
                     a_depth_map(), INTR)
    assert np.isnan(located[0].score)


def test_duplicates_are_found_by_separation_in_millimetres():
    dets = [{"query": "cube", "score": 0.9, "center_uv": [320.0, 240.0]},
            {"query": "block", "score": 0.8, "center_uv": [321.0, 240.0]}]
    located = locate(dets, a_depth_map(), INTR)

    assert len(find_duplicates(located, separation_mm=50.0)) == 1
    assert find_duplicates(located, separation_mm=0.1) == []


def test_duplicate_separation_defaults_to_the_configured_value():
    """Reads architecture_a.perception.duplicate_separation_mm rather than a constant."""
    dets = [{"query": "cube", "score": 0.9, "center_uv": [320.0, 240.0]},
            {"query": "block", "score": 0.8, "center_uv": [500.0, 240.0]}]
    assert find_duplicates(locate(dets, a_depth_map(), INTR)) == []


def test_invalid_detections_are_never_called_duplicates():
    dets = [{"query": "cube", "score": 0.9, "center_uv": [320.0, 240.0]},
            {"query": "block", "score": 0.8, "center_uv": [321.0, 240.0]}]
    located = locate(dets, np.zeros((480, 640)), INTR)
    assert find_duplicates(located, separation_mm=50.0) == []


def test_nearest_truth_picks_the_closest_and_reports_the_distance():
    truth = {"cube": np.array([0.0, 0.0, 0.5]), "tray": np.array([0.3, 0.0, 0.5])}
    name, dist = nearest_truth(np.array([0.02, 0.0, 0.5]), truth)

    assert name == "cube"
    assert dist == pytest.approx(0.02)

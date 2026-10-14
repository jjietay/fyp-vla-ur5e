"""Unit tests for the hand-eye transform, stage 4 of Architecture A.

Nothing downstream can address a real object without this, and until now none of
it was covered. The solver stays in the repo as the fallback for when the
measured mount transform proves too inaccurate, so it has to still work on the
day it is needed rather than on the day it was written.

Every test builds a KNOWN transform, applies it to make synthetic point pairs,
and checks the code recovers it. That is the only honest way to test a solver:
comparing it against real data would need a calibration, which is the thing being
produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_a.calibration.hand_eye import (ball_centre_in_base,
                                                     calibration_grid, camera_to_base,
                                                     fit_with_outlier_rejection, load,
                                                     report, residuals, save,
                                                     solve_rigid_transform)
from fyp.shared.helpers.rotations import euler_to_R

RNG = np.random.default_rng(7)


def seen_by_camera(p_base, T_base_cam) -> np.ndarray:
    """
    It takes base-frame points and a transform and gives you the same points in
    camera frame, to simulate what the camera measures. Test-only: the pipeline
    never needs this direction.
    """
    return camera_to_base(p_base, np.linalg.inv(np.asarray(T_base_cam, dtype=float)))


def a_transform(rpy=(0.3, -0.4, 1.1), t=(0.6, -0.3, 0.7)) -> np.ndarray:
    """It takes an orientation and a position and gives you a 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(np.asarray(rpy, dtype=float))
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def cloud(n: int = 12) -> np.ndarray:
    """It takes a count and gives you non-collinear points spread over a workspace."""
    return RNG.uniform(-0.3, 0.3, size=(n, 3))


# ------------------------------------------------------------------- solver

def test_the_solver_recovers_the_transform_it_was_given():
    T = a_transform()
    p_cam = cloud()
    p_base = np.array([T[:3, :3] @ p + T[:3, 3] for p in p_cam])

    np.testing.assert_allclose(solve_rigid_transform(p_cam, p_base), T, atol=1e-9)


def test_the_solver_is_unaffected_by_where_the_points_sit():
    """
    Kabsch centres both clouds before solving, so a cloud far from the origin
    must give the same answer as one near it. If it does not, translation is
    leaking into the rotation.
    """
    T = a_transform()
    near = cloud()
    far = near + np.array([5.0, -3.0, 2.0])

    for p_cam in (near, far):
        p_base = np.array([T[:3, :3] @ p + T[:3, 3] for p in p_cam])
        np.testing.assert_allclose(solve_rigid_transform(p_cam, p_base), T, atol=1e-9)


def test_the_result_is_a_rotation_and_never_a_reflection():
    """
    The SVD in Kabsch can produce a reflection when the points are noisy, and a
    reflected calibration mirrors the whole workspace. The arm reaches to the
    wrong side of the table and the transform looks plausible.
    """
    T = a_transform()
    p_cam = cloud()
    p_base = np.array([T[:3, :3] @ p + T[:3, 3] for p in p_cam])
    p_base += RNG.normal(scale=0.002, size=p_base.shape)

    R = solve_rigid_transform(p_cam, p_base)[:3, :3]
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_points_is_refused(n):
    """Three non-collinear pairs is the minimum for the problem to be solvable."""
    with pytest.raises(ValueError, match="at least 3"):
        solve_rigid_transform(cloud(n), cloud(n))


def test_mismatched_counts_are_refused():
    with pytest.raises(ValueError, match="point counts differ"):
        solve_rigid_transform(cloud(5), cloud(6))


# ---------------------------------------------------------------- reporting

def test_a_perfect_fit_reports_zero_error():
    T = a_transform()
    p_cam = cloud()
    p_base = np.array([T[:3, :3] @ p + T[:3, 3] for p in p_cam])

    stats = report(T, p_cam, p_base)
    assert stats["rms_mm"] == pytest.approx(0.0, abs=1e-6)
    assert stats["max_mm"] == pytest.approx(0.0, abs=1e-6)
    assert stats["n_points"] == len(p_cam)


def test_the_report_finds_the_worst_point():
    """
    `worst_index` is what you act on. It tells you which sample to re-take, and
    pointing at the wrong one sends you back to the bench for nothing.
    """
    T = a_transform()
    p_cam = cloud(6)
    p_base = np.array([T[:3, :3] @ p + T[:3, 3] for p in p_cam])
    p_base[3] += [0.0, 0.0, 0.05]   # 50 mm out, on purpose

    stats = report(T, p_cam, p_base)
    assert stats["worst_index"] == 3
    assert stats["max_mm"] == pytest.approx(50.0, abs=1e-6)


def test_residuals_are_reported_per_point_in_metres():
    T = a_transform()
    p_cam = cloud(4)
    p_base = np.array([T[:3, :3] @ p + T[:3, 3] for p in p_cam])

    assert residuals(T, p_cam, p_base).shape == (4,)


# ------------------------------------------------------------ use and store

def test_camera_to_base_moves_the_point_where_the_transform_says():
    T = a_transform()
    p = np.array([0.1, -0.2, 0.4])
    np.testing.assert_allclose(camera_to_base(p, T), T[:3, :3] @ p + T[:3, 3], atol=1e-12)


def test_save_then_load_returns_the_same_transform(tmp_path):
    T = a_transform()
    path = tmp_path / "calib" / "T_base_cam.json"

    save(T, path, meta={"method": "test"})
    np.testing.assert_allclose(load(path), T, atol=1e-12)


def test_save_creates_the_directory(tmp_path):
    """It writes into config/calibration/, which may not exist on a fresh clone."""
    path = tmp_path / "does" / "not" / "exist" / "T.json"
    save(a_transform(), path)
    assert path.is_file()


def test_loading_something_that_is_not_a_transform_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"T_base_cam": [[1, 0], [0, 1]]}')
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        load(path)


# ------------------------------------------------- the per-session ball fit

def _config_grid():
    from fyp.shared.hardware.safety import WorkspaceEnvelope
    from fyp.shared.helpers.config import get_config
    cfg = get_config()
    cc = cfg["architecture_a"]["calibration"]
    env = WorkspaceEnvelope(cfg)
    poses, rejected = calibration_grid(cc["grid_centre_xy"], cc["grid_half_extent_m"],
                                       cc["grid_n"], cc["z_levels"],
                                       cfg["architecture_a"]["skills"]["down_rotvec"], env)
    return cfg, env, poses, rejected


def test_the_configured_grid_gives_enough_poses_inside_the_envelope():
    """
    Checked against the REAL config, because a grid that mostly falls outside the
    envelope is found at the start of a lab session otherwise, with the camera set
    up and the clock running.
    """
    cfg, env, poses, _ = _config_grid()
    assert len(poses) >= int(cfg["architecture_a"]["calibration"]["min_points"])
    for pose in poses:
        env.check_pose(pose)


def test_the_configured_grid_is_not_coplanar():
    """A flat set of points leaves the rotation undetermined and the solver refuses it."""
    _, _, poses, _ = _config_grid()
    pts = np.asarray([p[:3] for p in poses])
    assert np.linalg.matrix_rank(pts - pts.mean(axis=0), tol=1e-6) == 3


def test_the_grid_keeps_one_tool_orientation():
    cfg, _, poses, _ = _config_grid()
    down = np.asarray(cfg["architecture_a"]["skills"]["down_rotvec"], dtype=float)
    assert all(np.allclose(p[3:], down) for p in poses)


def test_the_grid_walks_in_short_steps():
    """
    Snake order. A raster that jumps back across the table each row is a long
    unwatched moveL with a ball held only by friction.
    """
    _, _, poses, _ = _config_grid()
    from fyp.shared.helpers.config import get_config
    cc = get_config()["architecture_a"]["calibration"]
    step = 2 * cc["grid_half_extent_m"] / (cc["grid_n"] - 1)
    zs = cc["z_levels"]
    longest = np.sqrt(2) * step + max(abs(a - b) for a in zs for b in zs) + 1e-9
    for a, b in zip(poses, poses[1:]):
        assert np.linalg.norm(np.subtract(a[:3], b[:3])) <= longest


def test_one_height_is_refused():
    from fyp.shared.hardware.safety import WorkspaceEnvelope
    with pytest.raises(ValueError, match="coplanar"):
        calibration_grid([0.5, -0.3], 0.1, 3, [0.3, 0.3], [0, 3.14, 0], WorkspaceEnvelope())


def test_positions_outside_the_envelope_are_left_out():
    from fyp.shared.hardware.safety import WorkspaceEnvelope
    env = WorkspaceEnvelope()
    poses, rejected = calibration_grid([0.5, -0.3], 5.0, 3, [0.3, 0.4], [0, 3.14, 0], env)
    assert rejected, "a 10 m wide grid cannot fit"
    assert all(env.contains(p[:3]) for p in poses)


def test_the_ball_offset_is_rotated_with_the_tool():
    """
    Pointing down, tool z is base -z, so an offset back towards the flange
    (negative tool z) is UP in base frame. Adding the offset unrotated would put
    the ball 4 cm away from where it is.
    """
    down = [0.0, np.pi, 0.0]          # tool z along base -z
    c = ball_centre_in_base([0.5, -0.3, 0.3, *down], [0.0, 0.0, -0.02])
    assert np.allclose(c, [0.5, -0.3, 0.32])


def test_the_ball_offset_at_identity_is_added_as_it_is():
    c = ball_centre_in_base([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.01, 0.02, 0.03])
    assert np.allclose(c, [0.11, 0.22, 0.33])


def test_one_wrong_detection_is_dropped_and_the_fit_recovered():
    """
    The case the rejection exists for: one point centimetres off, the rest within
    a millimetre. The kept fit must recover the true transform.
    """
    T = a_transform()
    p_cam = cloud(15)
    p_base = camera_to_base(p_cam, T) + RNG.normal(0, 0.0005, size=(15, 3))
    p_base[4] += [0.05, -0.03, 0.02]
    T_fit, keep = fit_with_outlier_rejection(p_cam, p_base)
    assert not keep[4]
    assert keep.sum() == 14
    assert np.allclose(T_fit, T, atol=2e-3)


def test_a_clean_fit_drops_nothing():
    T = a_transform()
    p_cam = cloud(12)
    p_base = camera_to_base(p_cam, T) + RNG.normal(0, 0.0005, size=(12, 3))
    _, keep = fit_with_outlier_rejection(p_cam, p_base)
    assert keep.all()


# ------------------------------------------------------------ level to table

def test_levelling_removes_a_height_and_tilt_error_from_the_table():
    """
    The real case of 13 Oct 2026: a ball fit that put the table about 35 mm low
    and slightly tilted. Seen through the wrong transform, a flat table at z=0.172
    comes out tilted and offset; after levelling it must be flat at 0.172 again,
    and a point on the table must keep its x and y.
    """
    from fyp.architecture_a.calibration.hand_eye import fit_plane, level_to_plane
    from fyp.shared.helpers.rotations import euler_to_R

    T_true = a_transform(rpy=(-2.2, 0.05, 3.0), t=(0.70, 0.67, 0.88))
    err = np.eye(4)
    err[:3, :3] = euler_to_R([0.02, -0.015, 0.0])     # about 1 deg each way
    err[:3, 3] = [0.0, 0.0, -0.035]
    T_wrong = err @ T_true

    xy = RNG.uniform([0.30, -0.55], [0.60, -0.25], size=(3000, 2))
    table = np.column_stack([xy, np.full(len(xy), 0.172)])
    table[:, 2] += RNG.normal(0, 0.002, len(table))
    clutter = np.column_stack([RNG.uniform([0.40, -0.45], [0.45, -0.40], size=(200, 2)),
                               RNG.uniform(0.18, 0.26, 200)])          # an object on it
    p_cam = seen_by_camera(np.vstack([table, clutter]), T_true)
    seen = camera_to_base(p_cam, T_wrong)

    normal, centroid, keep = fit_plane(seen)
    T_fixed = level_to_plane(T_wrong, normal, centroid, 0.172)

    table_fixed = camera_to_base(p_cam[:len(table)], T_fixed)
    assert abs(table_fixed[:, 2].mean() - 0.172) < 0.001
    assert table_fixed[:, 2].std() < 0.003
    assert keep[len(table):].mean() < 0.2               # the object is not table
    mid = np.array([0.45, -0.40, 0.172])
    moved = camera_to_base(seen_by_camera(mid, T_true), T_fixed)
    assert np.allclose(moved, mid, atol=0.005)

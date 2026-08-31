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

from fyp.architecture_a.calibration.hand_eye import (base_to_camera, camera_to_base,
                                                     load, report, residuals, save,
                                                     solve_rigid_transform)
from fyp.shared.helpers.rotations import euler_to_R

RNG = np.random.default_rng(7)


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

def test_camera_to_base_and_back_is_a_round_trip():
    T = a_transform()
    p_cam = cloud(5)

    recovered = base_to_camera(camera_to_base(p_cam, T), T)
    np.testing.assert_allclose(recovered, p_cam, atol=1e-9)


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

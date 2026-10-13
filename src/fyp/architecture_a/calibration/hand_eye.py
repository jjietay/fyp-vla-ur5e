""" hand_eye.py

Architecture A stage 4: converts CAMERA-frame 3D points into ROBOT BASE-frame
3D points. This is the only thing standing between stage 3 (which already gives
sub-millimetre camera-frame XYZ) and the pick/place primitives, which can only
speak base frame.

Setup assumed here is EYE-TO-HAND: the camera stands still in the world,
viewing the table from the side at roughly 45 degrees, NOT mounted on the arm.
So the unknown is one transform T_base_cam. The lab is shared, so the camera is
taped in place and taken down between sessions, which means T_base_cam holds
for one session and is re-measured at the start of the next.

How it is obtained (decided 12 Aug 2026, no fiducial marker):
- MEASURED from the mount with a rule and written by scripts/write_calibration.py.
  This file only loads it. check_calibration.py is the only check on it, because
  a measured transform has no residual of its own.
- FITTED per session by scripts/calibrate_ball.py (added 13 Oct 2026), which
  runs the fallback below automatically. `ball_centre_in_base` and
  `calibration_grid` below are its testable parts.
- FALLBACK if the measured one is too coarse to grasp with: a matte ball held in
  the gripper, the arm moved to N poses, the ball centre seen in the camera and
  the TCP read from get_state. That gives N pairs of the SAME physical point in
  two frames, and `solve_rigid_transform` below (Kabsch) finds the rotation and
  translation that best maps one cloud onto the other. It needs at least 3
  non-collinear points; use 10-20 spread across the workspace, or the fit is
  accurate only where you sampled.

CAREFUL: this solves camera -> BASE. Everything downstream must stay in base
frame, because that is the only frame the arm understands: get_state reports the
TCP in base frame and moveL expects targets in it. There is no separate world
frame on the real cell, so do not invent one, and do not mix in coordinates
measured against the table or the room without converting them first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fyp.shared.helpers.rotations import rotvec_to_R
from fyp.shared.helpers.transforms import transform_point


def solve_rigid_transform(p_cam: np.ndarray, p_base: np.ndarray) -> np.ndarray:
    """
    It takes matched 3D points seen in the camera and the same points measured
    in the robot base frame, and gives you the 4x4 transform T_base_cam that
    maps the first onto the second.

    Kabsch algorithm. Centre both clouds on their means so translation drops
    out, then the rotation that best aligns them comes straight from the SVD of
    the cross-covariance. Translation is recovered afterwards from the centroids.
    """
    p_cam = np.asarray(p_cam, dtype=float).reshape(-1, 3)
    p_base = np.asarray(p_base, dtype=float).reshape(-1, 3)

    if len(p_cam) != len(p_base):
        raise ValueError(f"point counts differ: {len(p_cam)} camera, {len(p_base)} base")
    if len(p_cam) < 3:
        raise ValueError(f"need at least 3 point pairs, got {len(p_cam)}")

    cam_centre = p_cam.mean(axis=0)
    base_centre = p_base.mean(axis=0)
    A = p_cam - cam_centre
    B = p_base - base_centre

    if np.linalg.matrix_rank(A, tol=1e-6) < 3:
        raise ValueError(
            "camera points are collinear or coplanar - the rotation is not "
            "determined. Sample poses that vary in all three axes, height included."
        )

    U, _S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))

    # without this the SVD can hand back a reflection (det = -1), which fits the
    # points just as well but mirrors the workspace
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = base_centre - R @ cam_centre
    return T


def residuals(T: np.ndarray, p_cam: np.ndarray, p_base: np.ndarray) -> np.ndarray:
    """
    It takes a calibration and the points it was fitted on, and gives you how
    far off each one is, in metres.

    A calibration always produces a number. This is what tells you whether to
    believe it.
    """
    predicted = transform_point(T, np.asarray(p_cam, dtype=float).reshape(-1, 3))
    return np.linalg.norm(predicted - np.asarray(p_base, dtype=float).reshape(-1, 3), axis=1)


def report(T: np.ndarray, p_cam: np.ndarray, p_base: np.ndarray) -> dict:
    """
    It takes a calibration and its points and gives you the numbers you would
    put in a report: RMS, worst point, and which sample that was.
    """
    e = residuals(T, p_cam, p_base)
    return {
        "n_points": int(len(e)),
        "rms_mm": float(np.sqrt(np.mean(e ** 2)) * 1000.0),
        "max_mm": float(e.max() * 1000.0),
        "worst_index": int(np.argmax(e)),
        "per_point_mm": (e * 1000.0).tolist(),
    }


def save(T: np.ndarray, path: str | Path, meta: dict | None = None) -> None:
    """
    It takes a calibration and writes it to JSON so every later run reuses the
    same numbers instead of re-deriving them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"T_base_cam": np.asarray(T, dtype=float).tolist()}
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2))


def load(path: str | Path) -> np.ndarray:
    """
    It takes a path and gives you back the 4x4 transform saved there.
    """
    data = json.loads(Path(path).read_text())
    T = np.asarray(data["T_base_cam"], dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{path} holds a {T.shape} array, expected (4, 4)")
    return T


def camera_to_base(p_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """
    It takes a point the camera saw and gives you where the robot should reach.

    The one function the rest of the pipeline actually calls. Stage 3 hands you
    p_cam, this hands stage 6 something it can move to.
    """
    return transform_point(T_base_cam, p_cam)


def base_to_camera(p_base: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """
    It takes a point in the robot's frame and gives you where the camera sees it.

    The inverse direction, used for checking: project a pose the arm is holding
    back into the image and confirm it lands on the marker.
    """
    T = np.asarray(T_base_cam, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return transform_point(T_inv, p_base)


def ball_centre_in_base(tcp_pose, offset_tool) -> np.ndarray:
    """
    It takes the TCP pose the arm reports and where the ball centre sits in the
    tool frame, and gives you the ball centre in base frame.

    The offset is rotated by the tool orientation before it is added. Adding it
    straight onto the TCP position is the same only while the tool points
    exactly down, and the error appears the first time anything else is tried.

    CAREFUL: with every sample at the same tool orientation, as
    `calibrate_ball.py` does, an error in `offset_tool` is the same base-frame
    shift at every point. The fit absorbs it into the translation and the
    residual stays small, so the residual cannot see it. Only a reach test can.
    """
    p = np.asarray(tcp_pose, dtype=float).reshape(6)
    return p[:3] + rotvec_to_R(p[3:]) @ np.asarray(offset_tool, dtype=float).reshape(3)


def calibration_grid(centre_xy, half_extent: float, n: int, z_levels,
                     rotvec, envelope) -> tuple[list[list[float]], list[list[float]]]:
    """
    It takes the middle of the area to sample, its half width, a grid size and
    a list of heights, and gives you (poses inside the envelope, positions left
    out because they were not).

    An n by n square at every height, all at one tool orientation. Rows are
    walked in a snake so consecutive moves are short and none crosses the whole
    table. More than one height is required, because points that all share a z
    are coplanar and `solve_rigid_transform` refuses them, rightly: a flat cloud
    leaves the rotation about the table plane undetermined.
    """
    z_levels = [float(z) for z in z_levels]
    if len(set(z_levels)) < 2:
        raise ValueError("need at least two different heights, or the points are coplanar")
    if int(n) < 2:
        raise ValueError(f"grid size must be at least 2, got {n}")

    cx, cy = (float(v) for v in centre_xy)
    offsets = np.linspace(-float(half_extent), float(half_extent), int(n))
    rv = [float(v) for v in np.asarray(rotvec, dtype=float).reshape(3)]

    # One level walked as a snake, then every second level walked backwards, so
    # the step up to the next height starts where the last level ended.
    level = [(float(dx), float(dy))
             for i, dx in enumerate(offsets)
             for dy in (offsets if i % 2 == 0 else offsets[::-1])]

    poses, rejected = [], []
    for k, z in enumerate(z_levels):
        for dx, dy in (level if k % 2 == 0 else level[::-1]):
            xyz = [cx + dx, cy + dy, z]
            if envelope.contains(xyz):
                poses.append(xyz + rv)
            else:
                rejected.append(xyz)
    return poses, rejected


def fit_with_outlier_rejection(p_cam, p_base, factor: float = 3.0,
                               floor_m: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    """
    It takes matched camera and base points and gives you (T_base_cam, a mask of
    the points it kept).

    Fits once, drops points whose residual is more than `factor` times the
    median and above `floor_m`, and fits again. One wrong detection, such as the
    detector boxing a gripper finger instead of the ball, pulls a least squares
    fit by centimetres, while the honest points disagree by millimetres. The floor
    stops a very good fit from discarding points over sub-millimetre noise.

    One pass only. Repeating it until nothing is dropped can eat a calibration
    one point at a time.
    """
    p_cam = np.asarray(p_cam, dtype=float).reshape(-1, 3)
    p_base = np.asarray(p_base, dtype=float).reshape(-1, 3)
    T = solve_rigid_transform(p_cam, p_base)
    e = residuals(T, p_cam, p_base)
    keep = (e <= max(factor * float(np.median(e)), floor_m))
    if keep.all() or keep.sum() < 3:
        return T, np.ones(len(e), dtype=bool)
    try:
        return solve_rigid_transform(p_cam[keep], p_base[keep]), keep
    except ValueError:
        # What is left is coplanar. Keeping everything is the lesser evil, and the
        # large residual it reports says so.
        return T, np.ones(len(e), dtype=bool)

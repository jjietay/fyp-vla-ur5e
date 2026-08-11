""" hand_eye.py

Architecture A stage 4: converts CAMERA-frame 3D points into ROBOT BASE-frame
3D points. This is the only thing standing between stage 3 (which already gives
sub-millimetre camera-frame XYZ) and the pick/place primitives, which can only
speak base frame.

Setup assumed here is EYE-TO-HAND: the camera is bolted to the world looking
down at the table, NOT mounted on the arm. So the unknown is one fixed transform
T_base_cam, and it never changes unless the camera is knocked.

How it is solved:
- Move the arm to N poses. At each one you know where the TCP is in base frame
  (from get_state) and you can see a marker on the gripper in the camera.
- That gives N pairs of the SAME physical point expressed in two frames.
- Kabsch/Procrustes finds the single rotation + translation that best maps one
  cloud onto the other. No iteration, no initial guess, one SVD.

Needs at least 3 non-collinear points to be solvable at all; use 10-20 spread
across the whole workspace or the fit will be accurate only where you sampled.

CAREFUL: this solves camera -> BASE, not camera -> world. In the MuJoCo scene
the arm base carries quat="0 0 0 -1", so base and world are NOT the same frame.
Feeding world-frame targets into a base-frame transform gives an answer that
looks plausible and is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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

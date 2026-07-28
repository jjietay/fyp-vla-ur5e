"""Pinhole camera model: pixels + depth <-> camera-frame 3D points.

This is the maths half of Architecture A stage 3. It is deliberately free of
MuJoCo: the same equations run against a real RGB-D camera, only the source of
`CameraIntrinsics` changes (derived from fovy in sim, from a checkerboard
calibration on hardware).

DEPTH CONVENTION
    Depth here is *z-depth*: the perpendicular distance from the camera plane
    along the viewing axis, NOT the length of the ray to the surface. The two
    agree only at the principal point and diverge towards the image edges.
    MuJoCo's buffer is z-depth (measured, not assumed - see check_camera.py),
    so it divides straight into the model below with no ray-length correction.

CAMERA FRAME (MuJoCo / OpenGL convention)
    +X right, +Y UP, camera looks down its own -Z.
    Image rows run downward, so image +v maps to camera -Y. Back-projection
    must flip that sign; see `pixel_to_camera`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    def __repr__(self) -> str:
        return (f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> CameraIntrinsics:
    if not (0.0 < fovy_deg < 180.0):
        raise ValueError(f"fovy must be in (0, 180) degrees, got {fovy_deg}")
    f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return CameraIntrinsics(fx=float(f), fy=float(f),
                            cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                            width=int(width), height=int(height))


def pixel_to_camera(u, v, depth, intr: CameraIntrinsics) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(depth, dtype=float)
    xyz = np.stack([(u - intr.cx) * z / intr.fx,
                    -(v - intr.cy) * z / intr.fy,
                    -z], axis=-1)
    return xyz


def camera_to_pixel(p_cam, intr: CameraIntrinsics) -> tuple:
    p = np.asarray(p_cam, dtype=float)
    depth = -p[..., 2]
    if np.any(depth <= 0):
        raise ValueError("point(s) at or behind the camera plane")
    u = intr.fx * (p[..., 0] / depth) + intr.cx
    v = intr.cy - intr.fy * (p[..., 1] / depth)
    return u, v, depth


def surface_to_centroid(p_cam: np.ndarray, half_height: float) -> np.ndarray:
    out = np.array(p_cam, dtype=float, copy=True)
    out[..., 2] -= half_height
    return out


def depth_at(depth: np.ndarray, u: float, v: float, radius: int = 2,
             max_depth: float | None = None) -> float:
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return float("nan")
    patch = depth[max(0, vi - radius):vi + radius + 1,
                  max(0, ui - radius):ui + radius + 1]
    valid = np.isfinite(patch) & (patch > 0)
    if max_depth is not None:
        valid &= patch < max_depth
    if not valid.any():
        return float("nan")
    return float(np.median(patch[valid]))

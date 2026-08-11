""" pixel_to_3d.py

This file represents the Pinhole camera model where:
pixels + depth <---> camera-frame 3D points

- the equations are universal and works with a real RGB-D camera
- convention is depth is z-depth which is the perpendicular distance from
  camera plane along viewing axis
- mjc's buffer is z-depth
"""

from __future__ import annotations
import numpy as np

from fyp.shared.hardware.intrinsics import CameraIntrinsics


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> CameraIntrinsics:
    """
    This function uses field of view in y direction (fovy) to update camera intrinsic values.
    Condition is that the fovy is between 0 and 180 degrees.
    """
    if not (0.0 < fovy_deg < 180.0):
        raise ValueError(f"fovy must be in (0, 180) degrees, got {fovy_deg}")
    f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return CameraIntrinsics(fx=float(f), fy=float(f),
                            cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                            width=int(width), height=int(height))


def pixel_to_camera(u, v, depth, intr: CameraIntrinsics) -> np.ndarray:
    """
    This function converts a 2d pixel with coordinate (u,v) and depth
    into a 3d coordinates (x,y,z) with respect to the camera
    as the global frame (0,0,0).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(depth, dtype=float)
    xyz = np.stack([(u - intr.cx) * z / intr.fx,
                    -(v - intr.cy) * z / intr.fy,
                    -z], axis=-1)
    return xyz


def camera_to_pixel(p_cam, intr: CameraIntrinsics) -> tuple:
    """
    This function converts the 3d coordinate with respect to the global frame
    where camera == (0,0,0) back into the 2d pixel.

    p_cam (position in camera frame) just means the x,y,z vector
    we calculated earlier via pixel_to_camera.
    """
    p = np.asarray(p_cam, dtype=float)
    depth = -p[..., 2]
    if np.any(depth <= 0):
        raise ValueError("point(s) at or behind the camera plane")
    u = intr.fx * (p[..., 0] / depth) + intr.cx
    v = intr.cy - intr.fy * (p[..., 1] / depth)
    return u, v, depth


def surface_to_centroid(p_cam: np.ndarray, half_height: float) -> np.ndarray:
    """
    This function is used to find the 3d point from camera (0,0,0) to the
    center of the object. Since we know the coordinate of the face of object,
    we can add the extra distance that is half of depth of thickness of obj.
    """
    out = np.array(p_cam, dtype=float, copy=True)
    out[..., 2] -= half_height
    return out


def depth_at(depth: np.ndarray, u: float, v: float, radius: int = 2,
             max_depth: float | None = None) -> float:
    """
    This function calculates a more accurate depth reading of
    a point at (u,v) by taking median with points around it.

    example:    a b c
                d e f
                g h i
                
        - point e's depth is to be calculated
        - radius 1 means we take a b c d e f g h i's depth and find median
    """
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

""" pixel_to_3d.py

It takes a pixel and a depth reading and gives you a 3D point in the camera's
own frame, and back again.

The pinhole camera model. Depth is z-depth, meaning the perpendicular distance
from the camera plane along the viewing axis, which is what a RealSense reports.

## The axis convention, and why it changed on 31 Aug 2026

This file uses the **OpenCV** camera frame:

    x  right across the image
    y  DOWN the image
    z  FORWARD, away from the camera, POSITIVE

That is the convention `pyrealsense2` deprojects into, and it is the convention
the intrinsics assume. `shared/hardware/camera.py` reads fx, fy, ppx and ppy
straight from the SDK and converts depth to positive metres, so the numbers
arriving here are OpenCV numbers and the maths has to match them.

It did not, until 31 Aug. `pixel_to_camera` negated y and z, which is the
**OpenGL** convention MuJoCo uses, inherited from the simulation this project
started in. Nothing caught it, because `camera_to_pixel` negated them right back,
so the round trip was exact and every test of it passed. Everything was
self-consistent and only the physical camera disagreed.

It would not have stayed harmless. A FITTED `T_base_cam` absorbs a fixed axis
flip silently, because Kabsch finds whatever map takes one cloud to the other.
The Calibration Marker decision closed on **no marker**, so `T_base_cam` is
measured from the mount, and a measured transform describes where the camera
physically is. It cannot absorb a flip that is not physically there. Every
back-projected point would have landed mirrored in y and behind the camera in z,
and it would have read as a badly wrong calibration.

If you ever point this at a MuJoCo buffer again, negate y and z on the way in
rather than changing the functions below.
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
    It takes a pixel and its depth and gives you the 3D point in the camera
    frame, with the camera itself at the origin.

    OpenCV axes, so y increases DOWN the image and z is positive going away from
    the camera. A point on the table in front of the camera comes back with a
    positive z of roughly the standoff distance. A negative z means the flip
    described in the module docstring has come back.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(depth, dtype=float)
    xyz = np.stack([(u - intr.cx) * z / intr.fx,
                    (v - intr.cy) * z / intr.fy,
                    z], axis=-1)
    return xyz


def camera_to_pixel(p_cam, intr: CameraIntrinsics) -> tuple:
    """
    This function converts the 3d coordinate with respect to the global frame
    where camera == (0,0,0) back into the 2d pixel.

    p_cam (position in camera frame) just means the x,y,z vector
    we calculated earlier via pixel_to_camera.
    """
    p = np.asarray(p_cam, dtype=float)
    depth = p[..., 2]
    if np.any(depth <= 0):
        raise ValueError("point(s) at or behind the camera plane")
    u = intr.fx * (p[..., 0] / depth) + intr.cx
    v = intr.fy * (p[..., 1] / depth) + intr.cy
    return u, v, depth


def surface_to_centroid(p_cam: np.ndarray, half_depth: float) -> np.ndarray:
    """
    It takes the point where the camera sees the near FACE of an object and
    gives you the point at its centre.

    The depth reading lands on the surface facing the camera, and a grasp wants
    the middle, so this pushes the point `half_depth` metres further away ALONG
    ITS OWN RAY from the camera. For a round object the centre lies on that ray.
    Pushing along the optical axis instead, as this did until 1 Oct 2026, is the
    same only at the image centre: 30 degrees off axis it lands about half the
    push to one side.
    """
    out = np.array(p_cam, dtype=float, copy=True)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out * (1.0 + half_depth / norm)


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


def smooth_depth(depth: np.ndarray, radius: int = 2,
                 max_depth: float | None = None) -> np.ndarray:
    """
    It takes a whole depth image and gives you the depth `depth_at` would return
    at every pixel, NaN where it would return NaN.

    Only for LOOKING at what the pipeline sees. The pipeline itself still calls
    `depth_at` on the few pixels it needs, which is far cheaper than this.

    Same rule as `depth_at`, pixel for pixel: the median of the valid readings in
    a (2r+1)^2 patch, where valid means finite, above 0 and below max_depth, and
    a patch at the border is cut short rather than padded with made-up values.
    The NaN padding below does that cutting, because nanmedian skips it.
    """
    d = np.asarray(depth, dtype=np.float32)
    bad = ~np.isfinite(d) | (d <= 0)
    if max_depth is not None:
        bad |= d >= max_depth
    d = np.where(bad, np.nan, d)
    padded = np.pad(d, radius, mode="constant", constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(padded, (2 * radius + 1,) * 2)
    import warnings
    with warnings.catch_warnings():
        # an all-hole patch is the expected NaN, not something to warn about
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(win, axis=(-2, -1))


def ray_to_plane(u: float, v: float, intr: CameraIntrinsics, T_base_cam,
                 z: float) -> np.ndarray | None:
    """
    It takes a pixel, the intrinsics, the camera-to-base transform and a height,
    and gives you the base-frame point where that pixel's ray reaches that height,
    or None when it never does (the ray points away from the plane).

    No depth reading is used. The direction of a pixel is the thing this camera
    measures well; how far away is the thing it measures worst.
    """
    T = np.asarray(T_base_cam, dtype=float)
    d = T[:3, :3] @ np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
    o = T[:3, 3]
    if abs(d[2]) < 1e-9:
        return None
    s = (float(z) - o[2]) / d[2]
    if s <= 0:
        return None
    return o + s * d

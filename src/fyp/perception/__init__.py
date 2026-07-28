"""Perception stages of Architecture A.

frame -> detector -> depth-to-3D -> camera->base transform

Only the frame source changes when hardware arrives (MuJoCo Renderer -> RGB-D
camera); everything downstream of `render_rgbd` is hardware-agnostic.
"""

from fyp.perception.camera import (
    CameraIntrinsics,
    camera_to_pixel,
    depth_at,
    intrinsics_from_fovy,
    intrinsics_for_camera,
    pixel_to_camera,
    render_rgbd,
    surface_to_centroid,
)

__all__ = [
    "CameraIntrinsics",
    "camera_to_pixel",
    "depth_at",
    "intrinsics_from_fovy",
    "intrinsics_for_camera",
    "pixel_to_camera",
    "render_rgbd",
    "surface_to_centroid",
]

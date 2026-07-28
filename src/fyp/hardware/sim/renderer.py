"""RGB-D frame source backed by MuJoCo. Architecture A stage 1.

This is the sim half of the camera. The pinhole maths it feeds
(`helpers/pixel_to_depth.py`) is hardware-agnostic and survives the move to a
real RGB-D camera; everything in THIS file does not:

  - `render_rgbd` becomes a RealSense frame grab
  - `intrinsics_for_camera` becomes a checkerboard calibration read off disk

Both produce the same two arrays — an (H,W,3) uint8 RGB image and an (H,W)
float32 depth map in metres — so nothing downstream changes.

Note that in sim the RGB and depth buffers come from a single `update_scene`
call and are therefore pixel-aligned by construction. A real RGB-D camera has
its colour and depth sensors at different physical points and needs an explicit
registration step, so do not assume alignment on hardware.
"""
from __future__ import annotations

import numpy as np

from fyp.helpers.pixel_to_depth import CameraIntrinsics, intrinsics_from_fovy


def intrinsics_for_camera(model, camera: str, width: int, height: int) -> CameraIntrinsics:
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"no camera named {camera!r} in the model")
    return intrinsics_from_fovy(float(model.cam_fovy[cam_id]), width, height)


def render_rgbd(
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
    scene: str | None = None,
    data=None,
    model=None,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    import mujoco

    from fyp.helpers.config import get_config, resolve

    sim = get_config()["sim"]
    cam = camera or sim["camera"]["name"]

    if model is None:
        model = mujoco.MjModel.from_xml_path(str(resolve(scene or sim["scene"])))
        data = None
    if data is None:
        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().copy()


        renderer.enable_depth_rendering()
        depth = renderer.render().copy().astype(np.float32)
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    intr = intrinsics_for_camera(model, cam, width, height)
    return rgb, depth, intr

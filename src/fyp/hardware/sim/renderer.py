""" renderer.py

This file renders the image and captures the scene using camera in MuJoCo.
Captured format include both RGB and depth as shown:

1) (H,W,3) uint8 RGB image
2) (H,W) float32 depth image in metres
"""
from __future__ import annotations

import numpy as np

from fyp.helpers.pixel_to_3d import CameraIntrinsics, intrinsics_from_fovy


def intrinsics_for_camera(model, camera: str, width: int, height: int) -> CameraIntrinsics:
    """
    This function only reads fovy from the model (mjc's compiled scene including
    arms gripper cubes bin camera actuators, and passes them to
    the CameraIntrinsics class's method intrinsics_from_fovy.
    """
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
    """
    This function is takes a photograph of the MuJoCo scene from a named camera
    and returns 3 things:
    - colour image
    - depth map in metres
    - intrinsics for that particular render
    """
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
        renderer.update_scene(data, camera=cam) # 1 snapshot of everything
        rgb = renderer.render().copy() # read that snapshot as colour
        renderer.enable_depth_rendering() # enable depth
        depth = renderer.render().copy().astype(np.float32) # read as depth
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    intr = intrinsics_for_camera(model, cam, width, height)
    return rgb, depth, intr

""" scene.py

Load scene by performing forward dynamics. Checking camera pose, block and object coordinates.
This is only for MuJoCo's elements.
"""
from __future__ import annotations

import numpy as np

from fyp.helpers.config import get_config, resolve

BLOCKS = ["block_red", "block_green", "block_blue", "block_yellow"]


def load_scene(scene_key: str = "scene"):
    """
    This function loads the model and data performs forward dynamics
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(resolve(get_config()["sim"][scene_key])))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0) # load save file #0 into data
    mujoco.mj_forward(model, data) # forward dynamics
    return model, data


def camera_pose(model, data, camera: str):
    """
    We input a camera name and this camera will give back its pose and which way
    its pointing at wrt the world coordinates"""
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise ValueError(f"no camera named {camera!r} in the model")
    return (np.array(data.cam_xpos[cid], dtype=float),
            np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3))


def world_to_camera(p_world, cam_pos, cam_mat) -> np.ndarray:
    """
    finds the camera coordinates (a point) with respect to the world/global coordinates"""
    return cam_mat.T @ (np.asarray(p_world, dtype=float) - cam_pos)


def block_truth(model, data) -> dict:
    """
    as the name implies, it is the single source of truth for the blocks.
    It can read from the model, where each cube really is."""
    import mujoco

    out = {}
    for name in BLOCKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        if bid < 0 or gid < 0:
            continue
        centre = np.array(data.xpos[bid], dtype=float)
        half_z = float(model.geom_size[gid][2])
        out[name] = {"centroid": centre,
                     "top": centre + np.array([0.0, 0.0, half_z]),
                     "half_z": half_z}
    return out


def body_position(model, data, name: str) -> np.ndarray | None:
    """
    This function takes in an object's name and gives its current position"""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return np.array(data.xpos[bid], dtype=float)

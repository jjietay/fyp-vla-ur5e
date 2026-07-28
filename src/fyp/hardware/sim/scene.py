"""Scene loading and MuJoCo ground truth.

Sim-only, and worth being explicit about WHY: every function here answers
"where is this object really?" by reading the model. On real hardware there is
no such oracle — verification becomes repeatability and measurement against a
physical fixture. So none of this transfers, and no downstream code should come
to depend on ground truth being available.

Reading truth from the model (rather than hardcoding numbers) does mean the
checks keep working when the camera or the props move.
"""
from __future__ import annotations

import numpy as np

from fyp.helpers.config import get_config, resolve

BLOCKS = ["block_red", "block_green", "block_blue", "block_yellow"]


def load_scene(scene_key: str = "scene"):
    """Load the configured scene and reset it to the `home` keyframe."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(resolve(get_config()["sim"][scene_key])))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def camera_pose(model, data, camera: str):
    """(position, 3x3 rotation) of a named camera, in the WORLD frame.

    In sim this is exact. On hardware the equivalent quantity is the output of
    hand-eye calibration and carries real uncertainty.
    """
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise ValueError(f"no camera named {camera!r} in the model")
    return (np.array(data.cam_xpos[cid], dtype=float),
            np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3))


def world_to_camera(p_world, cam_pos, cam_mat) -> np.ndarray:
    """World point -> camera frame. Columns of cam_mat are the camera axes in world."""
    return cam_mat.T @ (np.asarray(p_world, dtype=float) - cam_pos)


def block_truth(model, data) -> dict:
    """True top-face centre, centroid and half-height of each block, WORLD frame."""
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
    """World position of a named body, or None if it is not in the model."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return np.array(data.xpos[bid], dtype=float)

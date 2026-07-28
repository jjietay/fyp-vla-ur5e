"""RGB-D rendering and pinhole intrinsics for the workspace camera.

This is Architecture A stage 1 (frame) plus the camera model stage 3
(depth-to-3D) needs. In simulation the frame comes from `mujoco.Renderer`; on
hardware it comes from an RGB-D camera. Both produce the same two arrays -
an (H,W,3) uint8 RGB image and an (H,W) float32 depth map in metres - so
everything downstream is unchanged when the real camera arrives. The only
thing that must be swapped is `intrinsics_*`: in sim we derive them from the
MuJoCo camera's fovy, on hardware you read them off the camera's calibration.

DEPTH CONVENTION
    MuJoCo's depth buffer is *z-depth*: the perpendicular distance from the
    camera plane along the camera's viewing axis, NOT the length of the ray to
    the surface. Those agree only at the principal point and diverge towards
    the image edges, so the distinction matters for back-projection. The
    pinhole model below assumes z-depth, which is what MuJoCo gives us.

CAMERA FRAME (MuJoCo / OpenGL convention)
    +X right, +Y UP, camera looks down its own -Z.
    Image rows run downward, so image +v maps to camera -Y. Back-projection
    must flip that sign; see `pixel_to_camera` (added with stage 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    def __repr__(self) -> str:  # keeps verification output readable
        return (f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> CameraIntrinsics:
    """Derive pinhole intrinsics from a MuJoCo camera's vertical FOV.

    MuJoCo specifies only `fovy` (vertical field of view, degrees) and assumes
    square pixels, so fx == fy and the horizontal FOV falls out of the aspect
    ratio. The principal point sits at the image centre.

        fy = (height / 2) / tan(fovy / 2)

    Principal point uses (n - 1) / 2, not n / 2: pixel i is sampled at its
    centre, so index 0 is coordinate 0.0 and the centre of a 480-row image is
    239.5. The half-pixel matters once you back-project - at 0.85 m it is
    ~0.8 mm of lateral error, small but free to get right.
    """
    if not (0.0 < fovy_deg < 180.0):
        raise ValueError(f"fovy must be in (0, 180) degrees, got {fovy_deg}")
    f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return CameraIntrinsics(fx=float(f), fy=float(f),
                            cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                            width=int(width), height=int(height))


def intrinsics_for_camera(model, camera: str, width: int, height: int) -> CameraIntrinsics:
    """Read `fovy` off a named MuJoCo camera and build its intrinsics."""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"no camera named {camera!r} in the model")
    return intrinsics_from_fovy(float(model.cam_fovy[cam_id]), width, height)


def pixel_to_camera(u, v, depth, intr: CameraIntrinsics) -> np.ndarray:
    """Back-project pixel(s) + z-depth to camera-frame XYZ. Stage 3 of Arch A.

    Accepts scalars or arrays; returns (3,) or (N, 3).

        X = (u - cx) * Z / fx
        Y = -(v - cy) * Z / fy        <- image rows run DOWN, camera +Y is UP
        Z = -depth                     <- camera looks down its own -Z

    The two sign flips are the whole subtlety. Verified empirically against
    MuJoCo: the depth buffer is z-depth (perpendicular distance from the camera
    plane), so `depth` divides in directly with no ray-length correction.

    NOTE what this point physically IS: the camera sees an object's TOP SURFACE,
    so back-projecting a detection's box centre gives a point on top of the
    object, not its centroid. `pick()` must offset downward to get a grasp
    centre - see `surface_to_centroid`.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(depth, dtype=float)
    xyz = np.stack([(u - intr.cx) * z / intr.fx,
                    -(v - intr.cy) * z / intr.fy,
                    -z], axis=-1)
    return xyz


def camera_to_pixel(p_cam, intr: CameraIntrinsics) -> tuple:
    """Forward projection - the exact inverse of `pixel_to_camera`.

    Kept beside it so the round-trip can be tested to machine precision; also
    used to predict where a known world point should appear.
    """
    p = np.asarray(p_cam, dtype=float)
    depth = -p[..., 2]
    if np.any(depth <= 0):
        raise ValueError("point(s) at or behind the camera plane")
    u = intr.fx * (p[..., 0] / depth) + intr.cx
    v = intr.cy - intr.fy * (p[..., 1] / depth)
    return u, v, depth


def surface_to_centroid(p_cam: np.ndarray, half_height: float) -> np.ndarray:
    """Shift a top-surface point down to the centroid of an object of known height.

    The camera only ever sees an object's top face, but a grasp wants its centre.
    For a cube of half-size h resting on the table, the centroid sits h below the
    observed top surface.

    ASSUMES A TOP-DOWN CAMERA - specifically that the camera's Z axis is parallel
    to world Z, which holds for the `workspace` camera (xyaxes "1 0 0 0 1 0").
    Then "down in the world" is exactly -Z in camera coordinates and X, Y are
    untouched. For a tilted camera you would instead rotate world -Z into the
    camera frame and step along that, so revisit this when the real RGB-D camera
    is mounted at an angle.
    """
    out = np.array(p_cam, dtype=float, copy=True)
    out[..., 2] -= half_height      # deeper = further from the camera = smaller Z
    return out


def depth_at(depth: np.ndarray, u: float, v: float, radius: int = 2,
             max_depth: float | None = None) -> float:
    """Sample the depth map at (u, v), robustly.

    Takes the median of a (2r+1)^2 patch rather than one pixel: a box centre can
    land on an object edge, and a single reading there straddles the depth
    discontinuity and lands halfway between the object and the table - a
    plausible-looking value that is wrong by centimetres.

    Returns NaN if the pixel is out of frame or the patch has no valid depth.
    `max_depth` rejects background: MuJoCo returns the far-plane distance for
    pixels that hit nothing near, which is a finite number, not inf.
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


def render_rgbd(
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
    scene: str | None = None,
    data=None,
    model=None,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    """Render aligned RGB + depth from one MuJoCo camera.

    Returns
        rgb    (H, W, 3) uint8
        depth  (H, W)    float32, metres, z-depth from the camera plane
        intr   CameraIntrinsics matching this render size

    RGB and depth come from the same `update_scene` call, so the two buffers are
    pixel-aligned by construction - no registration step, unlike a real RGB-D
    camera where the colour and depth sensors sit at different physical points.

    Pass an existing (model, data) pair to capture the *live* scene during an
    episode. With neither, the scene is loaded fresh and reset to the `home`
    keyframe, which is what you want for static verification.
    """
    import mujoco
    from fyp.config import get_config, resolve

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

        # Same scene, depth mode: guarantees the buffers correspond.
        renderer.enable_depth_rendering()
        depth = renderer.render().copy().astype(np.float32)
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    intr = intrinsics_for_camera(model, cam, width, height)
    return rgb, depth, intr

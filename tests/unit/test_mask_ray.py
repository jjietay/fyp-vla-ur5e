"""The mask_ray localiser on a synthetic orange (13 Oct 2026).

A camera placed like the real one (about 0.73, 0.65, 0.86, looking along -y and
about 34 degrees down) renders a sphere resting on the table, pixel by pixel:
mask, depth on the sphere, depth on the table around it. Then the depth on the
sphere is pushed DEEPER, which is what the D435 did to the real orange, and the
two localisers are compared on the same picture.

No torch: the mask is the rendered one, so this checks the geometry, not SAM.
"""
import numpy as np
import pytest

from fyp.architecture_a.perception.localiser import (erode, locate, locate_on_table,
                                                     localise)
from fyp.architecture_a.perception.pixel_to_3d import ray_to_plane
from fyp.shared.hardware.intrinsics import CameraIntrinsics

TABLE_Z = 0.172
R_ORANGE = 0.027
INTR = CameraIntrinsics(fx=640.0, fy=640.0, cx=640.0, cy=360.0, width=1280, height=720)


def _camera() -> np.ndarray:
    """It takes nothing and gives you a T_base_cam like the cell's on 13 Oct 2026."""
    o = np.array([0.725, 0.651, 0.863])
    z = np.array([0.04, -0.824, -0.565]); z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    y = -(up - (up @ z) * z); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = o
    return T


def _render(T, centre, r, bias=0.0, hole=False):
    """
    It takes the camera, a sphere centre and radius, and gives you (mask, depth):
    the sphere's pixels, and z-depth everywhere (sphere front face, else table).
    `bias` adds metres of depth along each sphere ray; `hole` makes it NaN.
    """
    R, o = T[:3, :3], T[:3, 3]
    uu, vv = np.meshgrid(np.arange(INTR.width), np.arange(INTR.height))
    dc = np.stack([(uu - INTR.cx) / INTR.fx, (vv - INTR.cy) / INTR.fy,
                   np.ones_like(uu, dtype=float)], axis=-1)        # camera rays, z = 1
    db = dc @ R.T                                                  # base frame
    oc = o - centre
    a = (db ** 2).sum(-1)
    b = 2 * (db @ oc)
    c = oc @ oc - r * r
    disc = b * b - 4 * a * c
    hit = disc > 0
    s_sphere = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0))) / (2 * a), np.nan)
    s_table = (TABLE_Z - o[2]) / db[..., 2]
    depth = np.where(hit, s_sphere, s_table)                       # s is z-depth, since dc_z = 1
    if bias:
        depth = np.where(hit, depth + bias / np.linalg.norm(dc, axis=-1), depth)
    if hole:
        depth = np.where(hit, np.nan, depth)
    return hit, depth


def _record(mask):
    vs, us = np.nonzero(mask)
    box = [float(us.min()), float(vs.min()), float(us.max()), float(vs.max())]
    return {"query": "orange", "score": 0.6, "box_xyxy": box,
            "center_uv": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}


def _base(obj, T):
    # box_depth works in camera frame; mask_ray already gives base.
    if obj.p_base is not None:
        return obj.p_base
    return (T @ np.append(obj.p_cam, 1.0))[:3]


FAR, NEAR = (0.36, -0.59), (0.62, -0.17)


@pytest.mark.parametrize("xy", [FAR, NEAR])
def test_mask_ray_finds_the_centre_with_honest_depth(xy):
    T = _camera()
    centre = np.array([*xy, TABLE_Z + R_ORANGE])
    mask, depth = _render(T, centre, R_ORANGE)
    (obj,) = locate_on_table([_record(mask)], [mask], depth, INTR, T, TABLE_Z)
    err = np.linalg.norm(_base(obj, T)[:2] - centre[:2])
    assert obj.height_from == "depth_top"
    assert err < 0.006, f"{err * 1000:.1f} mm off"


def test_deep_depth_throws_box_depth_behind_and_mask_ray_less_so():
    """The 13 Oct failure: depth on the whole orange 27 mm too deep, orange far away."""
    T = _camera()
    centre = np.array([*FAR, TABLE_Z + R_ORANGE])
    mask, depth = _render(T, centre, R_ORANGE, bias=0.027)
    rec = _record(mask)

    (old,) = locate([rec], depth, INTR, max_centre_offset=0.040)
    (dep,) = locate_on_table([rec], [mask], depth, INTR, T, TABLE_Z, height_from="depth")
    (siz,) = locate_on_table([rec], [mask], depth, INTR, T, TABLE_Z, height_from="mask_size")
    old_e = _base(old, T)[:2] - centre[:2]
    dep_e = np.linalg.norm(_base(dep, T)[:2] - centre[:2])
    siz_e = np.linalg.norm(_base(siz, T)[:2] - centre[:2])

    # Behind, as the camera sees it, is -y in this cell.
    assert old_e[1] < -0.018, f"box_depth only {old_e[1] * 1000:.0f} mm behind"
    # depth-height still feels a bias that covers the WHOLE object, but less.
    assert dep_e < 0.8 * np.linalg.norm(old_e)
    # size-height never reads the object's depth, so the bias cannot reach it.
    assert siz_e < 0.005, f"mask_size {siz_e * 1000:.1f} mm off"


@pytest.mark.parametrize("xy", [FAR, NEAR])
def test_mask_size_height_is_right_for_a_resting_ball(xy):
    T = _camera()
    centre = np.array([*xy, TABLE_Z + R_ORANGE])
    mask, depth = _render(T, centre, R_ORANGE)
    (obj,) = locate_on_table([_record(mask)], [mask], depth, INTR, T, TABLE_Z,
                             height_from="mask_size")
    assert abs(obj.height_m - 2 * R_ORANGE) < 0.004
    assert np.linalg.norm(_base(obj, T)[:2] - centre[:2]) < 0.004


def test_depth_height_finds_something_not_on_the_table():
    """A drawer-handle stand-in 15 cm above the table: the table must not be assumed."""
    T = _camera()
    centre = np.array([0.50, -0.35, TABLE_Z + 0.15])
    mask, depth = _render(T, centre, 0.015)
    (obj,) = locate_on_table([_record(mask)], [mask], depth, INTR, T, TABLE_Z,
                             height_from="depth")
    p = _base(obj, T)
    assert obj.height_from == "depth_mid"
    # The camera sees more of its top, so the midpoint sits a little high and
    # the point lands a little toward the camera: about 1 cm for a 3 cm knob.
    assert np.linalg.norm(p[:2] - centre[:2]) < 0.012
    assert abs(p[2] - centre[2]) < 0.008


def test_no_depth_on_the_object_falls_back_to_its_pixel_size():
    T = _camera()
    centre = np.array([*FAR, TABLE_Z + R_ORANGE])
    mask, depth = _render(T, centre, R_ORANGE, hole=True)
    (obj,) = locate_on_table([_record(mask)], [mask], depth, INTR, T, TABLE_Z)
    assert obj.height_from == "mask_size"
    assert abs(obj.height_m - 2 * R_ORANGE) < 0.004
    assert np.linalg.norm(_base(obj, T)[:2] - centre[:2]) < 0.004


def test_centre_never_goes_below_the_table():
    T = _camera()
    centre = np.array([*NEAR, TABLE_Z + R_ORANGE])
    mask, depth = _render(T, centre, R_ORANGE, bias=0.5)    # absurdly deep: "under" the table
    (obj,) = locate_on_table([_record(mask)], [mask], depth, INTR, T, TABLE_Z,
                             min_height=0.01)
    assert _base(obj, T)[2] == pytest.approx(TABLE_Z + 0.005)


def test_height_from_is_checked():
    with pytest.raises(ValueError, match="mask_size"):
        locate_on_table([], [], np.zeros((2, 2)), INTR, np.eye(4), TABLE_Z, height_from="x")


def test_erode_shrinks_a_square_by_px_each_side():
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True
    e = erode(m, 2)
    vs, us = np.nonzero(e)
    assert (vs.min(), vs.max(), us.min(), us.max()) == (7, 12, 7, 12)


def test_ray_to_plane_lands_on_the_height_asked():
    T = _camera()
    p = ray_to_plane(640.0, 360.0, INTR, T, 0.2)
    assert p[2] == pytest.approx(0.2)


def test_localise_refuses_an_unknown_method():
    cfg = {"camera": {"max_depth": 2.0, "depth_patch_radius": 2},
           "architecture_a": {"perception": {"localiser": "magic"}}}
    with pytest.raises(ValueError, match="mask_ray"):
        localise([], None, np.eye(4), cfg)

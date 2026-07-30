""" check_camera.py

This file renders RGB-D from the workspace camera and verify against MuJoCo's truth.

    uv run python scripts/check_camera.py --camera workspace --verify

Three things are checked here:
  1. the depth buffer is metric (metres, not normalised);
  2. it is z-depth, not ray length (they differ off-axis, and the pinhole
     back-projection assumes z-depth);
  3. the pinhole model actually predicts where objects land in the image, which
     is exactly the projection stage 3 will invert.

This is only for Sim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fyp.hardware.sim.renderer import render_rgbd
from fyp.hardware.sim.scene import (BLOCKS, body_position, camera_pose,
                                    load_scene, world_to_camera)
from fyp.helpers.pixel_to_3d import camera_to_pixel

TOL = 3e-3 # this is the tolerance, how much it can get wrong before we say it doesn't work


def colourise(depth: np.ndarray) -> np.ndarray:
    """
    This function takes the depth map (as an ndarray), and outputs as a greyscale image.
    It normalizes the map such that each pixel contains a value from 0 to 255 (png image)
    """
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    lo, hi = float(finite.min()), float(finite.max())
    norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
    return (255 * (1.0 - np.clip(norm, 0, 1))).astype(np.uint8)


def patch_median(depth: np.ndarray, u: float, v: float, r: int = 2) -> float:
    """
    This function takes a depth map and a pixel, and gives the medain depth of the small patch/square
    of pixels around it. We use median instead of a single pixel because a single reading on an
    object's corner will land between the object and the table.

    This is not helpers/pixel_to_3d.depth_at. That throws away NaN, -ve and infinite values which
    is correct for live inference. This is for verification so we wanna compute everything.

    u - column
    v - row
    r - how many pixels per side that forms the patch of pixels for computation of median
    """
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return float("nan")
    return float(np.median(depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]))


def verify(depth: np.ndarray, intr, model, data, camera: str) -> bool:
    """
    This takes the rendered depth map and gives a
    True if the camera is verified and working. False if it isn't.
    """
    cam_pos, cam_mat = camera_pose(model, data, camera)

    h, w = depth.shape
    ok = True

    def compare(label: str, got: float, want: float, tol: float = TOL) -> None:
        """
        This is a Helper Function that does the actual checks, by acting as a
        comparator. It compares 'got' and 'want' with a tolerance as a threshold for whether
        the test has passed or not.
        """
        nonlocal ok # this gives fn check the rights to write (edit/change) fn verify's 'ok' value
        good = np.isfinite(got) and abs(got - want) <= tol
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {label:<38} "
              f"{got:.4f} m  (want {want:.4f}, err {1000 * (got - want):+.1f} mm)")

    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    print(f"\n{intr}")
    print(f"camera  pos={np.round(cam_pos, 4).tolist()}  fovy={model.cam_fovy[cam_id]:.1f}")
    print(f"depth   {depth.dtype}  range {depth.min():.4f}..{depth.max():.4f} m")

    # CHECK 1 - Metric Scale to confirm that the depth buffer is in metres
    print("\nmetric scale")
    table_z = 0.0
    compare("bare table @ image centre", patch_median(depth, intr.cx, intr.cy),
          float(cam_pos[2]) - table_z)

    # CHECK 2 - Ensures that the block's top faces are at the correct distance
    print("\nblock top faces (pinhole projection -> depth probe)")
    for name in BLOCKS:
        centre = body_position(model, data, name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        if centre is None or gid < 0:
            print(f"  [SKIP] {name}: not in model")
            continue
        half_z = float(model.geom_size[gid][2])
        top = centre + np.array([0.0, 0.0, half_z])
        u, v, want = camera_to_pixel(world_to_camera(top, cam_pos, cam_mat), intr)
        in_frame = 0 <= u < w and 0 <= v < h
        compare(f"{name} @ (u={u:6.1f}, v={v:6.1f})" + ("" if in_frame else " OUT OF FRAME"),
              patch_median(depth, u, v, r=1), float(want))

    # CHECK 3 - Ensure the bin is in the image
    print("\nframing")
    bin_pos = body_position(model, data, "bin")
    if bin_pos is not None:
        u, v, _ = camera_to_pixel(world_to_camera(bin_pos, cam_pos, cam_mat), intr)
        inside = 0 <= u < w and 0 <= v < h
        ok &= bool(inside)
        print(f"  [{'PASS' if inside else 'FAIL'}] bin centre at (u={u:.1f}, v={v:.1f}) "
              f"in a {w}x{h} image")

    # CHECK 4 - To find the pixel of the furthest end of the table from the principal point
    print("\nconvention")
    z_depth = float(cam_pos[2])
    vv, uu = np.mgrid[0:h, 0:w]
    flat = np.isfinite(depth) & (np.abs(depth - z_depth) < 1e-3)
    if not flat.any():
        print("  [FAIL] no bare-table pixels found at the expected height")
        return False
    radius2 = (uu - intr.cx) ** 2 + (vv - intr.cy) ** 2
    idx = np.argmax(np.where(flat, radius2, -1.0))
    v, u = int(vv.flat[idx]), int(uu.flat[idx])
    got = float(depth[v, u])
    ray_len = z_depth * np.sqrt(1.0 + ((u - intr.cx) / intr.fx) ** 2
                                + ((v - intr.cy) / intr.fy) ** 2)
    if abs(ray_len - z_depth) < 5e-3:
        print(f"  [WARN] furthest table pixel is only {np.sqrt(radius2.flat[idx]):.0f} px "
              f"off-axis; the two conventions differ by <5 mm here - inconclusive")
    print(f"  flat table at (u={u:.0f}, v={v:.0f}): measured {got:.4f} m")
    print(f"    z-depth would be {z_depth:.4f} m")
    print(f"    ray length would be {ray_len:.4f} m")
    is_z = abs(got - z_depth) < abs(got - ray_len)
    ok &= is_z
    print(f"  [{'PASS' if is_z else 'FAIL'}] buffer is "
          f"{'z-depth (pinhole model valid as written)' if is_z else 'RAY LENGTH - back-projection must divide it out'}")

    return bool(ok)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="workspace")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out-dir", default="data/frames")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    from PIL import Image

    model, data = load_scene()
    rgb, depth, intr = render_rgbd(args.width, args.height, camera=args.camera,
                                   model=model, data=data)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / f"{args.camera}_rgb.png")
    Image.fromarray(colourise(depth)).save(out / f"{args.camera}_depth.png")
    np.save(out / f"{args.camera}_depth.npy", depth)
    print(f"saved {args.camera}_rgb.png / _depth.png / _depth.npy -> {out}")

    if args.verify:
        ok = verify(depth, intr, model, data, args.camera)
        print("\n" + ("ALL CHECKS PASS" if ok else "CHECKS FAILED"))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

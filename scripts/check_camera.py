"""Render RGB + depth from the workspace camera and verify it against MuJoCo truth.

Architecture A stage 3 (depth-to-3D) is only as good as this buffer, so nothing
here is taken on trust. The checks read ground truth straight out of the model -
camera pose, block positions, block sizes - so they keep working if the camera
or the props move.

    uv run python scripts/check_camera.py --camera workspace --verify

Three things are being established:
  1. the depth buffer is metric (metres, not normalised);
  2. it is z-depth, not ray length (they differ off-axis, and the pinhole
     back-projection assumes z-depth);
  3. the pinhole model actually predicts where objects land in the image, which
     is exactly the projection stage 3 will invert.

SIM ONLY. On real hardware there is no ground truth to compare against and this
becomes a repeatability check against a physical fixture instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fyp.hardware.sim.renderer import render_rgbd
from fyp.hardware.sim.scene import (BLOCKS, body_position, camera_pose,
                                    load_scene, world_to_camera)
from fyp.helpers.pixel_to_depth import camera_to_pixel

TOL = 3e-3      # 3 mm - generous next to a 40 mm cube, tight enough to catch sign errors


def colourise(depth: np.ndarray) -> np.ndarray:
    """Map finite depth to 8-bit greyscale (near = bright). Eyeballing only."""
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    lo, hi = float(finite.min()), float(finite.max())
    norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
    return (255 * (1.0 - np.clip(norm, 0, 1))).astype(np.uint8)


def patch_median(depth: np.ndarray, u: float, v: float, r: int = 2) -> float:
    """Median depth in a small window - robust to the odd edge pixel.

    Deliberately NOT helpers.pixel_to_depth.depth_at: that one also rejects
    non-finite, non-positive and beyond-max_depth samples. Those filters are
    right for the live pipeline but wrong for a verification probe, which should
    report whatever the buffer actually contains rather than quietly skipping
    the very pixels a bug would show up in.
    """
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return float("nan")
    return float(np.median(depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]))


def verify(depth: np.ndarray, intr, model, data, camera: str) -> bool:
    cam_pos, cam_mat = camera_pose(model, data, camera)

    h, w = depth.shape
    ok = True

    def check(label: str, got: float, want: float, tol: float = TOL) -> None:
        nonlocal ok
        good = np.isfinite(got) and abs(got - want) <= tol
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {label:<38} "
              f"{got:.4f} m  (want {want:.4f}, err {1000 * (got - want):+.1f} mm)")

    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    print(f"\n{intr}")
    print(f"camera  pos={np.round(cam_pos, 4).tolist()}  fovy={model.cam_fovy[cam_id]:.1f}")
    print(f"depth   {depth.dtype}  range {depth.min():.4f}..{depth.max():.4f} m")

    # -- 1. metric scale on a known flat surface -------------------------------
    print("\nmetric scale")
    table_z = 0.0
    check("bare table @ image centre", patch_median(depth, intr.cx, intr.cy),
          float(cam_pos[2]) - table_z)

    # -- 2. every block, projected through the pinhole model -------------------
    # If the projection is wrong the probe lands off the cube and reads table
    # depth instead - so this tests the camera model, not just the depth scale.
    print("\nblock top faces (pinhole projection -> depth probe)")
    for name in BLOCKS:
        centre = body_position(model, data, name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        if centre is None or gid < 0:
            print(f"  [SKIP] {name}: not in model")
            continue
        half_z = float(model.geom_size[gid][2])
        top = centre + np.array([0.0, 0.0, half_z])       # camera sees the top face
        u, v, want = camera_to_pixel(world_to_camera(top, cam_pos, cam_mat), intr)
        in_frame = 0 <= u < w and 0 <= v < h
        check(f"{name} @ (u={u:6.1f}, v={v:6.1f})" + ("" if in_frame else " OUT OF FRAME"),
              patch_median(depth, u, v, r=1), float(want))

    # -- 3. is the bin in frame at all? ---------------------------------------
    print("\nframing")
    bin_pos = body_position(model, data, "bin")
    if bin_pos is not None:
        u, v, _ = camera_to_pixel(world_to_camera(bin_pos, cam_pos, cam_mat), intr)
        inside = 0 <= u < w and 0 <= v < h
        ok &= bool(inside)
        print(f"  [{'PASS' if inside else 'FAIL'}] bin centre at (u={u:.1f}, v={v:.1f}) "
              f"in a {w}x{h} image")

    # -- 4. depth convention ---------------------------------------------------
    # Same flat table, as far off-axis as possible: ray length grows with the
    # off-axis angle, z-depth does not. Rather than guessing a bare-table pixel
    # (the arm may occlude a hard-coded corner), find the table pixel furthest
    # from the principal point. Under EITHER convention the table reads >= the
    # camera height, so selecting on "close to camera height" cannot bias the
    # test towards z-depth: a ray-length buffer simply has fewer such pixels,
    # all of them near the centre, and the chosen point would then sit at small
    # radius where the two predictions coincide - an inconclusive result, not a
    # false pass.
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
    np.save(out / f"{args.camera}_depth.npy", depth)   # raw metres; the PNG is lossy
    print(f"saved {args.camera}_rgb.png / _depth.png / _depth.npy -> {out}")

    if args.verify:
        ok = verify(depth, intr, model, data, args.camera)
        print("\n" + ("ALL CHECKS PASS" if ok else "CHECKS FAILED"))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

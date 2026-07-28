"""Architecture A stage 3: detections + depth -> camera-frame 3D points.

Two modes.

  --verify        Self-test with no detector in the loop. Projects each block's
                  known position to a pixel, reads depth there, back-projects,
                  and compares to ground truth. Isolates the geometry: if this
                  fails the camera model is wrong, not the detector.

  --detections    The real thing. Reads check_detector.py's JSON, back-projects
                  each box centre, and (in sim) reports the error against the
                  nearest true block. This is the number that matters.

    uv run python scripts/check_depth.py --verify
    uv run python scripts/check_depth.py --detections data/frames/detections.json

Output is in the CAMERA frame. Stage 4 (hand-eye) converts to the robot base
frame, which is what the primitives actually consume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fyp.hardware.sim.renderer import render_rgbd
from fyp.hardware.sim.scene import (block_truth, camera_pose, load_scene,
                                    world_to_camera)
from fyp.helpers.pixel_to_depth import (camera_to_pixel, depth_at,
                                        pixel_to_camera, surface_to_centroid)
from fyp.policy.modular.localiser import find_duplicates, locate, nearest_truth

TOL_MM = 5.0        # sim has no sensor noise; anything above this is a real bug


def run_verify(depth, intr, model, data, camera: str, far: float) -> bool:
    print("\n" + "=" * 72)
    print("A. round trip (pure maths, no rendering)")
    rng = np.random.default_rng(0)
    uv = rng.uniform([0, 0], [intr.width, intr.height], size=(2000, 2))
    d = rng.uniform(0.3, 2.0, size=2000)
    p = pixel_to_camera(uv[:, 0], uv[:, 1], d, intr)
    u2, v2, d2 = camera_to_pixel(p, intr)
    err = max(np.abs(u2 - uv[:, 0]).max(), np.abs(v2 - uv[:, 1]).max(), np.abs(d2 - d).max())
    ok = err < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] 2000 random pixels deproject/reproject, "
          f"max error {err:.2e}")

    print("\nB. back-projection vs MuJoCo ground truth (camera frame, mm)")
    cam_pos, cam_mat = camera_pose(model, data, camera)
    truth = block_truth(model, data)
    for name, t in truth.items():
        want = world_to_camera(t["top"], cam_pos, cam_mat)
        u, v, _ = camera_to_pixel(want, intr)
        z = depth_at(depth, u, v, radius=1, max_depth=far)
        got = pixel_to_camera(u, v, z, intr)
        e = np.abs(got - want) * 1000.0
        good = e.max() <= TOL_MM
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<13} "
              f"got ({got[0]:+.4f}, {got[1]:+.4f}, {got[2]:+.4f})  "
              f"err ({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}) mm")

    print("\nC. top surface -> centroid offset")
    for name, t in truth.items():
        want = world_to_camera(t["centroid"], cam_pos, cam_mat)
        u, v, _ = camera_to_pixel(world_to_camera(t["top"], cam_pos, cam_mat), intr)
        z = depth_at(depth, u, v, radius=1, max_depth=far)
        got = surface_to_centroid(pixel_to_camera(u, v, z, intr), t["half_z"])
        e = np.abs(got - want) * 1000.0
        good = e.max() <= TOL_MM
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<13} centroid err "
              f"({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}) mm")
    return bool(ok)


def run_detections(path: str, depth, intr, model, data, camera: str, far: float) -> bool:
    p = Path(path)
    if not p.exists():
        # Almost always means the detector hasn't been run yet, and it lives in a
        # different venv - so say exactly how to produce the file.
        print(f"\nno detections file at {p}\n"
              "Stage 2 has to run first, and it needs the lerobot venv (torch):\n\n"
              "  # 1. render the frame (FYP venv)\n"
              "  uv run python scripts/check_detector.py --render-only \\\n"
              "      --camera workspace --out data/frames/frame_top.png\n\n"
              "  # 2. detect (lerobot venv)\n"
              "  cd ~/lerobot && uv run python "
              "/home/jj/Documents/NTU/Y4S1/FYP/scripts/check_detector.py \\\n"
              "      --image /home/jj/Documents/NTU/Y4S1/FYP/data/frames/frame_top.png \\\n"
              "      --queries \"red cube\" \"green cube\" \"blue cube\" \"yellow cube\" \"bin\" \\\n"
              "      --model google/owlv2-base-patch16-ensemble --threshold 0.3 \\\n"
              "      --nms-iou 0.5 --top1-per-query \\\n"
              "      --out  /home/jj/Documents/NTU/Y4S1/FYP/data/frames/detections_owlv2.png \\\n"
              "      --json /home/jj/Documents/NTU/Y4S1/FYP/data/frames/detections.json\n\n"
              "Meanwhile `--verify` needs no detector and tests the geometry on its own.")
        return False

    try:
        dets = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"\n{p} is not valid JSON ({e}). Re-run check_detector.py --json.")
        return False
    if not dets:
        print(f"\n{p} is empty - the detector kept nothing. Lower --threshold, "
              "or check the overlay PNG to see what it saw.")
        return False

    cam_pos, cam_mat = camera_pose(model, data, camera)
    truth = block_truth(model, data)
    truth_cam = {n: world_to_camera(t["top"], cam_pos, cam_mat) for n, t in truth.items()}

    located = locate(dets, depth, intr, max_depth=far, radius=2)

    print(f"\n{len(dets)} detection(s) from {path}\n")
    hdr = f"{'query':<14}{'score':>6}  {'u':>6} {'v':>6} {'depth':>7}   " \
          f"{'camera-frame XYZ (m)':<30} nearest truth"
    print(hdr)
    print("-" * len(hdr))

    ok = True
    for obj in located:
        u, v = obj.center_uv
        if not obj.valid:
            print(f"{obj.query:<14}{obj.score:>6.2f}  {u:>6.1f} {v:>6.1f}  "
                  f"{'INVALID':>7}   <- no usable depth (occluded or background)")
            ok = False
            continue
        pt = obj.p_cam
        near, dist = nearest_truth(pt, truth_cam)
        tag = f"{near} {1000 * dist:.1f} mm" if near else "-"
        flag = "" if dist * 1000 <= 20 else "   <-- far from any block"
        print(f"{obj.query:<14}{obj.score:>6.2f}  {u:>6.1f} {v:>6.1f} {obj.depth_m:>7.4f}   "
              f"({pt[0]:+.4f}, {pt[1]:+.4f}, {pt[2]:+.4f})   {tag}{flag}")

    # A duplicate that survived NMS shows up as two detections on one 3D point.
    for i, j, sep in find_duplicates(located):
        print(f"\n  [WARN] '{located[i].query}' and '{located[j].query}' are "
              f"only {sep:.1f} mm apart - likely the same object; tighten NMS")
        ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="workspace")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--detections", default=None, help="JSON from check_detector.py --json")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    model, data = load_scene()
    _rgb, depth, intr = render_rgbd(args.width, args.height, camera=args.camera,
                                    model=model, data=data)
    # Anything at or beyond the floor is background, not an object on the table.
    far = float(depth.max()) - 1e-3
    print(f"{intr}\ndepth range {depth.min():.4f}..{depth.max():.4f} m  "
          f"(background rejected at >= {far:.4f})")

    ok = True
    if args.verify:
        ok &= run_verify(depth, intr, model, data, args.camera, far)
    if args.detections:
        ok &= run_detections(args.detections, depth, intr, model, data, args.camera, far)
    if not (args.verify or args.detections):
        p.error("give --verify and/or --detections")

    print("\n" + ("ALL CHECKS PASS" if ok else "CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

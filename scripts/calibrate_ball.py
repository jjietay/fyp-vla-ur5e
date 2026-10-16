"""calibrate_ball.py

It takes a ball held in the gripper and gives you a FITTED
`config/calibration/T_base_cam.json`, with the error of the fit in millimetres.

    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/calibrate_ball.py
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/calibrate_ball.py --plan-only

Run from the FYP folder, with the LEROBOT venv's python, because finding the ball
needs OWLv2 and so torch, which the FYP venv does not have by design (13 Oct 2026:
`uv run` here failed at the first pose). Calling the venv's python directly
rather than `uv run` in ~/lerobot skips uv's sync, so it cannot remove the UR5e
adapter and drivers installed there by hand. `--plan-only` would run in either.

Run it at the start of every session, after the camera is back on its tape
marks. It replaces measuring the mount with a rule. No markers: the only target
is a plain ball, which the detector finds by name like any task object.

How it works. The arm carries the ball to a grid of positions at two or more
heights. At each one the camera finds the ball and the depth gives its centre in
the CAMERA frame, while RTDE gives the TCP and so the ball centre in the BASE
frame. That is the same physical point seen in both frames, many times over, and
`solve_rigid_transform` finds the one transform that maps one cloud onto the
other. Unlike a measured transform, the fit reports its own error.

Two things the fit cannot check, both decided ONCE, not per session:

  `ball_offset_tool_m` in config.yaml. Every sample uses the same tool
  orientation, so an error in where the ball sits in the fingers is the same
  shift at every point. The fit absorbs it and the residual stays small. Check it
  once with `check_calibration.py` after the first fit, and always place the ball
  the same way: bottom flush with the fingertips.

  `ball_radius_m`. Measure the ball. The depth lands on its near surface and is
  pushed back along the ray by this much to reach the centre.

The previous calibration is never overwritten. It is moved aside with a time
stamp, so every result in the logbook can still name the transform behind it.

THIS MOVES THE ARM through the middle of the table at `calibration.speed`. Clear
the table first and keep a hand on the e-stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan-only", action="store_true",
                   help="print the poses that would be visited, then stop. No arm, no camera")
    p.add_argument("--out", default=None,
                   help="output path; defaults to the calibration_dir in config.yaml")
    p.add_argument("--note", default="", help="free text kept in the file")
    args = p.parse_args()

    from fyp.architecture_a.calibration.hand_eye import calibration_grid
    from fyp.architecture_a.calibration.store import CalibrationStore
    from fyp.shared.hardware.safety import WorkspaceEnvelope
    from fyp.shared.helpers.config import get_config

    cfg = get_config()
    cc = cfg["architecture_a"]["calibration"]
    envelope = WorkspaceEnvelope(cfg)
    rotvec = cfg["architecture_a"]["skills"]["down_rotvec"]
    store = CalibrationStore(cfg, args.out)

    poses, rejected = calibration_grid(cc["grid_centre_xy"], float(cc["grid_half_extent_m"]),
                                       int(cc["grid_n"]), cc["z_levels"], rotvec, envelope)
    print(f"{len(poses)} poses inside the envelope, {len(rejected)} left out")
    for xyz in rejected:
        print(f"    left out {np.round(xyz, 3).tolist()}")
    if len(poses) < int(cc["min_points"]):
        print(f"fewer than calibration.min_points ({cc['min_points']}) poses fit inside the "
              "envelope. Move grid_centre_xy or shrink grid_half_extent_m.", file=sys.stderr)
        return 1
    if args.plan_only:
        for i, pose in enumerate(poses, start=1):
            print(f"  {i:2d}. {np.round(pose[:3], 3).tolist()}")
        return 0

    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.ur5e_controller import URController

    controller = URController()
    try:
        controller.gripper_start()
        if not _take_the_ball(controller, cfg):
            return 1
        with RealSenseCamera(cfg) as camera:
            pairs = _collect(controller, camera, poses, envelope, cfg)
        _give_back_the_ball(controller)
    finally:
        controller.close()

    if len(pairs) < int(cc["min_points"]):
        print(f"\nonly {len(pairs)} usable points, calibration.min_points is "
              f"{cc['min_points']}. Nothing written. Check the ball is visible and the "
              "query in calibration.ball_query finds it (check_detector.py).", file=sys.stderr)
        return 1
    return _fit_and_save(np.asarray([c for c, _ in pairs]),
                         np.asarray([b for _, b in pairs]), store, cfg, args.note)


def _take_the_ball(controller, cfg) -> bool:
    """
    It takes nothing and gives you the ball closed in the fingers, or False when
    the fingers closed on nothing.

    The same width test as the pick skill. Carrying on without the ball would
    spend the whole run detecting nothing.
    """
    min_w = float(cfg["architecture_a"]["skills"]["min_grasp_width_mm"])
    controller.gripper_toggle(1)
    input("\nPlace the ball between the fingers, bottom flush with the fingertips, "
          "and press Enter. The fingers will close.")
    width = float(controller.gripper_toggle(0))
    if width < min_w:
        print(f"the fingers closed to {width:.1f} mm, so no ball is held. Stopping.",
              file=sys.stderr)
        controller.gripper_toggle(1)
        return False
    diameter = 2000.0 * float(cfg["architecture_a"]["calibration"]["ball_radius_m"])
    print(f"holding {width:.1f} mm (ball_radius_m says {diameter:.1f} mm)")
    if abs(width - diameter) > 5.0:
        # Fingers stopping well short of or past the diameter means it is held
        # off-centre or the radius in config is wrong. Either one biases the fit.
        print("  that does not match the configured ball. Check ball_radius_m and the grip.")
    return True


def _collect(controller, camera, poses, envelope, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    It takes the grid of poses and gives you (ball centre in camera frame, ball
    centre in base frame) for every pose where the ball was found.

    A pose where the ball is not found, or has no depth, is skipped rather than
    guessed. A missing point costs a little precision; an invented one is bias.
    """
    from fyp.architecture_a.calibration.hand_eye import ball_centre_in_base
    from fyp.architecture_a.perception.scene import Perception
    from fyp.architecture_a.perception.pixel_to_3d import (depth_at, pixel_to_camera,
                                                           surface_to_centroid)

    cc = cfg["architecture_a"]["calibration"]
    a = cfg["architecture_a"]
    query = str(cc["ball_query"])
    radius = float(cc["ball_radius_m"])
    offset = cc["ball_offset_tool_m"]
    speed, acc = envelope.clamp_motion(float(cc["speed"]), float(a["skills"]["acc"]))
    # No transform yet, that is what is being fitted. Only detection is used, so
    # the detector settings are A's own without the identity matrix meaning anything.
    perception = Perception(cfg, np.eye(4), envelope)

    pairs = []
    for i, pose in enumerate(poses, start=1):
        print(f"\n  {i:2d}/{len(poses)} {np.round(pose[:3], 3).tolist()}", end="  ")
        if controller.move_to_pose(envelope.check_pose(pose, "calibration pose"),
                                   speed, acc) is False:
            print("controller refused the pose, skipped")
            continue
        time.sleep(float(cc["settle_s"]))

        # Averaged like A's own captures, so the fit and the runs see the same depth.
        frame = camera.capture(
            depth_frames=int(cfg["architecture_a"]["perception"]["depth_frames"]))
        kept = perception.detect(frame, [query], strict=False)
        if not kept:
            print("ball not detected, skipped")
            continue
        _q, score, (x0, y0, x1, y1) = kept[0]
        u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0

        z = depth_at(frame.depth, u, v, radius=int(cfg["camera"]["depth_patch_radius"]),
                     max_depth=float(cfg["camera"]["max_depth"]))
        if not np.isfinite(z):
            print("no depth on the ball, skipped")
            continue

        # The depth is on the near surface. A sphere's centre lies on the ray
        # through the middle of its image, exactly `radius` further along it.
        p_cam = surface_to_centroid(pixel_to_camera(u, v, z, frame.intrinsics), radius)
        tcp = controller.get_state()["tcp_pose"]
        p_base = ball_centre_in_base(tcp, offset)
        pairs.append((np.asarray(p_cam, dtype=float), p_base))
        print(f"score {score:.2f}, depth {z:.3f} m")
    return pairs


def _give_back_the_ball(controller) -> None:
    """It takes nothing and gives you the ball back in your hand, not on the table."""
    input("\nHold the ball and press Enter. The fingers will open.")
    controller.gripper_toggle(1)


def _fit_and_save(p_cam, p_base, store, cfg, note: str) -> int:
    """
    It takes the matched points and gives you the fitted transform written to
    the store, after showing the fit error and how far it moved from the last one.
    """
    from fyp.architecture_a.calibration.hand_eye import fit_with_outlier_rejection, report

    cc = cfg["architecture_a"]["calibration"]
    T, keep = fit_with_outlier_rejection(p_cam, p_base)
    r = report(T, p_cam[keep], p_base[keep])

    print(f"\n{'=' * 70}\n  FIT over {r['n_points']} points"
          f" ({int((~keep).sum())} dropped as outliers)\n")
    for i, e in enumerate(r["per_point_mm"], start=1):
        print(f"    {i:2d}.  {e:6.1f} mm")
    print(f"\n  RMS {r['rms_mm']:.1f} mm, worst {r['max_mm']:.1f} mm")

    if store.exists():
        old, _ = store.current()
        dt = float(np.linalg.norm(T[:3, 3] - old[:3, 3])) * 1000.0
        cos = (np.trace(old[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0
        dang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        print(f"  versus the current file: camera moved {dt:.0f} mm, turned {dang:.1f} deg")

    if r["rms_mm"] > float(cc["max_rms_mm"]):
        print(f"\n  RMS is above calibration.max_rms_mm ({cc['max_rms_mm']}). Usual causes: "
              "the ball slipped in the fingers, or depth holes on the ball.")
        if input("  write it anyway? [y/N] ").strip().lower() != "y":
            print("  nothing written")
            return 1

    archived = store.replace(T, "ball_fit", {
        "method": "ball fit, scripts/calibrate_ball.py",
        "note": note,
        "n_points": r["n_points"],
        "dropped_outliers": int((~keep).sum()),
        "fit_rms_mm": r["rms_mm"],
        "fit_max_mm": r["max_mm"],
        "ball_query": cc["ball_query"],
        "ball_radius_m": float(cc["ball_radius_m"]),
        "ball_offset_tool_m": [float(v) for v in cc["ball_offset_tool_m"]],
    }, restart=True)
    if archived:
        print(f"  previous calibration kept as {archived.name}")
    print(f"\n  wrote {store.path}")
    print(json.dumps(np.round(T, 4).tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

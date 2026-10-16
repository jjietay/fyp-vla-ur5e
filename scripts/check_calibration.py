"""check_calibration.py

It takes the calibration you measured and gives you the one number that says
whether stage 4 works: the reach error, in millimetres.

    uv run python scripts/check_calibration.py --points 5

Not optional, and more important here than in a normal project. A FITTED
calibration reports its own residual and tells you when it is wrong. `Calibration
Marker` closed on no marker, so `T_base_cam` is hand measured and reports
nothing. This script is the ONLY evidence in the project that stage 4 works at
all, and without it every downstream failure gets blamed on calibration and none
of them can be ruled out.

What it does, per point:

  1. captures a frame and asks you to click an object in it
  2. back projects that pixel through the depth channel and `T_base_cam`, giving
     where the arm thinks the object is, in base frame
  3. checks that against the workspace envelope, then moves the TCP to hover
     directly above it
  4. asks you to measure the miss with a rule and type it in, in millimetres

The fourth step is manual on purpose. The whole question is whether the computed
position matches the real one, and anything the software measures is computed
from the same transform being tested, so it would agree with itself and prove
nothing.

Read the shape of the answer, not only its size. `Calibration Marker` predicts a
SYSTEMATIC offset of roughly 1 to 3 cm in one consistent direction, because
orientation error dominates and 1 degree displaces about 17 mm at a 1 m standoff.
So this reports the mean offset separately from the scatter:

  a large mean and small scatter means the six numbers are slightly wrong and can
  be corrected, either by re-measuring or by adjusting them against this result

  a large scatter means something else is broken, most likely depth alignment or
  the datum, and no amount of adjusting the numbers will fix it

THIS MOVES THE ARM towards points derived from a calibration that is not yet
trusted, which is the one situation where the envelope earns its keep. Keep a
hand on the estop.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=5,
                   help="how many points to test; W3 asks for at least 5, spread widely")
    p.add_argument("--calibration", default=None,
                   help="path to T_base_cam.json; defaults to the one in config.yaml")
    p.add_argument("--hover", type=float, default=None,
                   help="metres above the target to stop at; defaults to skills.approach_height")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and report base-frame targets without moving the arm")
    args = p.parse_args()

    from fyp.architecture_a.calibration.hand_eye import camera_to_base
    from fyp.architecture_a.calibration.store import CalibrationStore
    from fyp.architecture_a.perception.pixel_to_3d import depth_at, pixel_to_camera
    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
    from fyp.shared.helpers.config import get_config

    cfg = get_config()
    store = CalibrationStore(cfg, args.calibration)
    cal_path = store.path

    if not store.exists():
        print(f"no calibration at {cal_path}. Write one first:\n"
              f"    uv run python scripts/write_calibration.py --xyz X Y Z --rpy R P Y",
              file=sys.stderr)
        return 1

    T, _ = store.current()
    envelope = WorkspaceEnvelope(cfg)
    hover = args.hover if args.hover is not None else \
        float(cfg["architecture_a"]["skills"]["approach_height"])

    print(f"calibration: {cal_path}")
    print(f"envelope:    {envelope.describe()}")
    print(f"hover:       {hover * 1000:.0f} mm above each target\n")

    controller = None
    if not args.dry_run:
        from fyp.shared.hardware.ur5e_controller import URController
        controller = URController()

    misses: list[np.ndarray] = []
    try:
        with RealSenseCamera(cfg) as camera:
            for i in range(args.points):
                print(f"\n{'=' * 70}\n  point {i + 1} of {args.points}")
                miss = _one_point(camera, T, controller, envelope, hover, cfg,
                                  camera_to_base, pixel_to_camera, depth_at,
                                  SafetyViolation)
                if miss is not None:
                    misses.append(miss)
    finally:
        if controller is not None:
            controller.close()

    if not misses:
        print("\nno points measured, nothing to report")
        return 1

    return _report(np.asarray(misses), store)


def _one_point(camera, T, controller, envelope, hover, cfg,
               camera_to_base, pixel_to_camera, depth_at, SafetyViolation):
    """
    It takes one clicked pixel and gives you the miss vector in millimetres, or
    None when the point could not be used.
    """
    frame = camera.capture()
    uv = _click(frame.rgb)
    if uv is None:
        return None
    u, v = uv

    z = depth_at(frame.depth, u, v,
                 radius=int(cfg["camera"]["depth_patch_radius"]),
                 max_depth=float(cfg["camera"]["max_depth"]))
    if not np.isfinite(z) or z <= 0:
        # A hole in the depth image, which is common on dark, shiny or very thin
        # objects. Skipped rather than guessed, because a fabricated depth would
        # show up as calibration error and it is not one.
        print("  no valid depth at that pixel, skipping. Try a matte object.")
        return None

    p_cam = pixel_to_camera(u, v, z, frame.intrinsics)
    p_base = np.asarray(camera_to_base(p_cam, T), dtype=float).reshape(3)

    print(f"  pixel ({u:.0f}, {v:.0f}) at {z:.3f} m")
    print(f"  camera frame: {np.array2string(p_cam, precision=4)}")
    print(f"  base frame:   {np.array2string(p_base, precision=4)}")

    if controller is None:
        return None

    target = p_base.copy()
    target[2] += hover
    try:
        envelope.check_point(target, "calibration target")
    except SafetyViolation as e:
        print(f"  REFUSED: {e}")
        return None

    if input("  move the arm there? [y/N] ").strip().lower() != "y":
        return None

    pose = list(target) + list(cfg["architecture_a"]["skills"]["down_rotvec"])
    controller.move_to_pose(pose)

    print(f"  the TCP is now {hover * 1000:.0f} mm above where the arm THINKS the "
          f"object is.")
    print("  measure from the tool tip to the object, in millimetres, base frame axes.")
    return _read_miss()


def _click(rgb):
    """
    It takes an RGB frame and gives you the pixel you clicked, or None.

    matplotlib rather than OpenCV, because matplotlib is already a dependency and
    a second GUI toolkit for one click is not worth the install.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb)
    ax.set_title("click the object, or close the window to skip this point")
    ax.axis("off")
    picked = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not picked:
        print("  nothing clicked, skipping")
        return None
    return picked[0]


def _read_miss():
    """It takes three typed millimetre readings and gives you the miss vector."""
    while True:
        raw = input("  dx dy dz in mm (blank to skip): ").strip()
        if not raw:
            return None
        try:
            miss = np.array([float(v) for v in raw.split()], dtype=float)
            if miss.shape != (3,):
                raise ValueError
            return miss
        except ValueError:
            print("  expected three numbers, e.g.  12 -4 3")


def _report(misses: np.ndarray, store) -> int:
    """It takes the miss vectors and gives you the numbers that go in the report."""
    distances = np.linalg.norm(misses, axis=1)
    mean = misses.mean(axis=0)
    scatter = misses - mean

    print(f"\n{'=' * 70}\n  REACH ERROR over {len(misses)} points\n")
    for i, (m, d) in enumerate(zip(misses, distances), start=1):
        print(f"    {i}.  {np.array2string(m, precision=1):<28} |{d:6.1f}| mm")

    print(f"\n  RMS distance      {float(np.sqrt(np.mean(distances ** 2))):6.1f} mm")
    print(f"  worst point       {float(distances.max()):6.1f} mm")
    print(f"  mean offset       {np.array2string(mean, precision=1)} mm  "
          f"(|{float(np.linalg.norm(mean)):.1f}| mm)")
    print(f"  scatter about it  {float(np.sqrt(np.mean(np.sum(scatter ** 2, axis=1)))):6.1f} mm")

    # The diagnosis, not just the numbers. Which of the two it is decides whether
    # the next hour goes on re-measuring six numbers or on the depth pipeline.
    bias, spread = float(np.linalg.norm(mean)), float(np.sqrt(np.mean(np.sum(scatter ** 2, axis=1))))
    print()
    if bias > 2 * spread:
        print("  Mostly SYSTEMATIC, which is what a measured transform is expected to")
        print("  give. The six numbers are slightly wrong. Re-measure the mount, paying")
        print("  attention to orientation, since 1 degree is about 17 mm at 1 m.")
    elif spread > 2 * bias:
        print("  Mostly SCATTER, which a wrong transform does NOT produce. Suspect depth")
        print("  alignment or the optical origin datum before touching the six numbers.")
    else:
        print("  Bias and scatter are comparable, so more points would help before")
        print("  concluding anything.")

    store.mark_validated({
        "n_points": int(len(misses)),
        "rms_mm": float(np.sqrt(np.mean(distances ** 2))),
        "max_mm": float(distances.max()),
        "mean_offset_mm": [float(v) for v in mean],
        "scatter_mm": spread,
    })
    print(f"\n  recorded in {store.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

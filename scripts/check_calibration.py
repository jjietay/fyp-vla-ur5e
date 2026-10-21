"""check_calibration.py

It takes the calibration you measured and gives you the one number that says
whether stage 4 works: the reach error, in millimetres.

    uv run python scripts/check_calibration.py --points 5

Not optional. The ball fit reports its own residual, but a bias shared by every
fit pose is invisible to it (ADR 0007): on 21 Oct 2026 a fit at RMS 7.3 mm still
missed by 18 to 34 mm here. This script is the ONLY evidence in the project that
stage 4 works at all, and without it every downstream failure gets blamed on
calibration and none of them can be ruled out.

What it does, per point:

  1. captures a frame and asks you to click an object in it
  2. back projects that pixel through the depth channel and `T_base_cam`, giving
     where the arm thinks the object is, in base frame
  3. checks that against the workspace envelope, then moves the TCP to hover
     directly above it
  4. asks you to measure the miss with a rule and type it in, in millimetres,
     or with --jog, lets you jog the TCP from the keyboard until it sits over
     the object and records the miss from the arm's own position

The fourth step is manual on purpose. With --jog the reading comes from RTDE,
which is the arm's joint encoders through its own kinematics, and the decision
of where "centred" is comes from your eye, so neither one uses the transform
under test. Added 21 Oct 2026: a rule held 120 mm under the fingers could not
resolve a few millimetres, and 8 mm is the whole clearance beside the orange. The whole question is whether the computed
position matches the real one, and anything the software measures is computed
from the same transform being tested, so it would agree with itself and prove
nothing.

Read the shape of the answer, not only its size. So this reports the mean offset
separately from the scatter:

  a large mean and small scatter means a bias shared by the fit: the ball offset,
  or the fingers hiding part of the ball. Refit with a ball they do not hide, or
  correct against this result

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
    p.add_argument("--jog", action="store_true",
                   help="jog the TCP over the object from the keyboard instead of typing a "
                        "measured miss; the miss is read from RTDE. Use with --hover 0.02")
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
    # Where the arm started, which is where it is out of the camera's way. Each
    # point after the first is clicked in a fresh frame, and with the arm still
    # hovering over the last cross it hid the next ones (21 Oct 2026).
    start_pose = None if controller is None else list(controller.get_state()["tcp_pose"])
    if controller is not None:
        # Start from home, so that "back to the start pose" means back home, which
        # is in the envelope and at the edge of the camera's view.
        q_now = np.degrees(controller.get_state()["joint_pos"])
        q_home = np.degrees(cfg["robot"]["home_q"])
        if np.max(np.abs(q_now - q_home)) > 2.0:
            controller.close()
            print("the arm is not at home. Run  uv run python scripts/go_home.py  first, "
                  "so it can return there between points.", file=sys.stderr)
            return 1
    try:
        with RealSenseCamera(cfg) as camera:
            for i in range(args.points):
                if i > 0 and controller is not None:
                    _clear_view(controller, envelope, start_pose, SafetyViolation)
                print(f"\n{'=' * 70}\n  point {i + 1} of {args.points}")
                miss = _one_point(camera, T, controller, envelope, hover, cfg,
                                  camera_to_base, pixel_to_camera, depth_at,
                                  SafetyViolation, args.jog)
                if miss is not None:
                    misses.append(miss)
    finally:
        if controller is not None:
            controller.close()

    if not misses:
        print("\nno points measured, nothing to report")
        return 1

    if len(misses) < _MIN_POINTS:
        _report(np.asarray(misses), None)
        print(f"\n  only {len(misses)} point(s); at least {_MIN_POINTS} are needed to call "
              f"the calibration validated, so nothing was recorded.")
        return 1
    return _report(np.asarray(misses), store)


# One or two points cannot separate a constant offset from scatter, and on
# 21 Oct 2026 a single mis-jogged point was written as the validation result.
_MIN_POINTS = 3


def _one_point(camera, T, controller, envelope, hover, cfg,
               camera_to_base, pixel_to_camera, depth_at, SafetyViolation, jog=False):
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

    if _ask("  move the arm there? [y/N] ").strip().lower() != "y":
        return None

    pose = list(target) + list(cfg["architecture_a"]["skills"]["down_rotvec"])
    controller.move_to_pose(pose)

    print(f"  the TCP is now {hover * 1000:.0f} mm above where the arm THINKS the "
          f"object is.")
    if jog:
        return _jog_miss(controller, envelope, pose, SafetyViolation)
    print("  measure from the tool tip to the object, in millimetres, base frame axes.")
    return _read_miss()


# Jog keys, in the robot's own terms rather than x and y, because nobody at the
# bench can tell which way +y is by looking (21 Oct 2026). Out/in and left/right
# are taken at the point's own bearing: base x is only "out" straight ahead, and
# at -68 deg an x jog moved the gripper mostly sideways.
_JOG_KEYS = {"f": ("radial", +1), "b": ("radial", -1),   # out from the base, back towards it
             "l": ("around", +1), "r": ("around", -1),   # the robot's left (anticlockwise seen from above), its right
             "u": ("z", +1), "d": ("z", -1)}
_JOG_SPEED = 0.02        # m/s. Steps are millimetres, so slow costs nothing
_JOG_MAX_M = 0.05        # how far from the commanded point a correction may go


def _jog_miss(controller, envelope, start_pose, SafetyViolation):
    """
    It takes the pose the arm was sent to and gives you the miss vector in
    millimetres, read from where you jogged the TCP to, or None to skip.

    The miss is final minus commanded, which is the same sign as the typed
    reading (from the tool tip to the object): the object sits where you
    stopped. Height is only meaningful if you jogged z; otherwise dz is 0.

    Every step goes through the envelope like any other pose, and a correction
    is capped at `_JOG_MAX_M`. A miss larger than that is not a calibration
    residual, it is a broken calibration, and jogging further would hide it.
    """
    step_mm = 2.0
    start = np.asarray(start_pose[:3], dtype=float)
    pose = list(start_pose)
    print("  jog until the point midway between the fingertips is over the object.")
    print("    f/b  straight out from the base / back towards it")
    print("    l/r  round to the robot's left / right, at the same distance")
    print("    u/d  up / down      repeat a letter for more steps, e.g. 'lll'")
    print("    s<mm> sets the step, e.g. s1 or s0.5 (now 2 mm)")
    print("    Enter on an empty line records the point;  q skips it")
    while True:
        raw = _ask(f"  jog [{step_mm:g} mm] > ").strip().lower()
        if raw == "":
            break
        if raw == "q":
            return None
        if raw.startswith("s"):
            try:
                asked = float(raw[1:])
                if not 0 < asked <= 10:
                    raise ValueError
                step_mm = asked
            except ValueError:
                print("  step must be between 0 and 10 mm, e.g. s1")
            continue
        if any(c not in _JOG_KEYS for c in raw):
            print("  use only f b l r u d, s<mm>, q, or Enter")
            continue
        new = list(pose)
        out, left = _radial_axes(start)
        for c in raw:
            kind, sign = _JOG_KEYS[c]
            d = sign * step_mm / 1000.0
            if kind == "radial":
                new[0] += d * out[0]; new[1] += d * out[1]
            elif kind == "around":
                new[0] += d * left[0]; new[1] += d * left[1]
            else:
                new[2] += d
        offset = np.asarray(new[:3]) - start
        if np.linalg.norm(offset) > _JOG_MAX_M:
            print(f"  REFUSED: that is {np.linalg.norm(offset) * 1000:.0f} mm from the "
                  f"commanded point, over {_JOG_MAX_M * 1000:.0f} mm. The calibration is "
                  f"badly off; record or skip this point.")
            continue
        try:
            envelope.check_pose(new, "jog step")
        except SafetyViolation as e:
            print(f"  REFUSED: {e}")
            continue
        controller.move_to_pose(new, speed=_JOG_SPEED)
        pose = new
        print(f"  offset so far  {np.array2string(offset * 1000, precision=1)} mm")

    # From RTDE, not from `pose`: where the arm IS, not where it was told to go.
    actual = np.asarray(controller.get_state()["tcp_pose"][:3], dtype=float)
    miss = (actual - start) * 1000.0
    out, left = _radial_axes(start)
    print(f"  recorded miss  {np.array2string(miss, precision=1)} mm base frame;  "
          f"out {miss[:2] @ out:+.1f}, left {miss[:2] @ left:+.1f}, up {miss[2]:+.1f} mm")
    return miss


def _radial_axes(xyz):
    """It takes a base-frame point and gives you the horizontal unit vectors pointing
    straight out from the base axis through it, and 90 degrees to the robot's left."""
    out = np.asarray(xyz[:2], dtype=float)
    out = out / np.linalg.norm(out)
    return out, np.array([-out[1], out[0]])


def _clear_view(controller, envelope, start_pose, SafetyViolation):
    """
    It takes the pose the run started at and gives you the arm back there, out of
    the camera's view, lifting straight up first so it never drags across the
    table. Both moves go through the envelope; if either is refused the arm stays
    where it is and you are told why.

    Cartesian, not a joint move back: the start pose has the same tool
    orientation as every hover pose, so moveL cannot spin the wrist, and a joint
    move from the far side of the sector can exceed max_joint_step_deg.
    """
    now = list(controller.get_state()["tcp_pose"])
    if np.linalg.norm(np.subtract(now[:3], start_pose[:3])) < 0.005:
        return
    lift = list(now)
    lift[2] = max(now[2], min(start_pose[2], envelope.z[1]))
    try:
        envelope.check_pose(lift, "lift before the next point")
        envelope.check_pose(start_pose, "start pose")
    except SafetyViolation as e:
        print(f"  could not clear the view: {e}. Jog the arm out of the frame, then go on.")
        return
    print("  moving back to the start pose so the camera can see the table ...")
    controller.move_to_pose(lift, speed=0.05)
    controller.move_to_pose(start_pose, speed=0.05)


def _ask(prompt: str) -> str:
    """
    It takes a prompt and gives you the line typed AFTER it appeared.

    Anything typed while the arm was moving is thrown away first. The moves
    block, and on 21 Oct 2026 jog keys typed during the move back home were
    read as the answers to the next two "move the arm there?" prompts, which
    silently skipped both points.
    """
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass   # not a terminal (input piped in, or no termios); nothing to flush
    return input(prompt)


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
        raw = _ask("  dx dy dz in mm (blank to skip): ").strip()
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
        print("  Mostly SYSTEMATIC: a bias every fit pose shared (ADR 0007), such as the")
        print("  ball offset or the fingers hiding part of the ball. Refit with a ball they")
        print("  do not hide, or correct the transform against these points.")
    elif spread > 2 * bias:
        print("  Mostly SCATTER, which a wrong transform does NOT produce. Suspect depth")
        print("  alignment or the optical origin datum before touching the six numbers.")
    else:
        print("  Bias and scatter are comparable, so more points would help before")
        print("  concluding anything.")

    if store is None:
        return 0
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

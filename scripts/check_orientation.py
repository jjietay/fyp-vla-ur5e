"""check_orientation.py

It takes a stationary arm and gives you whether its orientation numbers are
stable, which decides how the dataset should store rotation.

    uv run python scripts/check_orientation.py --seconds 10

READ ONLY. It opens the RTDE receive interface and nothing else, so it cannot
command the arm and cannot interrupt a program running on the teach pendant.
Put the arm in the pose you will pick from, leave it alone, and run this.

## Why this matters more than it sounds

Every rotation format has a boundary where two different sets of numbers describe
the same physical orientation, and near that boundary sensor noise flips between
them. The arm has not moved and the numbers jump.

Your gripper-pointing-down pose sits on the boundary of BOTH formats in use:

  the rotation vector `down_rotvec` is [3.142, 0, 0], whose magnitude is pi, and
  a turn of pi about an axis equals a turn of pi about the opposite axis, so
  [3.142, 0, 0] and [-3.142, 0, 0] are the same pose

  converting that to Euler gives roll of pi, which is the wrap point, so roll can
  read +3.14 on one frame and -3.14 on the next

`observation.state` carries roll, so if this flips, the policy sees one of its
seven inputs jump by 6.28 at random while the arm is perfectly still, on the
exact pose used for every pick. Action deltas are unaffected, because they are
computed through rotation matrices.

## Reading the result

A stationary arm should show ranges of roughly sensor noise, which is small
fractions of a millimetre and of a milliradian. Any component whose range is
near 2*pi is flipping, not moving.

If it flips, store rotation as a continuous six-number form, being the first two
columns of the rotation matrix, instead of Euler. If it does not flip on your
hardware, note that in the report and carry on with Euler, because it is a
property of this controller's firmware rather than a law.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

FLIP_THRESHOLD_RAD = 1.0   # a stationary arm moving a radian is not moving


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=10.0,
                   help="how long to watch the stationary arm")
    p.add_argument("--host", default=None, help="override robot.host from config")
    args = p.parse_args()

    from fyp.architecture_b.demos.lerobot_export import STATE_NAMES, build_state
    from fyp.shared.helpers.config import get_config

    cfg = get_config()
    host = args.host or cfg["robot"]["host"]
    period = float(cfg["robot"]["servo"]["time"])

    try:
        import rtde_receive
    except ImportError:
        print("ur_rtde is not installed. Run: uv sync", file=sys.stderr)
        return 1

    print(f"connecting to {host}, receive only")
    try:
        # Receive only, deliberately. RTDEControlInterface would take control of
        # the arm, and this script has no business doing that.
        rtde_r = rtde_receive.RTDEReceiveInterface(host)
    except Exception as e:
        print(f"could not reach {host}: {e}\n"
              f"  check robot.host in config.yaml is the arm's IP", file=sys.stderr)
        return 1

    pin = int(cfg["robot"]["gripper"]["control_pin"])
    print(f"watching for {args.seconds:.0f}s. DO NOT MOVE THE ARM.\n")

    poses, states = [], []
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        pose = np.asarray(rtde_r.getActualTCPPose(), dtype=float)
        gripper = 0 if rtde_r.getDigitalOutState(pin) else 1
        poses.append(pose)
        states.append(build_state(pose, gripper))
        time.sleep(max(0.0, period - (time.monotonic() - t0)))

    return _report(np.asarray(poses), np.asarray(states), STATE_NAMES)


def _report(poses: np.ndarray, states: np.ndarray, state_names) -> int:
    """It takes the logged frames and gives you the verdict."""
    print(f"{len(poses)} frames captured\n")

    rotvec = poses[:, 3:]
    magnitude = np.linalg.norm(rotvec, axis=1)
    print(f"rotation vector magnitude {magnitude.min():.4f} to {magnitude.max():.4f} rad")
    print(f"  pi is {np.pi:.4f}. The closer to pi, the closer to the boundary.")
    if abs(float(np.median(magnitude)) - np.pi) < 0.05:
        print("  THIS POSE IS ON THE AXIS-ANGLE BOUNDARY.")

    print("\nper component range over a stationary arm:")
    flipped = []

    for i, name in enumerate(("rx", "ry", "rz")):
        col = rotvec[:, i]
        spread = float(col.max() - col.min())
        flag = "  <-- FLIPPING" if spread > FLIP_THRESHOLD_RAD else ""
        print(f"  tcp_pose.{name:<3} {col.min():+.4f} to {col.max():+.4f}  "
              f"range {spread:.4f}{flag}")
        if spread > FLIP_THRESHOLD_RAD:
            flipped.append(f"tcp_pose.{name}")

    for i, name in enumerate(state_names):
        col = states[:, i]
        spread = float(col.max() - col.min())
        flag = "  <-- FLIPPING" if spread > FLIP_THRESHOLD_RAD else ""
        print(f"  state.{name:<8} {col.min():+.4f} to {col.max():+.4f}  "
              f"range {spread:.4f}{flag}")
        if spread > FLIP_THRESHOLD_RAD:
            flipped.append(f"state.{name}")

    print()
    if flipped:
        print("VERDICT: unstable. These flip while the arm is stationary:")
        for name in flipped:
            print(f"  {name}")
        print("\nThe policy would see these jump at random on the pose used for every\n"
              "pick. Store rotation as the first two columns of the rotation matrix\n"
              "instead of Euler, and change STATE_NAMES, the converter and the\n"
              "adapter together, BEFORE recording any episodes.")
        return 1

    print("VERDICT: stable. Nothing flipped over this window.\n"
          "Euler is safe to record with on this controller. Worth repeating at the\n"
          "other tool orientations you plan to use, since this only proves the one\n"
          "pose you just held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

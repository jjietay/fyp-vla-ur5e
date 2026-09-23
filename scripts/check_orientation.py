"""check_orientation.py

It takes a stationary arm and gives you whether the state vector Architecture B
sees is stable, with the raw rotation vector shown beside it.

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

## What happened

On 23 Sep 2026 this script showed Euler roll flipping between -pi and +pi on the
lab arm at the vertical pose. `observation.state` therefore no longer carries
Euler angles. It carries the first two columns of the rotation matrix, which
change smoothly everywhere (see `build_state` in lerobot_export.py). Action
deltas were never affected, because they are computed through rotation matrices.

## Reading the result now

A stationary arm should show ranges of roughly sensor noise. Any component whose
range is near 2*pi is flipping, not moving.

The raw rotation vector (`tcp_pose.rx` to `rz`) may still flip at this pose. That
is expected and harmless, because nothing feeds it to the policy directly. The
verdict is decided by the `state.*` rows only, and with the 6D state they should
be stable. A flip there now means `build_state` is wrong.
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

    print(f"watching for {args.seconds:.0f}s. DO NOT MOVE THE ARM.\n")

    poses, states = [], []
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        pose = np.asarray(rtde_r.getActualTCPPose(), dtype=float)
        # The gripper value does not affect the flip check, and Python cannot
        # read the RG2 width anyway, so a fixed 1 stands in for it.
        gripper = 1
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
            print("    (raw rotation vector, not seen by the policy)")

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
              "pick. The state has been 6D since 23 Sep 2026 and should not flip, so\n"
              "check build_state in lerobot_export.py BEFORE recording any episodes.")
        return 1

    print("VERDICT: stable. Nothing flipped over this window.\n"
          "The 6D state is safe to record with. Worth repeating at the\n"
          "other tool orientations you plan to use, since this only proves the one\n"
          "pose you just held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

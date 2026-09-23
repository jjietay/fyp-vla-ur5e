"""go_home.py

It takes the arm wherever it is and gives you the arm at robot.home_q from
config.yaml, after it shows you the move and you agree to it.

    uv run python scripts/go_home.py
    uv run python scripts/go_home.py --speed 0.1

THIS MOVES THE ARM. Keep a hand on the emergency stop.

It is a joint move (moveJ), not a Cartesian move, for the same reason as
skills.home(): a straight-line TCP path from an unknown pose can pass through a
singularity, and a joint move cannot. Every joint turns at the same time and
all of them arrive together.

## Why it refuses large moves

A joint move is safe only when it is small. Each joint takes the direct route
between its current and target angle, so a large change on the shoulder or
elbow can swing the arm through the table. The script reads the real joint
angles from the controller, never from what you typed, and stops if any joint
must turn more than --max-deg. Jog the arm closer on the pendant first.

## Stopping it

The move runs asynchronously so that Ctrl+C reaches Python while the arm is
moving. Ctrl+C sends stopJ, which brakes the arm to rest. The pendant emergency
stop always works and is faster.

The pendant must be in Remote Control mode, because RTDEControlInterface cannot
command an e-Series arm in Local mode.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
from fyp.shared.helpers.config import get_config

ARRIVED_RAD = 0.005   # about 0.3 deg per joint counts as there
STOP_DECEL = 2.0      # rad/s^2 used by stopJ on Ctrl+C. Firm, not violent
POLL_S = 0.02


def main() -> int:
    cfg = get_config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=float, default=0.2, help="joint speed, rad/s")
    p.add_argument("--acc", type=float, default=0.3, help="joint acceleration, rad/s^2")
    p.add_argument("--max-deg", type=float, default=None,
                   help="refuse if any joint must turn more than this. "
                        "Default workspace.max_joint_step_deg")
    args = p.parse_args()

    home = cfg.get("robot", {}).get("home_q")
    if home is None or len(home) != 6:
        print("no robot.home_q with six values in config.yaml", file=sys.stderr)
        return 1
    home = np.asarray(home, dtype=float)
    host = cfg["robot"]["host"]
    # The same check skills.home() uses, so the script and the skill cannot
    # disagree about what counts as too large a joint move.
    env = WorkspaceEnvelope(cfg)
    if args.max_deg is not None:
        env.max_joint_step_deg = float(args.max_deg)
    args.max_deg = env.max_joint_step_deg

    # Receive first. It is read only, so the plan can be shown and refused
    # without ever opening the interface that can move the arm.
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface(host)
    now = np.asarray(r.getActualQ(), dtype=float)
    if np.allclose(now, 0.0):
        print("every joint reads exactly 0. The arm is powered off or the brakes are "
              "on. Power it on and release the brakes on the pendant.", file=sys.stderr)
        return 1

    delta = np.degrees(home - now)
    print(f"robot {host}\n")
    print(f"{'joint':<8}{'now deg':>10}{'home deg':>10}{'change':>10}")
    for j, (a, b, d) in enumerate(zip(np.degrees(now), np.degrees(home), delta)):
        flag = "  <-- too large" if abs(d) > args.max_deg else ""
        print(f"{'q' + str(j):<8}{a:>10.2f}{b:>10.2f}{d:>+10.2f}{flag}")

    worst = float(np.max(np.abs(delta)))
    try:
        env.check_joint_move(now, home, "go_home")
    except SafetyViolation as e:
        print(f"\nREFUSED. {e}, then run again.", file=sys.stderr)
        return 1
    if worst < np.degrees(ARRIVED_RAD):
        print("\nalready at home")
        return 0

    duration = np.radians(worst) / args.speed
    print(f"\nlargest change {worst:.2f} deg, about {duration:.1f} s at {args.speed} rad/s")
    if input("Clear the area, hand on the e-stop. Type y to move: ").strip().lower() != "y":
        print("not moved")
        return 0

    import rtde_control
    c = rtde_control.RTDEControlInterface(host)
    try:
        c.moveJ(home.tolist(), args.speed, args.acc, True)
        # Generous: the time at cruise speed, plus room for accelerating and settling.
        deadline = time.monotonic() + 3.0 * duration + 5.0
        while True:
            q = np.asarray(r.getActualQ(), dtype=float)
            qd = np.asarray(r.getActualQd(), dtype=float)
            if np.max(np.abs(q - home)) < ARRIVED_RAD and np.max(np.abs(qd)) < 0.01:
                break
            if time.monotonic() > deadline:
                c.stopJ(STOP_DECEL)
                print("did not arrive in time, stopped. Check the pendant for a "
                      "protective stop.", file=sys.stderr)
                return 1
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        c.stopJ(STOP_DECEL)
        print("\nstopped by Ctrl+C", file=sys.stderr)
        return 1
    finally:
        c.stopScript()

    err = np.degrees(np.max(np.abs(np.asarray(r.getActualQ()) - home)))
    print(f"at home, worst joint error {err:.3f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())

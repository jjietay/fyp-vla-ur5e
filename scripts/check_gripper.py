"""check_gripper.py

It takes a width, or a number of cycles, and gives you the OnRobot RG2 fingers
moved from this laptop, with what the controller measured after each move.

    uv run python scripts/check_gripper.py                 # open, close, open
    uv run python scripts/check_gripper.py --cycles 20     # endurance, counts failures
    uv run python scripts/check_gripper.py --width 50      # one move to 50 mm

THE ARM DOES NOT MOVE. Only the fingers do. The pendant must be in Remote
Control mode, because RTDEControlInterface cannot run on e-Series in Local mode.
Nothing may be between the fingers for the cycle test, because a close on an
object stops short and is reported as holding something.

## Same code as the skills

This script used to carry its own copy of the RG2 word and the URScript. It now
drives the gripper through `URController`, so a bench run tests exactly the code
that `skills.py` and Architecture B run: the width check before sending, the
resend when the fingers do not start, and the result read back from the RTDE
registers. How that works is written at the top of `ur5e_controller.py`.

`--width` uses `gripper_move`, which reports any width without judging it. The
cycles use `gripper_toggle`, which raises `GripperError` on a failure. Here the
error is caught, printed and counted, so one lost command does not end the run
and the endurance number is honest.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=float, default=None, help="one move to this width, mm")
    p.add_argument("--force", type=float, default=None,
                   help="grip force in N for --width. Default robot.gripper.force_n")
    p.add_argument("--cycles", type=int, default=1, help="close and open this many times")
    args = p.parse_args()

    from fyp.shared.hardware.ur5e_controller import GripperError, URController

    c = URController()
    try:
        c.gripper_start()
        print(f"connected. Before: width {c.gripper_width_mm:.1f} mm, "
              f"state {c.get_state()['gripper_state']} (1 open, 0 closed)")
        input("Keep hands away from the fingers. Press Enter to start.")

        if args.width is not None:
            t0 = time.monotonic()
            status, width = c.gripper_move(args.width, args.force)
            print(f"  target {args.width:5.1f} mm -> {status.name}, width {width:5.1f} mm, "
                  f"{time.monotonic() - t0:.2f} s")
            return 0

        # Start and end open, so the gripper is never left clamped on something.
        states = [1] + [0, 1] * args.cycles
        failures = 0
        for n, state in enumerate(states, 1):
            t0 = time.monotonic()
            try:
                width = c.gripper_toggle(state)
                result = f"width {width:5.1f} mm"
            except GripperError as e:
                failures += 1
                result = f"FAILED: {e}"
            print(f"  {n:3d} {'open ' if state else 'close'} -> {result}  "
                  f"{time.monotonic() - t0:.2f} s")
            time.sleep(0.3)  # so a person watching can tell one stroke from the next

        print(f"\n{len(states)} commands, {failures} failed")
        return 1 if failures else 0
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())

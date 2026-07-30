"""Does the arm actually reach the joint angles it was commanded?

move_joints sets data.ctrl and steps a fixed number of times. If the position
actuators can't close the gap in that window, the arm stops short and every
recorded demo contains poses the robot never truly reached.

This drives the arm to an IK solution and watches the error decay:

  * error keeps shrinking  -> just needs a longer settle window
  * error flattens out     -> steady-state actuator error; more steps won't help,
                              the fix is gains or gravity compensation

    uv run python scripts/check_settling.py
    uv run python scripts/check_settling.py --dx 0.10
"""
from __future__ import annotations

import argparse

import numpy as np
import mujoco

from fyp.hardware.sim.mujoco_controller import URControllerMuJoCo
from fyp.helpers.ik import solve_ik


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dx", type=float, default=0.05, help="metres to move in +x")
    p.add_argument("--gravity-comp", action="store_true",
                   help="cancel gravity/Coriolis each step, as the real UR controller does")
    args = p.parse_args()

    ctrl = URControllerMuJoCo()
    site = ctrl._tcp_site_id

    start = ctrl.data.site_xpos[site].copy()
    target = start.copy()
    target[0] += args.dx
    mat = ctrl.data.site_xmat[site].reshape(3, 3).copy()

    q_sol, ok = solve_ik(ctrl.model, ctrl.data, site, target, mat, **ctrl._ik_cfg)
    print(f"IK converged: {ok}")
    if not ok:
        print("IK failed - nothing to settle. Try a smaller --dx.")
        return

    q_start = ctrl.data.qpos[:6].copy()
    print(f"commanded joint travel: {np.abs(q_sol - q_start).max():.4f} rad")
    print(f"target TCP: ({target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f})\n")

    ctrl.data.ctrl[:6] = q_sol

    print(f"{'steps':>7} {'sim time':>9} {'joint err':>12} {'tcp err':>10}")
    print("-" * 42)

    n = 0
    prev = None
    for chunk in (50, 50, 100, 300, 500, 1000, 3000, 5000, 10000):
        for _ in range(chunk):
            if args.gravity_comp:
                # qfrc_bias is gravity + Coriolis + centrifugal; cancelling it means
                # the position servo no longer has to sit at an error to hold station
                ctrl.data.qfrc_applied[:6] = ctrl.data.qfrc_bias[:6]
            mujoco.mj_step(ctrl.model, ctrl.data)
        n += chunk

        q_err = float(np.abs(ctrl.data.qpos[:6] - q_sol).max())
        tcp_err = float(np.linalg.norm(ctrl.data.site_xpos[site] - target))
        print(f"{n:>7} {n * ctrl.control_dt:>8.2f}s {q_err:>11.5f}r {tcp_err * 1000:>8.2f}mm")

        if prev is not None and abs(prev - tcp_err) < 1e-6:
            print("\n-> error has stopped changing: STEADY-STATE, not a settle-time issue.")
            break
        prev = tcp_err
    else:
        print("\n-> still converging at the last sample; a longer settle window helps.")

    print(f"\nfor reference, move_joints currently settles for 50 steps "
          f"({50 * ctrl.control_dt:.2f}s)")


if __name__ == "__main__":
    main()

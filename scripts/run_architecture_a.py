"""run_architecture_a.py

It takes an instruction and gives you the UR5e carrying it out through the
modular pipeline, printing a per-stage trace.

    uv run python scripts/run_architecture_a.py \\
        --instruction "place that red cube into that metal tray"

The detector vocabulary is derived from the instruction by the planner, so there
is nothing to configure per task.

Omit --instruction for an interactive session that keeps the camera, detector and
arm connected between commands. Prefer that while developing: reloading OWLv2 per
instruction takes longer than the run.

Needs a real UR5e, a RealSense, a saved hand-eye calibration, and
ANTHROPIC_API_KEY in the environment. It moves a physical arm, so read the
workspace block in config.yaml before the first run and make sure the bounds
match the bench you are standing at.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", nargs="+", default=None,
                   help="DEBUG ONLY: pin the detector vocabulary and skip grounding. "
                        "Leave unset for any recorded trial, or Architecture A is handed "
                        "a list of what is on the table that Architecture B never gets.")
    p.add_argument("--instruction", default=None,
                   help="single instruction to run; omit for an interactive session")
    p.add_argument("--no-reperceive", action="store_true",
                   help="plan once against the opening scene instead of re-detecting each turn "
                        "(open-loop; use it to measure what closing the loop is worth)")
    args = p.parse_args()

    from fyp.architecture_a.pipeline import ArchitectureA
    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.ur5e_controller import URController

    controller = URController()
    controller.gripper_start()

    try:
        with RealSenseCamera() as camera:
            agent = ArchitectureA(camera, controller, args.queries,
                                  reperceive=not args.no_reperceive)

            instructions = ([args.instruction] if args.instruction
                            else _interactive())

            for instruction in instructions:
                outcome = agent.run(instruction)
                print(f"\n{'=' * 70}")
                print(f"  {'SUCCESS' if outcome.ok else 'FAILED'}: {outcome.summary}")
                if outcome.failed_stage:
                    print(f"  failed at stage: {outcome.failed_stage}")
                if outcome.queries:
                    print(f"  looked for: {outcome.queries}")
                if outcome.executed:
                    print(f"  executed: {' -> '.join(outcome.executed)}")
                if outcome.asked:
                    print(f"  asked: {outcome.asked}")
                print(f"  tokens: {outcome.usage.get('input_tokens', 0)} in / "
                      f"{outcome.usage.get('output_tokens', 0)} out "
                      f"over {outcome.usage.get('calls', 0)} calls")
                print(f"  trace: logs/traces/{outcome.run_id}.jsonl")
                print("=" * 70)
    finally:
        controller.close()

    return 0


def _interactive():
    """It takes nothing and gives you instructions typed at the prompt, until EOF."""
    print("Type an instruction, or Ctrl-D to quit.")
    while True:
        try:
            line = input("\ninstruction > ").strip()
        except EOFError:
            print()
            return
        if line:
            yield line


if __name__ == "__main__":
    sys.exit(main())

"""run_evaluation.py

It takes an architecture and a tier and gives you a scored trial block on disk.

    # print the layouts for a tier so you can set the cell up in advance
    uv run python scripts/run_evaluation.py --tier 0 --layouts-only

    # run the block (lerobot venv: Architecture A needs PyTorch for OWLv2)
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/run_evaluation.py --architecture A --tier 0

    # rebuild the results tables from every CSV recorded so far
    uv run python scripts/run_evaluation.py --summarise

Every trial is spoken: the harness shows the utterance, you press Enter and say
it, and the architecture gets the transcript. Clarifying questions are answered
by voice as well. Only the operator's scoring (success, interventions, notes) is
typed, because that is the experimenter recording a result, not input to the
robot. Architecture A also stops after each step (pick, place, ...) for a typed
y/n on whether that step worked. A never sees the answer; it goes to the trace and
to <architecture>_tier<n>_steps.csv beside the trial CSV.

Moves a physical arm. Read the tier's success criterion before starting, because
you are the one scoring it and consistency between trials is the whole point.
"""
from __future__ import annotations

import argparse
import sys

from fyp.evaluation.harness import Harness, export_summary, judge_step, load_records, summarise
from fyp.evaluation.suite import TASKS, Tier, build_layouts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", choices=["A", "B"], help="which architecture to score")
    p.add_argument("--tier", type=int, choices=[int(t) for t in Tier])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--layouts-only", action="store_true",
                   help="print the layouts and exit, without touching the robot")
    p.add_argument("--summarise", action="store_true", help="rebuild the tables and exit")
    p.add_argument("--input-device", default=None,
                   help="microphone index or name; scripts/listen.py --devices lists them")
    p.add_argument("--cpu", action="store_true", help="run the recogniser on CPU")
    args = p.parse_args()

    if args.summarise:
        print(summarise(load_records()))
        print(f"\nwritten to {export_summary()}")
        return 0

    if args.tier is None:
        p.error("--tier is required unless --summarise")
    tier = Tier(args.tier)
    task = TASKS[tier]

    if args.layouts_only:
        print(f"Tier {int(tier)}: {task.name}")
        print(f"objects: {', '.join(task.objects)}")
        print(f"success: {task.success_criterion}")
        if task.perturbation:
            print(f"PERTURBATION: {task.perturbation}")
        print()
        for layout in build_layouts(tier, args.trials):
            print(layout.describe())
        return 0

    if not args.architecture:
        p.error("--architecture is required to run trials")

    if args.architecture == "A":
        from fyp.architecture_a.pipeline import ArchitectureA
        from fyp.shared.hardware.camera import RealSenseCamera
        from fyp.shared.hardware.ur5e_controller import URController
        from fyp.shared.speech import SpeechToText

        # Loaded before the arm, as in run_architecture_a.py: a missing
        # PortAudio or CUDA library should fail with the arm still untouched.
        ears = SpeechToText(input_device=_as_device(args.input_device),
                            device="cpu" if args.cpu else None)
        print(f"  recogniser: {ears.model_name} on {ears.device}, "
              f"loaded in {ears.load_s:.1f}s")
        controller = URController()
        controller.gripper_start()
        try:
            with RealSenseCamera() as camera:
                # A gets only the instruction, the same string Architecture B
                # receives, and chooses its own detector words in perceive.
                # judge_step: the operator types y/n after each step. Recorded
                # only, never given to A (see ArchitectureA).
                agent = ArchitectureA(camera, controller, ask=ears.answer, judge=judge_step)
                Harness(agent, "A", ears, n_trials=args.trials).run_tier(tier)
        finally:
            controller.close()
            ears.close()
    else:
        print("Architecture B is not built yet (W6). Its wrapper must expose "
              "run(instruction) -> outcome with .ok, .failed_stage, .asked, .run_id.")
        return 1

    print(summarise(load_records()))
    return 0


def _as_device(value):
    """It takes the --input-device string and gives you an int index if it is one."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    sys.exit(main())

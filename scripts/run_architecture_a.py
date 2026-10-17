"""run_architecture_a.py

It takes a spoken instruction and gives you the UR5e carrying it out through the
modular pipeline, printing a per-stage trace.

    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/run_architecture_a.py

The lerobot venv, not `uv run`, because OWLv2 needs PyTorch and the FYP venv has
none.

Speech is the ONLY input (13 Oct 2026). The instruction is spoken, and so is the
answer to any clarifying question the planner asks. The typed escapes `--text`
and `--instruction` were removed on purpose: a spoken command is what the project
claims to support, and a run that can be typed into is a run that can quietly
stop testing that. To debug the recogniser without the arm, use
`scripts/listen.py`. Enter still starts and stops each recording (push to talk);
only the words are spoken.

The wiring lives here rather than in the pipeline on purpose. `ArchitectureA.run`
takes a plain string and must keep taking one, because ASR is a CONTROLLED
VARIABLE: the same transcript reaches Architecture A and Architecture B
unchanged. A pipeline that owned a microphone would be a pipeline that could
quietly clean up its own input, and the comparison would stop meaning anything.

The detector vocabulary is derived from the instruction by the planner, so there
is nothing to configure per task.

Before every instruction it saves what the camera sees into the trace folder,
named after the run so it sits next to `<run_id>.jsonl`:

    <run_id>_colour.png   the colour frame, which is all OWLv2 ever sees
    <run_id>_depth.png    the depth after the same patch median `depth_at` uses,
                          histogram-equalised jet like realsense-viewer, black
                          where the pipeline would get no depth

Both are taken just BEFORE the run, outside its trace, so the second or so they
cost never counts as Architecture A's latency. The arm is still at rest, so it is
the same scene the run's own first capture sees, not literally the same frame.
`run_evaluation.py` does not save them.

The session keeps the camera, detector and arm connected between commands, so
OWLv2 loads once rather than per instruction.

Needs a real UR5e, a RealSense, a saved hand-eye calibration, and
ANTHROPIC_API_KEY in the environment. Speech additionally needs a microphone and
PortAudio, `sudo apt install libportaudio2` on Ubuntu. It moves a physical arm, so read the
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
    p.add_argument("--no-reperceive", action="store_true",
                   help="plan once against the opening scene instead of re-detecting each turn "
                        "(open-loop; use it to measure what closing the loop is worth)")
    p.add_argument("--input-device", default=None,
                   help="microphone index or name; scripts/listen.py --devices lists them")
    p.add_argument("--cpu", action="store_true",
                   help="run the recogniser on CPU, for a machine with no CUDA")
    args = p.parse_args()

    # Opened BEFORE the arm and the camera. Loading large-v3 takes several
    # seconds and fails outright on a missing PortAudio, and neither is worth
    # discovering with a live arm connected and a gripper already powered up.
    ears = _open_ears(args)

    from fyp.architecture_a.pipeline import ArchitectureA
    from fyp.evaluation.harness import judge_step
    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.ur5e_controller import URController

    controller = URController()
    controller.gripper_start()

    try:
        with RealSenseCamera() as camera:
            # Clarifying questions are answered by voice too, through the same
            # recogniser, so no step of a run reads the keyboard for user input.
            agent = ArchitectureA(camera, controller, args.queries,
                                  ask=ears.answer,
                                  reperceive=not args.no_reperceive,
                                  # Typed y/n after each step, recorded in the
                                  # trace only. A never sees it.
                                  judge=judge_step)

            for instruction in _spoken(ears):
                scene = _save_scene(camera)
                outcome = agent.run(instruction)
                scene = _name_after_run(scene, outcome.run_id)
                print(f"\n{'=' * 70}")
                # A's own word, not a verdict: ok means only that the planner stopped.
                print(f"  A {'REPORTED DONE' if outcome.ok else 'REPORTED FAILURE'}: {outcome.summary}")
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
                for path in scene.values():
                    print(f"  scene: {path}")
                print("=" * 70)
    finally:
        controller.close()
        ears.close()

    return 0


def _save_scene(camera) -> dict:
    """
    It takes the open camera and gives you the paths of two PNGs of the table as
    the run is about to see it: the colour frame, and the smoothed depth.

    A failure here only prints a warning. The run's own capture stage is where a
    camera fault belongs, because that is what the trace records, and crashing
    here would stop that record from being written at all.
    """
    from datetime import datetime, timezone

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fyp.architecture_a.perception.pixel_to_3d import smooth_depth
    from fyp.shared.helpers.config import get_config, resolve

    try:
        cfg = get_config()
        out = resolve(cfg["architecture_a"]["trace"]["dir"])
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-prerun"
        paths = {"colour": out / f"{stem}_colour.png", "depth": out / f"{stem}_depth.png"}

        frame = camera.capture()
        d = smooth_depth(frame.depth, int(cfg["camera"]["depth_patch_radius"]),
                         float(cfg["camera"]["max_depth"]))

        # Histogram-equalised, as in check_camera.py: the table spans a small slice
        # of the range, and a linear scale paints it all one colour. A colour is
        # therefore NOT a fixed distance; the xyz in the trace are the numbers.
        valid = np.isfinite(d)
        eq = np.zeros_like(d)
        if valid.any():
            v = d[valid]
            eq[valid] = np.searchsorted(np.sort(v), v, side="right") / v.size
        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad("black")

        plt.imsave(paths["colour"], frame.rgb)
        plt.imsave(paths["depth"], np.ma.masked_where(~valid, eq),
                   cmap=cmap, vmin=0.0, vmax=1.0)
        return paths
    except Exception as e:
        print(f"  WARNING: could not save the pre-run images ({type(e).__name__}: {e})")
        return {}


def _name_after_run(paths: dict, run_id: str) -> dict:
    """
    It takes the pre-run image paths and gives you them renamed to
    `<run_id>_colour.png` and `<run_id>_depth.png`.

    The run id only exists once the run starts, and the images have to be taken
    before it, so they are saved under a time stamp and renamed here.
    """
    renamed = {}
    for kind, path in paths.items():
        new = path.with_name(f"{run_id}_{kind}.png")
        path.rename(new)
        renamed[kind] = new
    return renamed


def _open_ears(args):
    """
    It takes the parsed arguments and gives you a loaded recogniser.

    Always loaded: there is no typed path left to skip it for.
    """
    from fyp.shared.speech import SpeechToText

    print("loading the recogniser, this takes a few seconds the first time ...")
    ears = SpeechToText(input_device=_as_device(args.input_device),
                        device="cpu" if args.cpu else None)
    print(f"  {ears.model_name} on {ears.device}/{ears.compute_type}, "
          f"loaded in {ears.load_s:.1f}s")
    print(f"  transcripts and audio logged to {ears.log_dir}")
    return ears


def _spoken(ears):
    """
    It takes a loaded recogniser and gives you instructions spoken at the
    microphone, until EOF.

    The transcript is passed on exactly as heard. No punctuation stripping, no
    filler removal, no lowercasing, because Architecture B receives the same
    string and cleaning it for one side and not the other turns speech into a
    confound. A misheard word is data about the speech stack, and the log written
    by `SpeechToText` is what later separates "the architecture failed" from "the
    microphone did".
    """
    print("\nEnter starts recording, Enter stops it. Ctrl-D to quit.")
    while True:
        try:
            # hear() retries on silence and echoes what it heard before the arm
            # moves, where a mishearing is cheapest to catch.
            transcript = ears.hear()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        yield transcript.text


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

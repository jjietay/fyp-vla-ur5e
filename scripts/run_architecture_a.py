"""run_architecture_a.py

It takes a spoken instruction and gives you the UR5e carrying it out through the
modular pipeline, printing a per-stage trace.

    uv run python scripts/run_architecture_a.py                 # speak to it
    uv run python scripts/run_architecture_a.py --text          # type to it
    uv run python scripts/run_architecture_a.py --instruction "put the cube in the tray"

Speech is the DEFAULT, because a spoken command is what the project claims to
support and a recorded trial has to be one. `--text` is the debugging escape and
skips loading the recogniser entirely, which is worth several seconds per run.

The wiring lives here rather than in the pipeline on purpose. `ArchitectureA.run`
takes a plain string and must keep taking one, because ASR is a CONTROLLED
VARIABLE: the same transcript reaches Architecture A and Architecture B
unchanged. A pipeline that owned a microphone would be a pipeline that could
quietly clean up its own input, and the comparison would stop meaning anything.

The detector vocabulary is derived from the instruction by the planner, so there
is nothing to configure per task.

Both interactive modes keep the camera, detector and arm connected between
commands. Prefer them while developing: reloading OWLv2 per instruction takes
longer than the run.

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
    p.add_argument("--instruction", default=None,
                   help="single instruction to run; omit for an interactive session")
    p.add_argument("--no-reperceive", action="store_true",
                   help="plan once against the opening scene instead of re-detecting each turn "
                        "(open-loop; use it to measure what closing the loop is worth)")
    p.add_argument("--text", action="store_true",
                   help="type instructions instead of speaking them; skips loading the "
                        "recogniser, so use it while debugging anything that is not speech")
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
    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.ur5e_controller import URController

    controller = URController()
    controller.gripper_start()

    try:
        with RealSenseCamera() as camera:
            agent = ArchitectureA(camera, controller, args.queries,
                                  reperceive=not args.no_reperceive)

            instructions = ([args.instruction] if args.instruction
                            else _typed() if ears is None
                            else _spoken(ears))

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
        if ears is not None:
            ears.close()

    return 0


def _open_ears(args):
    """
    It takes the parsed arguments and gives you a loaded recogniser, or None when
    this run does not need one.

    None means the caller reads from the keyboard. Returning it rather than
    raising keeps the two text paths, `--instruction` and `--text`, free of any
    speech import at all, so they still work on a machine with no microphone and
    no PortAudio.
    """
    if args.instruction or args.text:
        return None

    from fyp.shared.speech import SpeechToText

    print("loading the recogniser, this takes a few seconds the first time ...")
    ears = SpeechToText(input_device=_as_device(args.input_device),
                        device="cpu" if args.cpu else None)
    print(f"  {ears.model_name} on {ears.device}/{ears.compute_type}, "
          f"loaded in {ears.load_s:.1f}s")
    print(f"  transcripts and audio logged to {ears.log_dir}")
    return ears


def _typed():
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
    from fyp.shared.speech import NoAudioCaptured

    print("\nEnter starts recording, Enter stops it. Ctrl-D to quit.")
    while True:
        try:
            transcript = ears.listen()
        except NoAudioCaptured as e:
            print(f"\n  {e}\n")
            continue
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not transcript:
            print("  nothing recognised, try again")
            continue

        # Echoed before running, because the arm is about to move on the strength
        # of this string and a mishearing is much cheaper to catch here.
        print(f'  heard: "{transcript.text}"')
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

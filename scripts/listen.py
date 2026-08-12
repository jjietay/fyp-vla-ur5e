"""listen.py

It takes your voice and gives you the transcript, printed, with nothing else
attached. No robot, no camera, no API key.

    uv run python scripts/listen.py                  # loop until Ctrl-D
    uv run python scripts/listen.py --once
    uv run python scripts/listen.py --devices        # which microphones exist
    uv run python scripts/listen.py --cpu            # no GPU on this machine
    uv run python scripts/listen.py --file clip.wav  # no microphone at all

This is the whole speech front end in isolation, which is the point: when a
trial goes wrong in the lab you need to know whether the microphone or the
architecture was at fault, and the only way to answer that quickly is to be able
to run this alone.

Same `SpeechToText` object the pipelines use, so a transcript that reads
correctly here reads correctly there.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true",
                   help="transcribe a single utterance and exit")
    p.add_argument("--devices", action="store_true",
                   help="list input devices and exit; start here if clips come back silent")
    p.add_argument("--input-device", default=None,
                   help="microphone index or name from --devices, if the default is wrong")
    p.add_argument("--model", default=None,
                   help="override the model in config, e.g. tiny.en for a fast smoke test")
    p.add_argument("--cpu", action="store_true",
                   help="run on CPU; slower, but works on a machine with no CUDA")
    p.add_argument("--file", default=None,
                   help="transcribe a wav instead of recording, for testing with no mic")
    args = p.parse_args()

    from fyp.shared.speech import NoAudioCaptured, SpeechToText, list_input_devices

    if args.devices:
        print(list_input_devices())
        return 0

    device = "cpu" if args.cpu else None
    input_device = _as_device(args.input_device)

    print("loading the model, this takes a few seconds the first time ...")
    with SpeechToText(input_device=input_device, model=args.model, device=device) as ears:
        print(f"  {ears.model_name} on {ears.device}/{ears.compute_type}, "
              f"loaded in {ears.load_s:.1f}s")
        print(f"  logging to {ears.log_dir}")

        if args.file:
            return _from_file(ears, Path(args.file))

        print("\nEnter starts recording, Enter stops it. Ctrl-D to quit.")
        while True:
            try:
                t = ears.listen()
            except NoAudioCaptured as e:
                print(f"\n  {e}\n")
                if args.once:
                    return 1
                continue
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            _report(t)
            if args.once:
                return 0 if t else 1


def _report(t) -> None:
    """It takes a Transcript and prints it the way you want to read it live."""
    print(f"\n{'=' * 70}")
    print(f"  {t.text!r}" if t else "  (nothing recognised)")
    # Real time factor is the number worth watching: above 1.0 the recogniser is
    # slower than the person speaking, and the demo will feel broken.
    rtf = t.latency_s / t.duration_s if t.duration_s else float("nan")
    print(f"  {t.duration_s:.1f}s audio, {t.latency_s:.1f}s to transcribe "
          f"(RTF {rtf:.2f}), language {t.language}")
    if t.audio_path:
        print(f"  audio: {t.audio_path}")
    print("=" * 70)


def _from_file(ears, path: Path) -> int:
    """
    It takes a wav path and gives you the transcript, bypassing the microphone.

    Useful for checking the recogniser against a clip you already know the words
    to, so a bad transcript can be pinned on the model rather than on the room.
    """
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != ears.samplerate:
            print(f"{path} is {w.getframerate()} Hz, expected {ears.samplerate}. "
                  f"Resample it first, or the transcript will be nonsense.",
                  file=sys.stderr)
            return 1
        if w.getsampwidth() != 2:
            print(f"{path} is not 16-bit PCM", file=sys.stderr)
            return 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    _report(ears.transcribe(pcm.astype(np.float32) / 32768.0))
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

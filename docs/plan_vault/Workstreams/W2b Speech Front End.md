---
status: written, never heard a real microphone
needs_lab: false
week: 2
---

# W2b Speech Front End

The user speaks, the system acts. About two days of work, and none of it is novel.

## The architectural rule

ASR sits **outside both architectures** and is identical for both. One microphone, one model, one transcript, handed to A and to B unchanged. Clean up the transcript for one and not the other and speech becomes a confound that invalidates the comparison. Reasoning in [[Speech Stack]].

## Steps

- [x] `shared/speech.py`: push to talk capture, then `faster-whisper`, then text, and nothing else
- [x] set `compute_type="float16"` explicitly, since the default int8 crashes on RTX 50 series
- [x] runnable alone, `scripts/listen.py`, so a bad transcript can be pinned on the microphone rather than the architecture
- [ ] keep a `--text` flag so either architecture can be driven from the keyboard while debugging
- [x] log every transcript with the audio, timestamped
- [x] paraphrase sets written, living in `evaluation/suite.py` as `train_utterances` and `heldout_utterances` for all five tiers
- [ ] optional text to speech for spoken clarification, if [[Tier 2 Ambiguity]] is working by W9

## The deadline hidden in here

The paraphrase sets must exist **before** [[W5 Demonstration Capture]] starts. The instruction string is per episode metadata, so varying it during recording is free, but retrofitting it means re recording everything. Miss this and the phrasing robustness hypothesis becomes untestable.

## Status 11 Aug 2026

Nothing here is built except the paraphrase sets, which landed early because the [[Evaluation Protocol]] needed them anyway. Five training phrasings and three held out per tier.

That ordering matters. The sets had to exist before recording starts, not before speech works, since the instruction string is per episode metadata and retrofitting it means re recording everything.

One thing already changed downstream: the grounding step in [[W4 Architecture A End to End]] consumes the transcript, so when speech lands it feeds an interface that is already waiting for it.

## Status 12 Aug 2026

`shared/speech.py` and `scripts/listen.py` are written and unit tested against a stubbed recogniser. No microphone has been attached, so nothing here has heard a real voice.

Push to talk is Enter to start, Enter to stop, rather than a held key. A held key needs a real display session and breaks the moment the cell is driven over SSH, and the [[Speech Stack]] note only ever asked for one complete clip.

Two things to expect the first time it runs on the lab machine:

* `sounddevice` needs PortAudio present, `sudo apt install libportaudio2` on Ubuntu, and the error message says so
* a USB headset does not become the default input just by being plugged in, so `--devices` is the first thing to try when a clip comes back silent

The `--text` flag is still open because nothing calls the module yet. `ArchitectureA.run` takes a plain string and must keep taking one, so the wiring belongs in the runner script rather than in the pipeline.

---
status: paraphrase sets done, speech not started
needs_lab: false
week: 2
---

# W2b Speech Front End

The user speaks, the system acts. About two days of work, and none of it is novel.

## The architectural rule

ASR sits **outside both architectures** and is identical for both. One microphone, one model, one transcript, handed to A and to B unchanged. Clean up the transcript for one and not the other and speech becomes a confound that invalidates the comparison. Reasoning in [[Speech Stack]].

## Steps

- [ ] `shared/speech.py`: push to talk capture, then `faster-whisper`, then text, and nothing else
- [ ] set `compute_type="float16"` explicitly, since the default int8 crashes on RTX 50 series
- [ ] keep a `--text` flag so either architecture can be driven from the keyboard while debugging
- [ ] log every transcript with the audio, timestamped
- [x] paraphrase sets written, living in `evaluation/suite.py` as `train_utterances` and `heldout_utterances` for all five tiers
- [ ] optional text to speech for spoken clarification, if [[Tier 2 Ambiguity]] is working by W9

## The deadline hidden in here

The paraphrase sets must exist **before** [[W5 Demonstration Capture]] starts. The instruction string is per episode metadata, so varying it during recording is free, but retrofitting it means re recording everything. Miss this and the phrasing robustness hypothesis becomes untestable.

## Status 11 Aug 2026

Nothing here is built except the paraphrase sets, which landed early because the [[Evaluation Protocol]] needed them anyway. Five training phrasings and three held out per tier.

That ordering matters. The sets had to exist before recording starts, not before speech works, since the instruction string is per episode metadata and retrofitting it means re recording everything.

One thing already changed downstream: the grounding step in [[W4 Architecture A End to End]] consumes the transcript, so when speech lands it feeds an interface that is already waiting for it.

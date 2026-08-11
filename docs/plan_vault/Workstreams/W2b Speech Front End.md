---
status: not started
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
- [ ] write the paraphrase sets, 5 to 10 phrasings per task, split into a record set and a held out set
- [ ] optional text to speech for spoken clarification, if [[Tier 2 Ambiguity]] is working by W9

## The deadline hidden in here

The paraphrase sets must exist **before** [[W5 Demonstration Capture]] starts. The instruction string is per episode metadata, so varying it during recording is free, but retrofitting it means re recording everything. Miss this and the phrasing robustness hypothesis becomes untestable.

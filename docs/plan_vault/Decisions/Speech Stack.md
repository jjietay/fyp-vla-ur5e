---
status: decided
---

# Speech Stack

**`faster-whisper` running locally, push to talk, `compute_type="float16"`.**

## Not a contribution

Speech recognition is an off the shelf component here, like the camera driver. Say so explicitly in [[D4 Final Report]], because claiming it invites an examiner to probe an area with no novel work in it.

## Why local rather than a cloud API

Cost is **not** the reason. Total spoken audio across the project is roughly 500 utterances at 5 seconds, about 45 minutes, which is under a dollar at any provider. Every option is effectively free at this volume.

The reason is the demonstration. A local model cannot fail because the campus network did, and [[D5 Demonstration]] happens in front of an examiner. Keep a cloud path behind a config flag as a fallback.

## The Blackwell trap

CTranslate2, the runtime under `faster-whisper`, has a known incompatibility with sm_120. The default int8 quantisation crashes with `CUBLAS_STATUS_NOT_SUPPORTED`, because Blackwell int8 tensor cores need padding older builds do not emit. **Force `compute_type="float16"`.** Also ensure CUDA 12 and cuDNN 9.

VRAM is not a constraint: `large-v3` at float16 needs about 3.4 GB, so use the largest model.

## Push to talk, not always on

Capturing a complete 3 to 5 second clip sidesteps endpointing, voice activity detection and partial hypotheses entirely. A robotics lab is noisy. Wake word detection is a research project of its own and is out of scope.

## The rule that protects the comparison

**Both architectures get the same raw transcript.** Architecture A absorbs disfluent speech natively because it is an LLM. Architecture B takes the instruction as a string conditioned on its training distribution and has no way to normalise wording. Normalising the transcript for B inserts an LLM into B and it stops being end to end.

That asymmetry is measured rather than patched, see [[Hypotheses]].

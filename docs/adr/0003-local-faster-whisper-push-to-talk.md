# Speech is local faster-whisper, push to talk, float16

Decided and working on the GPU, 31 Aug 2026.

Speech recognition is an off the shelf component here, like the camera driver. Say so explicitly in the D4 Final Report, because claiming it invites an examiner to probe an area with no novel work in it.

## Why local rather than a cloud API

Cost is not the reason. Total spoken audio across the project is roughly 500 utterances at 5 seconds, under a dollar at any provider.

The reason is the demonstration. A local model cannot fail because the campus network did, and the D5 Demonstration happens in front of an examiner. Keep a cloud path behind a config flag as a fallback.

## Two CUDA traps, both hit

`compute_type` must be **float16**. CTranslate2's default int8 crashes on sm_120 with `CUBLAS_STATUS_NOT_SUPPORTED`, because Blackwell int8 tensor cores need padding older builds do not emit.

CTranslate2 also needs **cuBLAS for CUDA 12 and cuDNN 9**, and dlopens them by bare name so it searches only the loader path. This venv has no torch by design, so nothing pulled them in, and the symptom is dishonest: the model loads, reports `cuda/float16`, and fails on the first utterance. Both are now dependencies, and `shared/speech.py` preloads them by absolute path so no `LD_LIBRARY_PATH` export is needed.

VRAM is not a constraint. `large-v3` at float16 needs about 3.4 GB, so use the largest model.

## Push to talk, not always on

Capturing a complete 3 to 5 second clip sidesteps endpointing, voice activity detection and partial hypotheses entirely. A robotics lab is noisy, and wake word detection is a research project of its own.

Enter starts, Enter stops, rather than a held key. A held key needs a real display session and breaks the moment the cell is driven over SSH.

## The rule that protects the comparison

**Both architectures get the same raw transcript.** Architecture A absorbs disfluent speech natively because it is an LLM. Architecture B takes the string conditioned on its training distribution and has no way to normalise wording. Cleaning the transcript for B inserts an LLM into B and it stops being end to end. That asymmetry is measured rather than patched, see the hypotheses in `docs/fyp_plan.md` §5.

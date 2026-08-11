---
status: code complete, unrun on hardware
needs_lab: true
week: 5
---

# W4 Architecture A End to End

Wiring [[W2 Architecture A Software]] and [[W3 Camera and Calibration]] into one runnable thing.

## Steps

- [x] single entry point, in `architecture_a/pipeline.py` and `scripts/run_architecture_a.py`
- [x] per stage logging in `architecture_a/trace.py`, one JSONL file per run
- [x] closed loop by default, re perceiving before each planning turn, with `--no-reperceive` to measure what that is worth
- [ ] reach 70 percent success on [[Tier 0 Pick and Place]] over 20 trials

## Build the logging properly the first time

Per stage diagnosability is the entire argument for Architecture A. It is one of the [[Hypotheses]], and this log becomes the evidence in the final report. Retrofitting it later means re running every trial.

Closing the loop is also what makes [[Tier 3 Drawer]] possible at all. Document the failure modes encountered while doing it, because they are results rather than bugs.

## Status 11 Aug 2026

Built and wired, never run on hardware. The instruction still arrives typed, because [[W2b Speech Front End]] is not built yet.

**The stage list changed.** It was going to be ASR, detection, depth, transform, planning, execution. It is now:

    capture, grounding, detection, depth, transform, planning, clarify, execution

Two additions, both deliberate:

* `grounding` is the instruction becoming detector queries, see [[Detector Query Format]]. It is its own stage because a wrong noun phrase is not a detection error, and recording it as one would corrupt the per stage breakdown that carries H6
* `clarify` is the run pausing to ask the user, which is the [[Tier 2 Ambiguity]] mechanism

`ASR` is not a stage yet and will be added with the speech front end.

The remaining box, 70 percent on [[Tier 0 Pick and Place]], needs the arm.

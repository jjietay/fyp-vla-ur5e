---
status: blocked
needs_lab: true
week: 5
---

# W4 Architecture A End to End

Wiring [[W2 Architecture A Software]] and [[W3 Camera and Calibration]] into one runnable thing.

## Steps

- [ ] single entry point: spoken instruction in, executed motion out
- [ ] per stage logging, tagging every failure as ASR, detection, depth, transform, planning or execution
- [ ] close the loop by re detecting before each grasp rather than trusting the plan time position
- [ ] reach 70 percent success on [[Tier 0 Pick and Place]] over 20 trials

## Build the logging properly the first time

Per stage diagnosability is the entire argument for Architecture A. It is one of the [[Hypotheses]], and this log becomes the evidence in the final report. Retrofitting it later means re running every trial.

Closing the loop is also what makes [[Tier 3 Drawer]] possible at all. Document the failure modes encountered while doing it, because they are results rather than bugs.

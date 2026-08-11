---
status: code complete, unrun on hardware
needs_lab: false
week: 2
---

# W2 Architecture A Software

The whole modular pipeline except the camera. Buildable now, against saved PNGs as fixtures, so it does not wait on lab access.

## Steps

- [x] `architecture_a/skills.py` with `pick`, `place`, `pour` and `open_drawer`
- [x] tool schemas in `tools.py`, matched to the signatures by an automated check at startup
- [x] `ask_user(question, options)` exposed alongside the motion primitives
- [x] `architecture_a/planner.py`: system prompt lists skills plus detected objects with base frame positions, model returns an ordered tool call plan
- [x] plan validation **before** dispatch, rejecting unknown skills, out of workspace coordinates and malformed arguments
- [x] clarification loop: on `ask_user`, surface the question, block for input, append the answer, re plan
- [x] query grounding: the instruction chooses the detector vocabulary, see [[Detector Query Format]]

## Rules

Primitives are written against the **shared controller interface only**. If a primitive reaches past the controller for a ur_rtde specific attribute, it is wrong, and the comparison starts measuring plumbing instead of architectures.

Parameterise approach height, descent speed, grasp width, tilt angle and tilt rate. The planner will want to vary all of them.

`server.py` already reaches into five private controller attributes. The primitives will want some of the same things. Promote them to a public API rather than adding a sixth reach through.

## Do not skip validation

A plan that reaches the controller unvalidated is how an arm gets damaged. This is the one place in the project where a software bug has a physical consequence. Related safety envelope for the other architecture is in [[W6 Architecture B Training]].

## Query grounding, added 11 Aug 2026

The instruction now chooses the detector vocabulary, instead of it being passed on the command line.

* the planner turns the transcript into short noun phrases, then OWLv2 searches for those
* same model as the planner, so no new component enters Architecture A
* it is its own trace stage, because a vocabulary error is not a detection error and recording it as one would corrupt the per stage breakdown
* extraction failure raises rather than falling back to a default list, since a silent fallback would restore the advantage this removes

Why it matters: Architecture A now starts from exactly the string Architecture B gets. Before, a human supplied the object list, which was information B never received and which weakened [[Tier 4 Generalisation]].

## Built 11 Aug 2026

Written against real hardware only, so none of it has been executed. What still gates a first run:

* measure the workspace bounds, `down_rotvec` and `robot.home_q` on the real cell
* run the hand-eye procedure, since the pipeline refuses to start without `T_base_cam.json`
* set `ANTHROPIC_API_KEY`

19 hardware-free checks cover the safety envelope and plan validation, which are the two things standing between generated output and a physical arm.

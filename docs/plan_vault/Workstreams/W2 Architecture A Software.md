---
status: not started
needs_lab: false
week: 2
---

# W2 Architecture A Software

The whole modular pipeline except the camera. Buildable now, against saved PNGs as fixtures, so it does not wait on lab access.

## Steps

- [ ] `architecture_a/skills.py` with `pick`, `place`, `pour` and `open_drawer`
- [ ] Anthropic SDK hello world, then tool schemas matching those signatures exactly
- [ ] add an `ask_user(question, options)` tool alongside the motion primitives
- [ ] `architecture_a/planner.py`: system prompt lists skills plus detected objects with base frame positions, model returns an ordered tool call plan
- [ ] plan validation **before** dispatch, rejecting unknown skills, out of workspace coordinates and malformed arguments
- [ ] clarification loop: on `ask_user`, surface the question, block for input, append the answer, re plan

## Rules

Primitives are written against the **shared controller interface only**. If a primitive reaches past the controller for a ur_rtde specific attribute, it is wrong, and the comparison starts measuring plumbing instead of architectures.

Parameterise approach height, descent speed, grasp width, tilt angle and tilt rate. The planner will want to vary all of them.

`server.py` already reaches into five private controller attributes. The primitives will want some of the same things. Promote them to a public API rather than adding a sixth reach through.

## Do not skip validation

A plan that reaches the controller unvalidated is how an arm gets damaged. This is the one place in the project where a software bug has a physical consequence. Related safety envelope for the other architecture is in [[W6 Architecture B Training]].

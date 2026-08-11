---
tier: 0
status: not started
---

# Tier 0 Pick and Place

Spoken: *"Place that red cube into that metal tray."*

Single unambiguous target, static scene, fixed tray. The deictic phrasing is deliberate, see [[Speech Stack]].

## Role

**Verification, not a headline result.** It proves both pipelines are wired correctly and that they are commanding the same robot through the same interface. If a pipeline cannot do this, nothing above it means anything.

## Targets

* Architecture A: 70 percent over 20 trials
* Architecture B: 50 percent over 20 trials

The thresholds differ because A is hand engineered and should be reliable on a task this simple, while B is learning from 50 demonstrations and will not be.

## Demonstrations

First fine tune uses a **constrained workspace**, roughly 10 to 15 cm, rather than the full reachable area. Density beats raw episode count. Widen only once the constrained version works.

Success: cube fully inside the tray, gripper clear, arm returned to home. Written into the harness source, not left to judgement, per [[Evaluation Protocol]].

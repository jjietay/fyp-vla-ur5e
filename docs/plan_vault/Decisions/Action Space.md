---
status: pending
needed_by: W5
---

# Action Space

**Pending. Must be settled before recording starts.**

## The choice

Joint positions, TCP pose, or deltas. Current lean is **joint positions**, matching SmolVLA's joint space convention and the asynchronous control framework it uses.

## The constraint that decides it

The action space **must match what Architecture A's primitives command**. If A commands TCP poses and B outputs joint positions, the two architectures are not solving the same problem and the comparison is unfair in a way a careful examiner will spot immediately.

This is the single most important paragraph in the methodology chapter of [[D4 Final Report]], so write down the choice **and the reason** at the time rather than reconstructing it later.

## Also to lock

* observation set: two RGB views plus the state vector
* resolution and control rate, and how the rate is enforced on hardware
* per episode instruction string, drawn from the paraphrase set rather than a single fixed template

## Reading that feeds this

The Open X-Embodiment paper covers action and observation formats across datasets and includes UR5. Read it before deciding, per [[Reading]].

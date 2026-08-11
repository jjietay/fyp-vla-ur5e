---
status: blocked
needs_lab: partly
week: 7
---

# W6 Architecture B Training

Fine tuning SmolVLA and driving the arm from it.

## Steps

- [ ] test a throwaway fine tune in W2 against the 2 existing episodes, batch size 1, purely to find out whether 12 GB is enough
- [ ] fine tune on [[Tier 0 Pick and Place]] first, logging config, seed, wall clock and GPU hours
- [ ] inference loop: checkpoint loads, observation in, action chunk out, chunk size 50
- [ ] safety envelope on every predicted action **before** it reaches the controller, clamping workspace bounds and velocity
- [ ] asynchronous inference so the arm is not idle during the forward pass
- [ ] re train per tier as demonstrations land

## Non negotiables

The safety envelope is not optional. A VLA given a novel observation can emit a chunk that slams the arm into the table.

Keep the checkpoint and the exact dataset version together. A checkpoint whose dataset you cannot identify is not a result, it is an anecdote.

## Read first, build second

The RTC paper on asynchronous inference and action chunking is directly about the problem the async loop solves. Read it **before** designing that loop, per [[Reading]].

## VRAM

12 GB is workable but not roomy. Weights, gradients and optimiser states come to roughly 7 GB before activations, with two camera streams on top. If it does not fit: smaller batch with gradient accumulation, gradient checkpointing, an 8 bit optimiser, or freeze the vision backbone.

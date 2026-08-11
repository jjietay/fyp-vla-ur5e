---
tier: 2
status: not started
---

# Tier 2 Ambiguity

Spoken: *"Pour me a drink."* Both orange juice and water are on the table. The instruction is under determined.

**This is the headline result of the project.**

## What each architecture does

Architecture A exposes `ask_user(question, options)` alongside the motion primitives. When detections are ambiguous relative to the instruction, the planner emits it, gets an answer, and re plans. Adding this is a schema change, not new architecture.

Architecture B **has no channel to ask**. SmolVLA maps images, state and an instruction string to joint action chunks. There is no output head capable of producing a question. Faced with ambiguity it must still act.

## The experiment

Do not patch this asymmetry with an LLM wrapper. Measure it.

- [ ] run the ambiguous instruction at least 20 times against Architecture B
- [ ] record which bottle it picks each time
- [ ] report the distribution

If it picks orange juice 19 times out of 20 regardless of what the user wanted, that is a quantified demonstration that an end to end policy substitutes its training prior for the user's intent. One figure, and it is worth more than any success rate table.

## As a demo

Running A and B side by side on this instruction is the thirty seconds that explains the entire project, see [[D5 Demonstration]].

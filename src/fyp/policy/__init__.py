"""Policies — the thing being compared.

`modular/` is Architecture A: an explicit, per-stage-diagnosable pipeline.
`vla/` will be Architecture B: an end-to-end SmolVLA checkpoint.

The FYP's headline result is a comparison between the two, so they should
eventually sit behind one interface (act(Observation, instruction) -> Action)
and be runnable by the same trial harness.
"""

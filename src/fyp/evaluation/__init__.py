"""The evaluation harness. Neither architecture, and it must stay that way.

`suite.py` is the experimental design: tiers, utterances, layouts, success
criteria, fixed before either pipeline was tuned. `harness.py` runs a trial block
and produces the results tables.

Nothing here imports `architecture_a` or `architecture_b`. The harness talks to
an agent through a single method, `run(instruction) -> outcome`. If it could see
inside one architecture it would grow an affordance for it, and the comparison
would quietly stop being like-for-like.
"""

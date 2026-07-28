"""Architecture A: the modular pipeline.

    frame -> detector -> filters -> localiser -> (hand-eye) -> planner -> skills

Each stage is a separate module so it can fail, and be verified, in isolation.
That per-stage diagnosability is the argument the whole architecture rests on,
so resist the urge to collapse stages together for convenience.

Torch dependency: `detector.py` only. Everything else in this package runs in
the FYP venv.
"""

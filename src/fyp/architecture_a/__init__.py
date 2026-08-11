"""Architecture A: the modular pipeline. Nothing in here is trained.

    speech -> detector -> filters -> localiser -> hand-eye -> planner -> skills

Each stage is a separate module so it can fail, and be diagnosed, in isolation.
That per-stage diagnosability is the argument the whole architecture rests on,
and it is a stated hypothesis in the plan, so resist collapsing stages together
for convenience.

Behaviour changes by editing prompts, tool schemas and primitives. There is no
dataset and no training run anywhere in this package.

`calibration/` lives here rather than in `shared/` because only this
architecture needs it: it converts camera-frame points into base-frame targets,
and Architecture B never forms a 3D point at all.
"""

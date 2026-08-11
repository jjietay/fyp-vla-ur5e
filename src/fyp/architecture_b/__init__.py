"""Architecture B: end-to-end SmolVLA. Nothing in here is hand-authored.

    demos/      teleoperated demonstrations -> HDF5 -> LeRobot dataset
    train.py    fine-tune SmolVLA on that dataset
    inference.py  observation in, 50-step action chunk out, safety-clamped

Behaviour changes only by collecting more data and retraining. There is no
prompt to edit and no primitive to tune, which is precisely the tradeoff being
measured against Architecture A.

The policy maps images, robot state and an instruction STRING to joint actions.
It has no output channel other than actions, so it cannot ask the user a
question. That limitation is a headline result, not a gap to be patched.
"""

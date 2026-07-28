"""Pure, stateless utilities shared across the FYP stack.

ADMISSION RULE — a module belongs here only if it is stateless, imports nothing
else from `fyp` (except other helpers), and could be lifted into an unrelated
project unchanged. Anything that touches a socket, a file, a model checkpoint or
the robot belongs in the subsystem that owns it, not here.

`ik.py` is the one module that stretches the rule: it imports mujoco, because a
numerical IK solver is only needed while the sim is the backend (the real UR5e
solves IK in firmware). It leaves with the sim.
"""

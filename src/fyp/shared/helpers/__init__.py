"""Pure, stateless maths and configuration used by both architectures.

ADMISSION RULE - a module belongs here only if it is stateless, imports nothing
else from `fyp` (except other helpers), could be lifted into an unrelated
project unchanged, AND is used by both architectures. Anything that touches a
socket, a file, a model checkpoint or the robot belongs in the subsystem that
owns it.

Two modules left, and both departures are informative:

  ik.py         went to `simulation/fyp_sim/` on 11 Aug 2026. It imported
                mujoco; the real UR5e solves IK in firmware via moveL.
  pixel_to_3d.py went to `architecture_a/perception/`. It is pure maths and
                passes every purity test, but only Architecture A back-projects
                pixels into 3D. SmolVLA consumes raw images and never converts
                a pixel to a point, so this was never shared.

`rotations.py` and `transforms.py` genuinely are shared: A uses them for the
hand-eye transform, B uses them in `lerobot_export` to build the action space.
"""

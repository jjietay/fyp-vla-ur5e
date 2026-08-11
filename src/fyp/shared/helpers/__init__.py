"""Pure, stateless maths and configuration used by both architectures.

ADMISSION RULE - a module belongs here only if it is stateless, imports nothing
else from `fyp` (except other helpers), could be lifted into an unrelated
project unchanged, AND is used by both architectures. Anything that touches a
socket, a file, a model checkpoint or the robot belongs in the subsystem that
owns it.

`pixel_to_3d.py` lives in `architecture_a/perception/` rather than here. It is
pure maths and passes every purity test, but only Architecture A back-projects
pixels into 3D: SmolVLA consumes raw images and never converts a pixel to a
point, so it was never shared.

`rotations.py` and `transforms.py` genuinely are shared: A uses them for the
hand-eye transform, B uses them in `lerobot_export` to build the action space.
"""

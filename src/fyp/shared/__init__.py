"""Everything both architectures use, identically.

This package IS the controlled variable of the experiment. Both A and B speak
to the robot through `hardware/`, receive commands through the same speech
front end, and share the frame maths in `helpers/`.

Nothing here may branch on which architecture is calling it. A conditional in
this package is a bug in the experiment, not just in the code.
"""

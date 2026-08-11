"""Stage 4: camera frame to robot base frame.

Architecture A only. Architecture B never converts a pixel into a 3D point, so
it has no use for `T_base_cam`.

Eye-to-hand setup: the fixed camera is bolted to the world looking at the
table, so the unknown is one fixed transform that never changes unless the
camera is knocked. Solved once by Kabsch, saved to `config/calibration/`, and
reused by every later run.

A calibration always produces a number. `report()` is what tells you whether to
believe it, so never save one without recording its residual RMS.
"""

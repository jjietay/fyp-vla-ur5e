"""Stage 4: camera frame to robot base frame.

Architecture A only. Architecture B never converts a pixel into a 3D point, so
it has no use for `T_base_cam`.

Eye-to-hand setup: the camera stands still in the world looking at the table,
so the unknown is one transform. The lab is shared and the camera is taped in
place per session, so it is re-measured each session with
`scripts/write_calibration.py` and saved to `config/calibration/`. No fiducial
marker. The Kabsch solver here is the fallback, used with a ball held in the
gripper if the measured transform is too coarse.

A calibration always produces a number. `scripts/check_calibration.py` is what
tells you whether to believe a measured one, and `report()` does the same for a
fitted one.
"""

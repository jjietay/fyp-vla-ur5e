"""Stage 4: camera frame to robot base frame.

Architecture A only. Architecture B never converts a pixel into a 3D point, so
it has no use for `T_base_cam`.

Eye-to-hand setup: the camera stands still in the world looking at the table,
so the unknown is one transform. The lab is shared and the camera is taped in
place per session, so the transform is redone each session and saved to
`config/calibration/`. No fiducial marker.

The session routine (13 Oct 2026):
  1. `scripts/calibrate_ball.py`, a ball held in the gripper and fitted with the
     Kabsch solver here, with outlier rejection (`report()` gives its RMS)
  2. `scripts/level_to_table.py`, which removes the height and tilt bias a fit's
     RMS can hide, using the known table height
  3. `scripts/place_and_look.py`, which measures the real pick-time miss at
     known table spots and removes the mean of it

`scripts/write_calibration.py`, a transform measured by hand from the mount, is
the fallback when the arm cannot run the fit. `scripts/check_calibration.py`
moves the arm to detected points as a quick visual check.
"""

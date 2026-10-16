---
status: accepted, revised 13 Oct 2026
---

# No calibration marker: T_base_cam is fitted each session with a ball held in the gripper

First decided 12 Aug 2026 as "no marker, measure the mount by hand". Revised 13 Oct 2026: still no marker, but the transform is now fitted each session with a ball. The first reasoning is kept below as the record of the original choice.

**Why no marker.** Prof Cheah's view is that a printed board in the cell looks out of place against a real deployment, while the project argues for an interface a person could actually use. This still holds.

## Current procedure, since 13 Oct 2026

On 1 Oct the hand-measured transform needed a correction of (+22, +62, +23) mm, about 70 mm, before the orange could be grasped. That is well past the 1 to 3 cm expected. The ball fit is now the per session procedure, in `scripts/calibrate_ball.py`.

- it carries a matte ball to 16 poses at two heights, fits with `solve_rigid_transform`, drops one wrong detection if there is one, and reports the RMS
- it costs one to two minutes a session instead of an afternoon, so the original cost argument against it no longer holds
- it still uses no marker, so the objection to a printed board in the cell still holds
- its blind spot is the ball offset in the gripper. Every pose has the same tool orientation, so an offset error is the same shift at every point and the fit absorbs it. The reach test (`check_calibration.py`) checks that once, and again if the ball or its placement changes
- measuring the mount by hand (`write_calibration.py`) is now the fallback

## The original choice, 12 Aug 2026

The fixed camera stands at a measured offset from the arm base, so `T_base_cam` is one transform. Rather than recover it by observing a target, it was written directly from the measured position into `config/calibration/T_base_cam.json`. The lab is shared, so the camera is taped in place and taken down between sessions, and the transform was re-measured at the start of each session. Nothing in the pipeline depends on the method, because `hand_eye.py` loads the transform from JSON and never cares how the numbers got there.

### The cost of measuring by hand

`T_base_cam` is six numbers, and the rotation is the exposed one. At a 1 m standoff, **1 degree of camera orientation error displaces every back projected point by about 17 mm**. Holding a 5 mm target would need the mount known to better than 0.3 degrees, which a bracket and a rule will not give. So expect a systematic offset of roughly 1 to 3 cm, in one consistent direction rather than randomly per object.

A hand written transform has no residual of its own, so the reach test against known points is the only number there is. Measure to the camera's **optical origin**, not the case front, since the colour sensor sits a few millimetres inside the housing and a datum error is pure bias; and re-measure after any knock, because there is no fit to notice the camera moved.

The ball fit was kept as rung two at first, because the measured transform might have been good enough and cost an hour instead of an afternoon. The 1 Oct result showed it was not.

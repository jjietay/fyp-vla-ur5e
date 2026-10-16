---
status: accepted, revised 13 Oct 2026
---

# No calibration marker: T_base_cam is fitted each session with a ball held in the gripper

No printed marker, because Prof Cheah judged a board in the cell out of place for an interface meant to be usable. First decided 12 Aug 2026 as a hand-measured mount, which predicted a 1 to 3 cm bias (1 degree of mount error is about 17 mm at 1 m). On 1 Oct it needed a 70 mm hand correction, so since 13 Oct `scripts/calibrate_ball.py` fits the transform from 16 ball poses each session in one to two minutes, and hand measurement is the fallback.

## Consequences

Every pose shares one tool orientation, so a wrong `calibration.ball_offset_tool_m` is the same shift at every point and the fit's RMS cannot see it. The reach test (`check_calibration.py`) checks it once, and again when the ball or its placement changes.

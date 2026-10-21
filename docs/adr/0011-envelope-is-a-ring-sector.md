---
status: accepted
---

# The envelope is a ring sector around the base, plus a height range

Decided 21 Oct 2026, after the arm was moved away from the table. The arm's reach is round and the table lies in an arc in front of it, so a box either left half its area off the table or cut the reach short. `WorkspaceEnvelope` now takes `min_radius`, `max_radius` and `angle_deg` (the bearing `atan2(y, x)` of the point, not the base joint angle); the old `x`/`y` box stays as a loose backstop, and both new keys are optional so older configs behave as before. One class checks both architectures, so the comparison stays fair.

## Consequences

The bearing limit is also what keeps the arm off the camera, which stands inside the reach radius on the robot's left (-25 deg edge, checked by eye at 0.70 m and z 0.38). Moving the camera means checking that edge again. `max_radius` stays 0.70: it is set by the approach pose at the top of the z range, not by the 0.867 m the straight arm reaches at the table.

A seen object is judged by the sector and the floor only, not the z ceiling, because an object held in the gripper reads above it; every pose a skill commands still meets the full envelope.

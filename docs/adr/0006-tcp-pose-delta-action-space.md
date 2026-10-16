# Architecture B acts in TCP pose deltas: seven numbers, base frame translation, rotation delta, absolute gripper

Frame decided 31 Aug 2026. Rotation format decided 23 Sep 2026.

An earlier lean towards joint positions predated `skills.py`. Architecture A's primitives command TCP poses through `move_to_pose`, and the deciding constraint is that the action space must match what A commands, or the two architectures are not solving the same problem and the comparison is unfair in a way a careful examiner will spot.

## Why TCP poses

- Architecture A commands them, so both architectures speak one language
- the policy learns the task only, because the controller's IK supplies the kinematics. In joint space it would have to learn both, and would underperform for a reason that is not architectural
- generalisation across placement is cheaper, because an object 5 cm further right means roughly 5 cm more action wherever it sits
- `WorkspaceEnvelope` bounds x, y and z, so the safety check is direct and testable with no arm attached
- the detector, the depth back projection and the hand-eye transform all already produce base frame positions, and calibration error is reported in millimetres of hand position

## What it costs

Inverse kinematics is in the loop, so near a singularity a small hand movement demands very large joint speeds. Keep the workspace away from full extension and from directly over the base. One hand pose also has several joint solutions, so the arm can reconfigure at a branch boundary.

## Rotation format, settled 23 Sep 2026

The down pose has a rotation vector of magnitude pi (`down_rotvec` is `[0.580, 3.087, 0.0]`). That is the boundary of **both** rotation formats in use. A turn of pi about an axis equals a turn of pi about the opposite axis, so the rotation vector can flip sign, and converting to Euler gives roll of pi, which is the wrap point, so roll can flip too. Same physical pose, opposite numbers.

`scripts/check_orientation.py` on the lab arm on 23 Sep 2026 showed roll flipping between -pi and +pi. So `observation.state` stores rotation as the first two columns of the rotation matrix: ten numbers, being x, y, z, six matrix entries and the gripper. They change smoothly everywhere, and the third column is their cross product, so nothing is lost. A test puts two poses either side of the flip and checks the state barely changes.

The action is unchanged: seven numbers with an Euler rotation delta. Action deltas were always immune, because they are computed through rotation matrices and a delta near zero is far from any wrap point. No episode had been recorded, so no data was lost.

## Also settled

Architecture B's per step ceiling is DERIVED as `workspace.max_speed * robot.servo.time` rather than written down. It had been a literal 0.05 m, which at 20 Hz is 1.0 m/s against Architecture A's 0.25 m/s ceiling. Both architectures are constrained identically again, and a test asserts it.

Still to lock before recording: the observation set of two RGB views plus the state vector, the resolution and control rate, and the per episode instruction string drawn from the paraphrase set rather than a fixed template.

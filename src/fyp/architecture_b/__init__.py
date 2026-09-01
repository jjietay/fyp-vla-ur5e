"""Architecture B: end-to-end SmolVLA. Nothing in here is hand-authored.

    demos/      teleoperated demonstrations -> HDF5 -> LeRobot dataset

Training and inference are deliberately NOT in this package.

Training is a LeRobot command, `lerobot-train`, run from the lerobot environment.
There is nothing to wrap: writing a train.py here would add a layer that could
only get out of step with the tool doing the work. See `docs/commands.md`.

Inference is LeRobot's own `policy_server` and `robot_client`, driving a UR5e
adapter that lives at `integrations/lerobot_robot_ur5e/`. It is outside this
package because it must be installed into the lerobot environment, which is the
only one holding torch, the arm driver and the camera driver at once. Reasoning
in `docs/plan_vault/Decisions to Make.md`.

What that leaves here is the demonstration path, and the format contract with it.
`demos/lerobot_export.py` owns `STATE_NAMES` and `ACTION_NAMES`, plus
`compute_delta_action` and its exact inverse `apply_delta_action`. The adapter
imports all four rather than restating them, so training and inference cannot
disagree about what the seven numbers mean.

Behaviour changes only by collecting more data and retraining. There is no
prompt to edit and no primitive to tune, which is precisely the tradeoff being
measured against Architecture A.

The policy maps images, robot state and an instruction STRING to TCP pose
deltas: seven numbers, being a base-frame translation delta, an Euler rotation
delta, and an absolute gripper target. Not joint positions. That choice is
inherited from `demos/lerobot_export.py` and is still open in
`docs/plan_vault/Decisions to Make.md`, so if it changes, the converter, the
adapter and both NAMES tuples change together and the dataset is re-exported.
It has no output channel other than actions, so it cannot ask the user a
question. That limitation is a headline result, not a gap to be patched.

The safety clamp still applies, and it applies in the adapter's `send_action`,
using the same `WorkspaceEnvelope` from `shared/` that Architecture A uses. Two
different clamps would mean the two architectures were constrained differently,
which breaks the comparison more quietly than it breaks the arm.
"""

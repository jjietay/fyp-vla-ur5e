# Architecture B runs on LeRobot's own policy_server and robot_client, with only a UR5e adapter added

Decided 31 Aug 2026.

Two processes, both LeRobot's own. `policy_server` holds the checkpoint and does the forward pass. `robot_client` holds the action queue and drives the robot. They talk over gRPC. We supply only the missing piece, a UR5e adapter conforming to LeRobot's `Robot` interface, as a separate installable package at `integrations/lerobot_robot_ur5e/`. Nothing in LeRobot is forked or patched.

## Why not write our own loop

The action queue is the part that matters, and it lives in `robot_client`. It keeps a buffer of predicted actions, executes from it, requests a fresh chunk before it empties, and merges the overlap. Without it the arm stops dead every time the policy thinks.

The argument against writing it ourselves is not effort, it is the comparison. If B underperforms A, the first question an examiner asks is whether the inference loop was correct. Using the reference implementation removes that question; a hand written queue keeps it open and would have to be defended with timing measurements there is no time to collect.

## What it costs

The client process needs `lerobot`, `torch`, `ur_rtde` and `pyrealsense2` together, so Architecture B runs entirely out of `~/lerobot/.venv` with the two hardware drivers added. The FYP venv stays free of torch and keeps running Architecture A. Splitting the processes does **not** remove the need for `ur_rtde` in the lerobot environment, because the client is the half that touches hardware.

## The two contracts

Both read out of the LeRobot 0.6.1 source rather than the documentation.

**State names.** LeRobot builds `observation.state` by looking each dataset feature name up in the dict the robot returns. So the adapter's observation keys are literally `x`, `y`, `z`, `r00`, `r10`, `r20`, `r01`, `r11`, `r21`, `gripper`, and the camera key is `top`. `rIJ` is row I, column J of the rotation matrix. Euler `roll`, `pitch`, `yaw` were dropped on 23 Sep 2026, see ADR 0006.

**Action order.** `robot_client` unpacks the policy's action tensor by the **order** of `action_features`, by index, with no names involved. A reordering produces no error and sends roll to z.

Both tuples live in `lerobot_export.py` as `STATE_NAMES` and `ACTION_NAMES`, and the adapter imports them. Retyping them is how the two halves would drift apart.

## Consequences

`URController` had only `move_to_pose` and `move_joints`, which block until the motion finishes. A chunked policy at 20 Hz needs a new target every 50 ms without waiting, so `servo_to_pose` and `servo_stop` were added on top of `servoL`, with the gains in `config.yaml` under `robot.servo`.

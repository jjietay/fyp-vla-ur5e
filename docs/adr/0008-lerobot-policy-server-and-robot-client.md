# Architecture B runs on LeRobot's own policy_server and robot_client, with only a UR5e adapter added

Decided 31 Aug 2026. The action queue in `robot_client` (buffer, early refill, overlap merge) is what keeps the arm from stopping each time the policy thinks. A hand-written queue would leave the examiner's first question open, whether B lost because its inference loop was wrong. So nothing in LeRobot is forked, and only the adapter at `integrations/lerobot_robot_ur5e/` is ours. It runs in `~/lerobot/.venv`, which needs `ur_rtde` and `pyrealsense2` too, so the FYP venv stays free of torch.

## Consequences

LeRobot matches state by key name and actions by index only, so a reorder sends roll to z with no error. The adapter therefore imports `STATE_NAMES` and `ACTION_NAMES` from `lerobot_export.py` and never retypes them. `servo_to_pose` was added because `moveL` blocks and a 20 Hz loop cannot wait.

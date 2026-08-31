"""config.py

It takes command line arguments and gives you the settings the UR5e adapter runs on.

Registered under the name "ur5e", so `robot_client` accepts `--robot.type=ur5e`.

DELIBERATELY ALMOST EMPTY. The arm IP, the workspace envelope, the camera
resolution, the servo gains, the per-step clamps, the gripper threshold and the
camera name all live in the FYP `config.yaml`, which is the single source of
truth for tunable values. Duplicating any of them here would give Architecture B
a second place to disagree with Architecture A, and the comparison would start
measuring configuration drift.

So the only field is which config.yaml to read.
"""

from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("ur5e")
@dataclass(kw_only=True)
class UR5eConfig(RobotConfig):
    # Path to the FYP config.yaml. None finds the one in the FYP repo, which is
    # correct whenever PYTHONPATH points at that repo's src.
    fyp_config: str | None = None

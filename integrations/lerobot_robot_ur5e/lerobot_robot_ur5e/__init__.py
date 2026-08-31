"""lerobot_robot_ur5e

It takes nothing and gives you a UR5e registered as a LeRobot robot type.

Importing this package is what makes `--robot.type=ur5e` work. LeRobot's
`register_third_party_plugins` imports it automatically because the distribution
name starts with `lerobot_robot_`, so nothing has to import it by hand.

Why this exists: Architecture B runs on LeRobot's `policy_server` and
`robot_client`, and `robot_client` drives a `Robot` subclass. The UR5e is not one
of LeRobot's supported robots, so this package supplies the missing adapter. The
alternative was writing our own client, which would mean writing our own action
queue, and a timing bug in that queue would be indistinguishable from a policy
that simply does not work. The comparison between A and B is the deliverable, so
the inference loop has to be the reference one.

Needs the FYP source on PYTHONPATH:

    PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src
"""
from lerobot_robot_ur5e.config import UR5eConfig
from lerobot_robot_ur5e.ur5e import UR5eRobot

__all__ = ["UR5eConfig", "UR5eRobot"]

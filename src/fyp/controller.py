import rtde_control
import rtde_receive
from fyp.config import load_config, DEFAULT_CONFIG_PATH
from pathlib import Path

'''
controller.py will perform the following:

- init connects via ur_rtde
- move_to_pose(pose, speed, acc) using moveL
- move_joints(q, speed, acc) using moveJ
- get_state() returns a dict: joint_pos, tcp_pose, gripper_state
- set_gripper(open: bool
- close()

'''

class URController:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH):
        self.cfg = load_config(path)
        ursim_ip = self.cfg["robot"]["host"]
        rtde_r = rtde_receive.RTDEReceiveInterface(ursim_ip)
        rtde_c = rtde_control.RTDEControlInterface(ursim_ip)



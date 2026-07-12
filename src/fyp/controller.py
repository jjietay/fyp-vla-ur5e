import rtde_control
import rtde_receive
import rtde_io
import time
from fyp.config import get_config, DEFAULT_CONFIG_PATH
from pathlib import Path

'''
controller.py will perform the following:

- init connects via ur_rtde
- move_to_pose(pose, speed, acc) using moveL
- move_joints(q, speed, acc) using moveJ
- get_state() returns a dict: joint_pos, tcp_pose, gripper_state
- gripper_start(default pin_power = 1, default pin_control = 2)
- gripper_toggle(state = 0 or 1, where 0 is closed and 1 is open)
- close()

'''

class URController:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH, ):
        self.cfg = get_config(path)
        ursim_ip = self.cfg["robot"]["host"]
        self.default_speed = self.cfg["robot"]["motion"]["default_speed"]
        self.default_acc = self.cfg["robot"]["motion"]["default_acc"]
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ursim_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(ursim_ip)
        self.rtde_io = rtde_io.RTDEIOInterface(ursim_ip)
        self.pin_power = None
        self.pin_control = None
    
    def gripper_start(self, pin_power : int | None = None, pin_control : int | None = None):
        """Gripper digital-output pins default from config (robot.gripper)."""
        
        grip = self.cfg["robot"]["gripper"]
        self.pin_power = pin_power if pin_power is not None else grip["power_pin"]
        self.pin_control = pin_control if pin_control is not None else grip["control_pin"]
        self.rtde_io.setStandardDigitalOut(self.pin_power, True)
        time.sleep(0.1)
        if self.rtde_r.getDigitalOutState(self.pin_power):
            return "Gripper initialized."
        else:
            print("Gripper unable to turn on.")

    def move_to_pose(self, pose, speed : float | None = None, acc : float | None = None):
        speed = speed if speed is not None else self.default_speed
        acc = acc if acc is not None else self.default_acc
        return self.rtde_c.moveL(pose, speed, acc)

    def move_joints(self, q, speed : float | None = None, acc : float | None = None):
        speed = speed if speed is not None else self.default_speed
        acc = acc if acc is not None else self.default_acc
        return self.rtde_c.moveJ(q, speed, acc)
    
    def gripper_toggle(self, state: int):
        if state == 1:      # OPEN
            self.rtde_io.setStandardDigitalOut(self.pin_control, False)
            time.sleep(0.1)
        elif state == 0:    # CLOSE
            self.rtde_io.setStandardDigitalOut(self.pin_control, True)
            time.sleep(0.1)
        else:
            return "Unable to control gripper."

    def get_state(self):
        return {
        "joint_pos": self.rtde_r.getActualQ(),
        "tcp_pose": self.rtde_r.getActualTCPPose(),
        "gripper_state": 0 if self.rtde_r.getDigitalOutState(self.pin_control) else 1
        }
    
    def close(self):
        if self.pin_power is not None:
            self.rtde_io.setStandardDigitalOut(self.pin_power, False)
        self.rtde_c.stopScript()

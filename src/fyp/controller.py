import rtde_control
import rtde_receive
import rtde_io
import time
from fyp.config import load_config, DEFAULT_CONFIG_PATH
from pathlib import Path

'''
controller.py will perform the following:

- init connects via ur_rtde
- move_to_pose(pose, speed, acc) using moveL
- move_joints(q, speed, acc) using moveJ
- get_state() returns a dict: joint_pos, tcp_pose, gripper_state
- gripper_start(default pin_power = 1, default pin_control = 2)
- gripper_toggle(state = open/close)
- close()

'''

class URController:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH, ):
        self.cfg = load_config(path)
        ursim_ip = self.cfg["robot"]["host"]
        self.default_speed = self.cfg["motion"]["default_speed"]
        self.default_acc = self.cfg["motion"]["default_acc"]
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ursim_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(ursim_ip)
        self.rtde_io = rtde_io.RTDEIOInterface(ursim_ip)
        self.pin_power = None
        self.pin_control = None
    
    def gripper_start(self, pin_power : int | None = 1, pin_control : int | None = 2):
        """Assuming the gripper is default connected to digital output pin 1 for power,
        and digital output pin 2 for control"""
        
        self.pin_power = pin_power if pin_power is not None else 1
        self.pin_control = pin_control if pin_control is not None else 2
        self.rtde_io.setStandardDigitalOut(pin_power, True)
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
    
    def gripper_toggle(self, state: str):
        if state == "open":
            self.rtde_io.setStandardDigitalOut(self.pin_control, False)
            time.sleep(0.1)
        elif state == "close":
            self.rtde_io.setStandardDigitalOut(self.pin_control, True)
            time.sleep(0.1)
        else:
            return "Unable to control gripper."

    def get_state(self):
        return {
        "joint_pos": self.rtde_r.getActualQ(),
        "tcp_pose": self.rtde_r.getActualTCPPose(),
        "gripper_state": "close" if self.rtde_r.getDigitalOutState(self.pin_control) else "open"
        }
    
    def close(self):
        if self.pin_power is not None:
            self.rtde_io.setStandardDigitalOut(self.pin_power, False)
        self.rtde_c.stopScript()

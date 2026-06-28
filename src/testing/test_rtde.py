import rtde_control
import rtde_receive
import math

URSIM_IP = "127.0.0.1"

def degtorad(degrees: list) -> list:
    return [math.radians(d) for d in degrees]

rtde_c = rtde_control.RTDEControlInterface(URSIM_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(URSIM_IP)

home_q = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
pos_1_q = [0, -0.7222, 0.73967054, -1.5708, -1.5708, 0]

print(f"Current joints: {rtde_r.getActualQ()}")
print(f"Current TCP Pose: {rtde_r.getActualTCPPose()}")

rtde_c.moveJ(home_q, 1.0, 1.0)
rtde_c.moveJ(pos_1_q, 1.0, 1.0)

print(f"Now at joints: {rtde_r.getActualQ()}")
print(f"Current TCP Pose: {rtde_r.getActualTCPPose()}")

rtde_c.stopScript()

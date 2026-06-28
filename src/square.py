import rtde_control
import rtde_receive
import math

URSIM_IP = "127.0.0.1"

def degtorad(degrees: list) -> list:
    return [math.radians(d) for d in degrees]

rtde_c = rtde_control.RTDEControlInterface(URSIM_IP)
rtde_r = rtde_control.RTDEReceiveInterface(URSIM_IP)

transform_1 = [[]]
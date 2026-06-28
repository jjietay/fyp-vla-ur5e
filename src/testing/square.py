import rtde_control
import rtde_receive
import math
import numpy as np

URSIM_IP = "127.0.0.1"

def degtorad(degrees: list) -> list:
    return [math.radians(d) for d in degrees]

rtde_c = rtde_control.RTDEControlInterface(URSIM_IP)
rtde_r = rtde_control.RTDEReceiveInterface(URSIM_IP)

kx, ky, kz = k

k = np.array([
    [0, -kz, ky]
    [kz, 0, -kx]
    [-ky, kx, 0]
])

np.eye(3)    # 3×3 identity
np.eye(4)    # 4×4 identity
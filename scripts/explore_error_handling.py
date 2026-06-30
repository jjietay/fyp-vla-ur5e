import sys
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "127.0.0.1"   # localhost for URSim native

def connect(ip: str, timeout_msg: bool = True):
    """Connect to UR controller, return (rtde_c, rtde_r) or exit cleanly."""
    try:
        rtde_c = RTDEControlInterface(ip)
        rtde_r = RTDEReceiveInterface(ip)
    except RuntimeError as e:
        print(f"[connect] failed to reach {ip}: {e}", file=sys.stderr)
        print("  - Is URSim running?", file=sys.stderr)
        print("  - Is the robot initialised (power on + brake release)?", file=sys.stderr)
        print("  - Is another script holding the connection?", file=sys.stderr)
        sys.exit(1)
    return rtde_c, rtde_r


if __name__ == "__main__":
    rtde_c, rtde_r = connect(ROBOT_IP)
    try:
        # robot moves
        print("TCP pose:", rtde_r.getActualTCPPose())
    finally:
        rtde_c.stopScript()   # cleanup at the end
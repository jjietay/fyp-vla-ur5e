import socket
import math

URSIM_IP = "127.0.0.1"
URSIM_PORT = 30002

def send_urscript(script: str) -> None:
    """Open a TCP socket, send a URScript program, then close."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((URSIM_IP, URSIM_PORT))
        s.sendall(script.encode("utf-8"))
    print("Script sent.")

def degtorad(degrees: list) -> list:
    return [math.radians(d) for d in degrees]


home_q  = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]         # q represents joint positions and pose represent tcp
pos_1_q = [0, -0.7222, 0.73967054, -1.5708, -1.5708, 0]     # already manually converted from degrees to radians
pos_2_q = [0.0, -0.57735, 1.01735, -2.33717, -1.5708, 0]    # same here, already converted to radians
pos_3_q = degtorad([32.48, -55.73, 67.19, -100.62, -89.46, 32.48])
pos_4_q = degtorad([31.48, -42.03, 80.28, -127.41, -89.46, 32.48])


program = f"""\
def motion_test():
    # --- Starting Configuration ---
    movej({home_q}, a=1.0, v=1.0) 

    # --- Move down and get ready to reach ---
    movej({pos_1_q}, a=1.0, v=1.0)

    # --- Go straight down and reach ---
    movel({pos_2_q}, a = 0.5, v=0.25)

    # --- Go back up from picking up ---
    movel({pos_1_q}, a = 0.5, v=0.25)

    # --- Move to another spot ---
    movej({pos_3_q}, a=1.0, v=1.0)

    # --- Move down slowly ---
    movel({pos_4_q}, a = 0.5, v=0.25)

    # --- Move back up ---
    movel({pos_3_q}, a = 0.5, v=0.25)

    # --- Move back to first spot ---
    movej({pos_1_q}, a=1.0, v=1.0)
end
"""

if __name__ == "__main__":
    print(f"Connecting to URSim at {URSIM_IP}:{URSIM_PORT}...")
    print(program)
    send_urscript(program)

    print("Waiting for motion to complete...")
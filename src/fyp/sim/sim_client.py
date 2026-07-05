"""Client for the MuJoCo sim server. Use interactively from a second terminal.

Example (in a REPL):
    from fyp.sim.sim_client import SimClient
    c = SimClient()
    c.get_state()
    c.move_joints([-1.0, -1.5, 1.5, -1.5, -1.5, 0.0], speed=1.0)
    c.gripper_toggle(0)   # close
    c.home()
"""

from __future__ import annotations

import json
import socket

HOST = "127.0.0.1"
PORT = 5555


class SimClient:
    def __init__(self, host: str = HOST, port: int = PORT):
        self.host = host
        self.port = port

    def _send(self, request: dict) -> dict:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            s.sendall((json.dumps(request) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            line = buf.split(b"\n", 1)[0]
            return json.loads(line.decode())

    # ---- commands ---------------------------------------------------------

    def get_state(self) -> dict:
        return self._send({"cmd": "get_state"})

    def move_joints(self, q, speed: float | None = None) -> dict:
        return self._send({"cmd": "move_joints", "q": list(q), "speed": speed})

    def move_to_pose(self, pose, speed: float | None = None) -> dict:
        return self._send({"cmd": "move_to_pose", "pose": list(pose), "speed": speed})

    def gripper_toggle(self, state: int) -> dict:
        return self._send({"cmd": "gripper_toggle", "state": int(state)})

    def home(self) -> dict:
        return self._send({"cmd": "home"})
    
    def start_recording(self) -> dict:
        return self._send({"cmd": "start_recording"})

    def stop_and_save(self, path: str) -> dict:
        return self._send({"cmd": "stop_and_save", "path": path})
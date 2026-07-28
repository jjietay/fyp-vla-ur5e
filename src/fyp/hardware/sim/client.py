"""Client for the MuJoCo sim server. Use interactively from a second terminal.

Holds no simulation state at all: it sends a newline-delimited JSON request over
a TCP socket and returns whatever the server reports back.

Example (in a REPL):
    from fyp.hardware.sim.client import SimClient
    c = SimClient()
    c.get_state()
    c.move_joints([-1.0, -1.5, 1.5, -1.5, -1.5, 0.0], speed=1.0)
    c.gripper_toggle(0)   # close
    c.home()
"""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

from fyp.helpers.config import get_config, resolve

_srv = get_config()["server"]
HOST = _srv["host"]
PORT = _srv["port"]


def next_episode_path(
    episodes_dir: str | Path | None = None,
    prefix: str = "ep_",
    digits: int = 3,
) -> str:
    """Return the next free '<prefix>NNN.h5' path in the episodes dir.

    Scans the folder for existing '<prefix>NNN.h5' files, takes the highest N,
    and returns N+1 (zero-padded). Starts at 1 when the folder is empty. This
    guarantees a new recording never overwrites an existing episode.
    """
    d = Path(episodes_dir) if episodes_dir else resolve(get_config()["paths"]["episodes_dir"])
    d.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.h5$")
    nums = [int(m.group(1)) for f in d.glob(f"{prefix}*.h5") if (m := pat.match(f.name))]
    n = max(nums) + 1 if nums else 1
    return str(d / f"{prefix}{n:0{digits}d}.h5")


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

    def stop_and_save(self, path: str | None = None) -> dict:
        """Stop recording and save.

        path=None (default): auto-pick the next free ep_NNN.h5 so each new
        recording gets a fresh name and never overwrites a previous episode.
        Pass an explicit path to override (e.g. a descriptive task name).
        """
        if path is None:
            path = next_episode_path()
        return self._send({"cmd": "stop_and_save", "path": path})

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from fyp.sim.mujoco_controller import URControllerMuJoCo
from fyp.sim.ik import solve_ik

HOST = "127.0.0.1"
PORT = 5555
SCENE = Path("assets/mujoco/ur5e/scene.xml")


class _Job:
    def __init__(self, request: dict):
        self.request = request
        self.result: dict | None = None
        self.done = threading.Event()


class SimServer:
    def __init__(self, scene_path: Path = SCENE, host: str = HOST, port: int = PORT):
        self.ctrl = URControllerMuJoCo(scene_path, default_speed=1.0)
        self.host = host
        self.port = port
        self._jobs: "queue.Queue[_Job]" = queue.Queue()
        self._stop = threading.Event()

    def _execute(self, req: dict, viewer) -> dict:
        cmd = req.get("cmd")
        try:
            if cmd == "get_state":
                return {"ok": True, "state": self.ctrl.get_state()}
            elif cmd == "move_joints":
                q = np.asarray(req["q"], dtype=float)
                self._interp_move(q, req.get("speed"), viewer)
                return {"ok": True, "state": self.ctrl.get_state()}
            elif cmd == "move_to_pose":
                pose = np.asarray(req["pose"], dtype=float)
                target_pos = pose[:3]
                rvec = pose[3:6]
                angle = float(np.linalg.norm(rvec))
                if angle < 1e-8:
                    target_mat = np.eye(3)
                else:
                    axis = rvec / angle
                    quat = np.zeros(4)
                    mujoco.mju_axisAngle2Quat(quat, axis, angle)
                    mat_flat = np.zeros(9)
                    mujoco.mju_quat2Mat(mat_flat, quat)
                    target_mat = mat_flat.reshape(3, 3)
                q_sol, ok = solve_ik(
                    self.ctrl.model, self.ctrl.data, self.ctrl._tcp_site_id,
                    target_pos, target_mat,
                    q_init=self.ctrl.data.qpos[:6].copy(),
                )
                if not ok:
                    return {"ok": False, "error": "IK did not converge."}
                self._interp_move(q_sol, req.get("speed"), viewer)
                return {"ok": True, "state": self.ctrl.get_state()}
            elif cmd == "gripper_toggle":
                self.ctrl.gripper_toggle(int(req["state"]))
                return {"ok": True, "state": self.ctrl.get_state()}
            elif cmd == "home":
                mujoco.mj_resetDataKeyframe(self.ctrl.model, self.ctrl.data, 0)
                self.ctrl.data.ctrl[:6] = self.ctrl.data.qpos[:6]
                mujoco.mj_forward(self.ctrl.model, self.ctrl.data)
                return {"ok": True, "state": self.ctrl.get_state()}
            else:
                return {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _interp_move(self, q_target, speed, viewer) -> None:
        speed = speed if speed is not None else self.ctrl.default_speed
        q_start = self.ctrl.data.qpos[:6].copy()
        delta = q_target - q_start
        max_move = float(np.max(np.abs(delta)))
        if max_move < 1e-6:
            return
        duration = max_move / speed
        n_steps = max(int(np.ceil(duration / self.ctrl.control_dt)), 1)
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            self.ctrl.data.ctrl[:6] = q_start + alpha * delta
            mujoco.mj_step(self.ctrl.model, self.ctrl.data)
            viewer.sync()
            time.sleep(self.ctrl.control_dt)
        self.ctrl.data.ctrl[:6] = q_target

    def _serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        srv.settimeout(0.5)
        print(f"[server] listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                buf = b""
                while not self._stop.is_set():
                    try:
                        conn.settimeout(0.5)
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        req = json.loads(line.decode())
                        job = _Job(req)
                        self._jobs.put(job)
                        job.done.wait()
                        conn.sendall((json.dumps(job.result) + "\n").encode())
        srv.close()

    def run(self) -> None:
        listener = threading.Thread(target=self._serve, daemon=True)
        listener.start()
        with mujoco.viewer.launch_passive(self.ctrl.model, self.ctrl.data) as viewer:
            print("[server] viewer open - ready for commands.")
            while viewer.is_running():
                try:
                    while True:
                        job = self._jobs.get_nowait()
                        job.result = self._execute(job.request, viewer)
                        job.done.set()
                except queue.Empty:
                    pass
                mujoco.mj_step(self.ctrl.model, self.ctrl.data)
                viewer.sync()
                time.sleep(self.ctrl.control_dt)
        self._stop.set()
        print("[server] viewer closed - shutting down.")


def main() -> None:
    SimServer().run()


if __name__ == "__main__":
    main()

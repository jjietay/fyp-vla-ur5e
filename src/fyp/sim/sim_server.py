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
from fyp.sim.demo_recorder import DemoRecorder

HOST = "127.0.0.1"
PORT = 5555
SCENE = Path("assets/mujoco/ur5e/scene_gripper.xml")

CAM_NAME = "fixed_cam"
IMG_W, IMG_H = 320, 240
RECORD_EVERY = 25          # every 25th tick at 500Hz control_dt -> ~20Hz
GRIPPER_ACT_IDX = 6
GRIPPER_MIDPOINT = 127.5   # ctrl[6] below -> open(1), above -> closed(0)


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

        # Recording state
        self.recorder = DemoRecorder()
        self.renderer = mujoco.Renderer(self.ctrl.model, height=IMG_H, width=IMG_W)
        self._recording = False
        self._tick = 0

    # ---- gripper state from the live slider command ------------------------

    def _gripper_from_ctrl(self) -> int:
        """Derive 0/1 gripper state from the actual actuator command."""
        if self.ctrl.model.nu <= GRIPPER_ACT_IDX:
            return self.ctrl._gripper_state
        return 1 if self.ctrl.data.ctrl[GRIPPER_ACT_IDX] < GRIPPER_MIDPOINT else 0

    # ---- command dispatch (runs on MAIN thread) ---------------------------

    def _execute(self, req: dict, viewer) -> dict:
        cmd = req.get("cmd")
        try:
            if cmd == "get_state":
                st = self.ctrl.get_state()
                st["gripper_state"] = self._gripper_from_ctrl()
                return {"ok": True, "state": st}

            elif cmd == "start_recording":
                self.recorder.start_episode()
                self._recording = True
                self._tick = 0
                return {"ok": True, "recording": True}

            elif cmd == "stop_and_save":
                if not self._recording:
                    return {"ok": False, "error": "not recording"}
                self._recording = False
                path = req.get("path", "data/episode.h5")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                n = len(self.recorder._buffer)
                if n == 0:
                    return {"ok": False, "error": "no frames recorded"}
                self.recorder.save_episode(path)
                return {"ok": True, "path": path, "frames": n}

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
                if self.ctrl._has_gripper:
                    self.ctrl.data.ctrl[GRIPPER_ACT_IDX] = 0.0
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
            self._maybe_record()
            viewer.sync()
            time.sleep(self.ctrl.control_dt)
        self.ctrl.data.ctrl[:6] = q_target

    # ---- recording ---------------------------------------------------------

    def _maybe_record(self) -> None:
        """Called every sim tick; logs a snapshot every RECORD_EVERY ticks."""
        if not self._recording:
            return
        self._tick += 1
        if self._tick % RECORD_EVERY != 0:
            return
        st = self.ctrl.get_state()
        self.renderer.update_scene(self.ctrl.data, camera=CAM_NAME)
        img = self.renderer.render()
        self.recorder.record(
            joint_positions=np.asarray(st["joint_pos"]),
            tcp_pose=np.asarray(st["tcp_pose"]),
            gripper_state=self._gripper_from_ctrl(),
            image=img,
        )

    # ---- socket listener (BACKGROUND thread) ------------------------------

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

    # ---- main loop --------------------------------------------------------

    def run(self) -> None:
        listener = threading.Thread(target=self._serve, daemon=True)
        listener.start()
        try:
            with mujoco.viewer.launch_passive(self.ctrl.model, self.ctrl.data) as viewer:
                print("[server] viewer open - drag sliders to teleop. Ready for commands.")
                while viewer.is_running():
                    try:
                        while True:
                            job = self._jobs.get_nowait()
                            job.result = self._execute(job.request, viewer)
                            job.done.set()
                    except queue.Empty:
                        pass
                    mujoco.mj_step(self.ctrl.model, self.ctrl.data)
                    self._maybe_record()
                    viewer.sync()
                    time.sleep(self.ctrl.control_dt)
        finally:
            self._stop.set()
            self.renderer.close()   # close renderer before teardown (avoids free() crash)
            print("[server] shut down.")


def main() -> None:
    SimServer().run()


if __name__ == "__main__":
    main()
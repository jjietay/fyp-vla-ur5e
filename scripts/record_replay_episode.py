"""Sunday deliverable: drive UR5e in MuJoCo, record an episode, save, replay.

Records a scripted joint-space trajectory via DemoRecorder, saves to HDF5,
then reloads and replays: playback the camera images + plot the TCP trajectory.
"""

from pathlib import Path

import numpy as np
import h5py
import mujoco
import matplotlib.pyplot as plt

from fyp.sim.mujoco_controller import URControllerMuJoCo
from fyp.sim.demo_recorder import DemoRecorder, RateControl

SCENE = Path("assets/mujoco/ur5e/scene.xml")
OUT = Path("data/mujoco_episode.h5")
CAM_NAME = "fixed_cam"
IMG_W, IMG_H = 320, 240


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    renderer.update_scene(data, camera=CAM_NAME)
    return renderer.render()  # (H, W, 3) uint8


def waypoints() -> list[np.ndarray]:
    """A few joint configs to sweep through (radians)."""
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    return [
        home,
        home + np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0]),
        home + np.array([0.6, 0.3, -0.4, 0.0, 0.2, 0.0]),
        home + np.array([-0.4, 0.2, -0.2, 0.3, 0.0, 0.5]),
        home,
    ]


def main() -> None:
    ctrl = URControllerMuJoCo(SCENE, default_speed=1.0)
    recorder = DemoRecorder()
    renderer = mujoco.Renderer(ctrl.model, height=IMG_H, width=IMG_W)

    recorder.start_episode()

    # Drive through waypoints; record a snapshot after each sim step chunk.
    # We hook recording into the motion by stepping manually here.
    for wp in waypoints():
        q_start = ctrl.data.qpos[:6].copy()
        delta = wp - q_start
        max_move = float(np.max(np.abs(delta)))
        if max_move < 1e-6:
            continue
        duration = max_move / ctrl.default_speed
        n_steps = max(int(np.ceil(duration / ctrl.control_dt)), 1)

        record_every = max(int(round(0.05 / ctrl.control_dt)), 1)  # ~20 Hz

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            ctrl.data.ctrl[:6] = q_start + alpha * delta
            mujoco.mj_step(ctrl.model, ctrl.data)

            if i % record_every == 0:
                st = ctrl.get_state()
                img = render_frame(renderer, ctrl.data)
                recorder.record(
                    joint_positions=np.asarray(st["joint_pos"]),
                    tcp_pose=np.asarray(st["tcp_pose"]),
                    gripper_state=st["gripper_state"],
                    image=img,
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    recorder.save_episode(OUT)
    print(f"Saved episode to {OUT}")

    # ---- reload + replay ---------------------------------------------------
    with h5py.File(OUT, "r") as f:
        timestamps = f["timestamps"][:]
        tcp_poses = f["tcp_poses"][:]
        images = f["images"][:]
        n = len(timestamps)

    print(f"Reloaded {n} frames.")

# Replay 1: TCP trajectory plots
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(timestamps, tcp_poses[:, 0], label="x")
    ax1.plot(timestamps, tcp_poses[:, 1], label="y")
    ax1.plot(timestamps, tcp_poses[:, 2], label="z")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("TCP position (m)")
    ax1.set_title("TCP trajectory")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(tcp_poses[:, 0], tcp_poses[:, 1], tcp_poses[:, 2], marker=".")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("TCP path (3D)")

    plt.tight_layout()
    plt.savefig("data/tcp_trajectory.png", dpi=120)
    print("Saved plot to data/tcp_trajectory.png")

    # Replay 2: image playback as a GIF
    import matplotlib.animation as animation

    fig2, imax = plt.subplots()
    im = imax.imshow(images[0])
    imax.set_title("Episode playback")
    imax.axis("off")

    def update(frame_idx):
        im.set_data(images[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig2, update, frames=n, interval=50, blit=True
    )
    ani.save("data/episode_playback.gif", writer="pillow", fps=20)
    print("Saved playback to data/episode_playback.gif")


if __name__ == "__main__":
    main()
""" record_episode.py

This moves the UR5e in MuJoCo along a scripted trajectory, record it, save to HDF5 and
replay back the camera images and plot the TCP trajectory.

Note: Sim only + Scripted Path.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from fyp.helpers.config import get_config, resolve
from fyp.hardware.sim.mujoco_controller import URControllerMuJoCo
from fyp.demos.recorder import DemoRecorder
from fyp.demos.hdf5_store import load_episode

_sim = get_config()["sim"]
SCENE = resolve(_sim["scene_arm_only"])
OUT = resolve(get_config()["paths"]["episodes_dir"]) / "mujoco_episode.h5"
FIGURES = resolve(get_config()["paths"]["figures_dir"])
CAM_NAME = _sim["camera"]["name"]
IMG_W, IMG_H = _sim["camera"]["width"], _sim["camera"]["height"]


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    """
    This renders a frame.
    """
    renderer.update_scene(data, camera=CAM_NAME)
    return renderer.render()


def waypoints() -> list[np.ndarray]:
    """
    This creates an ndarray of 5 fixed waypoints, from home to point 1,2,3 and back to home.
    """
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    return [
        home,
        home + np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0]),
        home + np.array([0.6, 0.3, -0.4, 0.0, 0.2, 0.0]),
        home + np.array([-0.4, 0.2, -0.2, 0.3, 0.0, 0.5]),
        home,
    ]


def main() -> None:
    ctrl = URControllerMuJoCo(SCENE)
    recorder = DemoRecorder()
    renderer = mujoco.Renderer(ctrl.model, height=IMG_H, width=IMG_W)

    recorder.start_episode()


    record_every = max(round((1 / _sim["record_hz"]) / ctrl.control_dt), 1)
    step = 0


    for wp in waypoints():
        q_start = ctrl.data.qpos[:6].copy()
        delta = wp - q_start
        max_move = float(np.max(np.abs(delta)))
        if max_move < 1e-6:
            continue
        duration = max_move / ctrl.default_speed
        n_steps = max(int(np.ceil(duration / ctrl.control_dt)), 1)

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            ctrl.data.ctrl[:6] = q_start + alpha * delta
            ctrl.step()
            step += 1

            if step % record_every == 0:
                st = ctrl.get_state()
                img = render_frame(renderer, ctrl.data)
                recorder.record(
                    joint_positions=np.asarray(st["joint_pos"]),
                    tcp_pose=np.asarray(st["tcp_pose"]),
                    gripper_state=st["gripper_state"],
                    image=img,
                    timestamp=step * ctrl.control_dt,
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    recorder.save_episode(OUT)
    print(f"Saved episode to {OUT}")


    ep = load_episode(OUT)
    timestamps, tcp_poses, images = ep["timestamps"], ep["tcp_poses"], ep["images"]
    n = len(timestamps)
    print(f"Reloaded {n} frames.")


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
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(FIGURES / "tcp_trajectory.png"), dpi=120)
    print(f"Saved plot to {FIGURES / 'tcp_trajectory.png'}")


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
    ani.save(str(FIGURES / "episode_playback.gif"), writer="pillow",
             fps=_sim["record_hz"])
    print(f"Saved playback to {FIGURES / 'episode_playback.gif'}")


if __name__ == "__main__":
    main()

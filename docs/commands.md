## To run URSim
URSIM=~/Documents/NTU/Y4S1/FYP/ursim/URSim_Linux-5.25.2.130406/ursim-5.25.2.130406

cd "$URSIM"

./start-ursim.sh UR5e

## Unrestricted starting pose values
pose = [0.44489, -0.24078, -0.23421, 3.075, 0.679, -0.002]

## Start MuJoCo
XDG_SESSION_TYPE=x11 uv run python -c "
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path('assets/mujoco/ur5e/scene_gripper.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
mujoco.viewer.launch(m, d)
"

## Rebuilds scene_gripper.xml after changes to scene.xml
uv run python scripts/build_scene.py

## Opens MuJoCo Server
XDG_SESSION_TYPE=x11 uv run python -m fyp.hardware.sim.server

## Opens Client
uv run python
from fyp.hardware.sim.client import SimClient
c = SimClient()
c.start_recording()
c.stop_and_save()
exit()

## Inspect the MP4 Video
uv run python scripts/replay_episode.py data/episodes/ep_001.h5
xdg-open data/episodes/ep_001.mp4

## Save first and last frame for visual context
uv run python -c "
import h5py, numpy as np
import matplotlib.pyplot as plt
with h5py.File('data/episodes/ep_001.h5', 'r') as f:
    imgs = f['images'][:]; grip = f['gripper_states'][:]; tcp = f['tcp_poses'][:]
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(imgs[0]);  ax[0].set_title('frame 0');  ax[0].axis('off')
ax[1].imshow(imgs[-1]); ax[1].set_title('frame -1'); ax[1].axis('off')
plt.savefig('data/episodes/ep_001_check.png', dpi=100)
print('gripper transitions at frames:', np.where(np.diff(grip) != 0)[0])
print('TCP z range:', round(float(tcp[:,2].min()),3), '->', round(float(tcp[:,2].max()),3))
"

## Architecture A — render RGB + depth and verify against MuJoCo truth (FYP venv)
uv run python scripts/check_camera.py --camera workspace --verify

- writes data/frames/workspace_{rgb.png,depth.png,depth.npy}
- use the .npy for maths — the PNG is normalised for eyeballing only

## Architecture A — render a frame for the detector (FYP venv)
uv run python scripts/check_detector.py --render-only --camera workspace \
  --out data/frames/frame_top.png

## Architecture A — detect, with NMS + per-query top-1 (lerobot venv)
cd ~/lerobot
uv run python /home/jj/Documents/NTU/Y4S1/FYP/scripts/check_detector.py \
  --image /home/jj/Documents/NTU/Y4S1/FYP/data/frames/frame_top.png \
  --queries "red cube" "green cube" "blue cube" "yellow cube" "bin" \
  --model google/owlv2-base-patch16-ensemble --threshold 0.3 \
  --nms-iou 0.5 --top1-per-query \
  --out  /home/jj/Documents/NTU/Y4S1/FYP/data/frames/detections_owlv2.png \
  --json /home/jj/Documents/NTU/Y4S1/FYP/data/frames/detections.json

- NMS is class-agnostic by default: OWLv2 fires two labels on one cube and per-query suppression would keep both.
- Drop --top1-per-query once the scene can hold more than one instance of a query.

## Architecture A — depth-to-3D (FYP venv)
uv run python scripts/check_depth.py --verify
uv run python scripts/check_depth.py --detections data/frames/detections.json

- `--verify` isolates the geometry (no detector)
- `--detections` runs the real chain.
- Output is CAMERA frame; stage 4 (hand-eye) converts to the robot base frame.

## Lerobot Initial Sync
cd ~/lerobot && uv sync --extra dataset --extra training --extra smolvla


## Convert HDF5 episodes into a LeRobot Dataset
cd ~/lerobot
PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
uv run --extra dataset --with h5py python /home/jj/Documents/NTU/Y4S1/FYP/scripts/export_lerobot.py \
  --episodes /home/jj/Documents/NTU/Y4S1/FYP/data/episodes \
  --repo-id  jj/ur5e_pickplace \
  --task     "pick and place the block" \
  --camera   top \
  --root     /home/jj/Documents/NTU/Y4S1/FYP/data/lerobot_ur5e


## Train SmolVLA2 Base
  cd ~/lerobot && uv run --extra dataset --extra training lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=jj/ur5e_pickplace \
  --dataset.root=/home/jj/Documents/NTU/Y4S1/FYP/data/lerobot_ur5e \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=1 \
  --output_dir=outputs/train/dryrun_smolvla \
  --job_name=dryrun \
  --policy.device=cuda \
  --wandb.enable=false


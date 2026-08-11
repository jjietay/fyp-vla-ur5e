# Commands

Live commands only.

## URSim

Universal Robots' own controller simulator. It is what makes
`pytest tests` pass without the real arm on `127.0.0.1:30004`.

```bash
URSIM=~/Documents/NTU/Y4S1/FYP/simulation/ursim/URSim_Linux-5.25.2.130406/ursim-5.25.2.130406
cd "$URSIM" && ./start-ursim.sh UR5e
```

Unrestricted starting pose:

```
pose = [0.44489, -0.24078, -0.23421, 3.075, 0.679, -0.002]
```

## Tests

```bash
uv sync
uv run pytest tests
```

## Episodes

```bash
# HDF5 -> mp4
uv run python scripts/replay_episode.py data/raw/episodes/ep_001.h5
xdg-open data/raw/episodes/ep_001.mp4
```

First and last frame plus gripper transitions, for a quick sanity look:

```bash
uv run python -c "
import h5py, numpy as np
import matplotlib.pyplot as plt
with h5py.File('data/raw/episodes/ep_001.h5', 'r') as f:
    imgs = f['images'][:]; grip = f['gripper_states'][:]; tcp = f['tcp_poses'][:]
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(imgs[0]);  ax[0].set_title('frame 0');  ax[0].axis('off')
ax[1].imshow(imgs[-1]); ax[1].set_title('frame -1'); ax[1].axis('off')
plt.savefig('data/cache/ep_001_check.png', dpi=100)
print('gripper transitions at frames:', np.where(np.diff(grip) != 0)[0])
print('TCP z range:', round(float(tcp[:,2].min()),3), '->', round(float(tcp[:,2].max()),3))
"
```

## Architecture A: detection

Runs in the lerobot venv, because that is where torch lives.

```bash
cd ~/lerobot
uv run python /home/jj/Documents/NTU/Y4S1/FYP/scripts/check_detector.py \
  --image /home/jj/Documents/NTU/Y4S1/FYP/data/cache/frame_top.png \
  --queries "red cube" "metal tray" "orange juice" "water bottle" "glass" \
  --model google/owlv2-base-patch16-ensemble --threshold 0.3 \
  --nms-iou 0.5 --top1-per-query \
  --out  /home/jj/Documents/NTU/Y4S1/FYP/data/cache/detections_owlv2.png \
  --json /home/jj/Documents/NTU/Y4S1/FYP/data/cache/detections.json
```

- always pass `--model` and `--threshold` explicitly; the script defaults to OWL-ViT at 0.1, which is the combination that does not work
- NMS is class-agnostic by default, because OWLv2 fires two labels on one object and per-query suppression would keep both
- drop `--top1-per-query` once a scene can hold more than one instance of a query

## Architecture B: LeRobot

```bash
# one-time
cd ~/lerobot && uv sync --extra dataset --extra training --extra smolvla
```

HDF5 to LeRobot dataset:

```bash
cd ~/lerobot
PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
uv run --extra dataset --with h5py python /home/jj/Documents/NTU/Y4S1/FYP/scripts/export_lerobot.py \
  --episodes /home/jj/Documents/NTU/Y4S1/FYP/data/raw/episodes \
  --repo-id  jj/ur5e_pickplace \
  --task     "pick and place the block" \
  --camera   top \
  --root     /home/jj/Documents/NTU/Y4S1/FYP/data/processed/lerobot_ur5e
```

Fine-tune SmolVLA. `--steps=1` is the plumbing dry run; raise it for a real run.

```bash
cd ~/lerobot && uv run --extra dataset --extra training lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=jj/ur5e_pickplace \
  --dataset.root=/home/jj/Documents/NTU/Y4S1/FYP/data/processed/lerobot_ur5e \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=1 \
  --output_dir=outputs/train/dryrun_smolvla \
  --job_name=dryrun \
  --policy.device=cuda \
  --wandb.enable=false
```

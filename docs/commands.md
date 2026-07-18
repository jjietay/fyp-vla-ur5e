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
uv run python scripts/attach_gripper.py

## Opens MuJoCo Server
XDG_SESSION_TYPE=x11 uv run python -m fyp.sim.sim_server

## Opens Client
uv run python
from fyp.sim.sim_client import SimClient
c = SimClient()
c.start_recording()
c.stop_and_save("data/episodes/ep_001.h5")
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

## Convert HDF5 episodes into a LeRobot Dataset
PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
python /home/jj/Documents/NTU/Y4S1/FYP/scripts/hdf5_to_lerobot.py \
  --episodes /home/jj/Documents/NTU/Y4S1/FYP/data/episodes \
  --repo-id  jj/ur5e_pickplace \
  --task     "pick and place the block" \
  --camera   top
  --root     /path_to_save_the_final_dataset



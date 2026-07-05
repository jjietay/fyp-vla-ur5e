"""Replay a recorded episode as a video.
Usage: python scripts/replay_episode.py data/episodes/ep_001.h5
"""

import sys
import h5py
import imageio

path = sys.argv[1] if len(sys.argv) > 1 else "data/episodes/ep_001.h5"
out = path.replace(".h5", ".mp4")

with h5py.File(path, "r") as f:
    images = f["images"][:]

imageio.mimsave(out, images, fps=20)
print(f"saved {out}  ({len(images)} frames)")

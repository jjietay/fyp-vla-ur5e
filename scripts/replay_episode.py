""" replay_episode.py

Replay a recorded episode as a video.

Usage: python scripts/replay_episode.py data/raw/episodes/ep_001.h5
"""

import sys

import imageio

from fyp.demos.hdf5_store import load_episode

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/episodes/ep_001.h5"
out = path.replace(".h5", ".mp4")

images = load_episode(path)["images"]

imageio.mimsave(out, images, fps=20)
print(f"saved {out}  ({len(images)} frames)")

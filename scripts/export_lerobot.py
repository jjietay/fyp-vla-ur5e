""" export_lerobot.py

Imports lerobot_export's convert function and actually does the conversion.
Convert recorded HDF5 demonstration episodes into a LeRobot dataset.

Thin CLI; the conversion lives in fyp.architecture_b.demos.lerobot_export. Needs the lerobot
venv (torch + lerobot), not the FYP venv.

    python scripts/export_lerobot.py \
        --episodes data/raw/episodes \
        --repo-id  <hf_user>/ur5e_pickplace \
        --task     "pick and place the block" \
        --camera   top
"""

from __future__ import annotations

import argparse

from fyp.architecture_b.demos.lerobot_export import convert


def main() -> None:
    p = argparse.ArgumentParser(description="Convert HDF5 demos to a LeRobot dataset.")
    p.add_argument("--episodes", required=True, help="dir containing *.h5 episodes")
    p.add_argument("--repo-id", required=True, help="HF dataset id, e.g. user/ur5e_pickplace")
    p.add_argument("--task", required=True, help="natural-language task string")
    p.add_argument("--camera", default="top", help="camera key name in observation.images.*")
    p.add_argument("--fps", type=int, default=None, help="override; else inferred from timestamps")
    p.add_argument("--root", default=None, help="output dir; else HF cache")
    args = p.parse_args()
    convert(args.episodes, args.repo_id, args.task, args.camera, args.fps, args.root)


if __name__ == "__main__":
    main()

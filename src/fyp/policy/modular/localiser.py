"""localiser.py

This file converts the detections + depth into 3D object positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fyp.helpers.pixel_to_3d import CameraIntrinsics, depth_at, pixel_to_camera

DUPLICATE_SEPARATION_MM = 20.0


@dataclass
class LocatedObject:
    """
    This is a container holding one detection + where it landed in 3D.
    """
    query: str
    score: float
    center_uv: tuple
    depth_m: float
    p_cam: np.ndarray | None

    @property
    def valid(self) -> bool:
        """
        fn valid is written like a method but will be accessed like a field same as those above.
        This valid function is never stored but computed from p_cam, therefore its a computed attribute.
        """
        return self.p_cam is not None and bool(np.all(np.isfinite(self.p_cam)))


def locate(detections: list[dict], depth: np.ndarray, intr: CameraIntrinsics,
           max_depth: float | None = None, radius: int = 2) -> list[LocatedObject]:
    """
    This function takes in the detections, depth map, camera intrinsics, max_depth allowed, and radius
    to return a confirmed list of located objects.
    """
    out: list[LocatedObject] = []
    for d in detections:
        u, v = d["center_uv"]
        z = depth_at(depth, u, v, radius=radius, max_depth=max_depth)
        p = pixel_to_camera(u, v, z, intr) if np.isfinite(z) else None
        out.append(LocatedObject(query=d["query"], score=d.get("score", float("nan")),
                                 center_uv=(u, v), depth_m=z, p_cam=p))
    return out


def find_duplicates(located: list[LocatedObject],
                    separation_mm: float = DUPLICATE_SEPARATION_MM) -> list[tuple]:
    """
    This function compares the list of located objects, and identifies those pairs that are
    suspiciously close together, deeming them as duplicates.
    """
    pairs = []
    for i in range(len(located)):
        for j in range(i + 1, len(located)):
            a, b = located[i], located[j]
            if a.valid and b.valid:
                sep = float(np.linalg.norm(a.p_cam - b.p_cam)) * 1000.0
                if sep < separation_mm:
                    pairs.append((i, j, sep))
    return pairs


def nearest_truth(p_cam: np.ndarray, truth_cam: dict) -> tuple:
    """
    Takes a 3D point and gives name of the closest ground truth block and how far away it is."""
    near, dist = None, float("inf")
    for name, tc in truth_cam.items():
        e = float(np.linalg.norm(p_cam - tc))
        if e < dist:
            near, dist = name, e
    return near, dist

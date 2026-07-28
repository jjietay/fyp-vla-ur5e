"""Detections + depth -> 3D object positions. Architecture A stage 3.

Takes the box centres produced by stage 2, samples the depth map there, and
back-projects through the pinhole model to camera-frame XYZ.

Output is in the CAMERA frame. Stage 4 (hand-eye) converts to the robot base
frame, which is what the primitives actually consume.

Two failure modes this module refuses to paper over:

  - **No usable depth.** An occluded or background pixel yields NaN, and the
    detection is reported as INVALID rather than given a plausible-looking
    wrong number. Silently substituting a guess here would put the arm
    somewhere the object is not.
  - **Duplicate detections.** Two boxes that back-project to nearly the same
    3D point almost always mean one physical object was detected twice (see the
    bin-labelled-as-blue-cube case). That is flagged, not merged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fyp.helpers.pixel_to_depth import CameraIntrinsics, depth_at, pixel_to_camera

DUPLICATE_SEPARATION_MM = 20.0


@dataclass
class LocatedObject:

    query: str
    score: float
    center_uv: tuple
    depth_m: float
    p_cam: np.ndarray | None

    @property
    def valid(self) -> bool:
        return self.p_cam is not None and bool(np.all(np.isfinite(self.p_cam)))


def locate(detections: list[dict], depth: np.ndarray, intr: CameraIntrinsics,
           max_depth: float | None = None, radius: int = 2) -> list[LocatedObject]:
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
    near, dist = None, float("inf")
    for name, tc in truth_cam.items():
        e = float(np.linalg.norm(p_cam - tc))
        if e < dist:
            near, dist = name, e
    return near, dist

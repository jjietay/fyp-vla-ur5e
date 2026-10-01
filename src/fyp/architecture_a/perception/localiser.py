"""localiser.py

This file converts the detections + depth into 3D object positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fyp.architecture_a.perception.pixel_to_3d import (CameraIntrinsics, depth_at,
                                                       pixel_to_camera, surface_to_centroid)



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
    centre_offset_m: float = 0.0   # how far p_cam was pushed from the seen surface

    @property
    def valid(self) -> bool:
        """
        fn valid is written like a method but will be accessed like a field same as those above.
        This valid function is never stored but computed from p_cam, therefore its a computed attribute.
        """
        return self.p_cam is not None and bool(np.all(np.isfinite(self.p_cam)))


def locate(detections: list[dict], depth: np.ndarray, intr: CameraIntrinsics,
           max_depth: float | None = None, radius: int = 2,
           max_centre_offset: float | None = None) -> list[LocatedObject]:
    """
    This function takes in the detections, depth map, camera intrinsics, max_depth allowed, and radius
    to return a confirmed list of located objects.

    Each point is the object's CENTRE, not the surface the depth landed on. The
    depth at the box centre is on the face turned to the camera, so a grasp aimed
    there puts a finger on the object (found on the first real pick, 1 Oct 2026:
    an orange, about 35 mm off). The point is pushed back along its ray by half
    the object's size, measured from the box: size = pixels * depth / focal
    length. The SMALLER of width and height is used, because a flat object seen
    at an angle is short in the image and must not be pushed a long way.

    `max_centre_offset` caps the push, in metres, because a box is only a rough
    size and a large flat object would otherwise be pushed through its own floor.
    It defaults to `architecture_a.perception.max_centre_offset_mm`. A detection
    with no box is not pushed at all.
    """
    if max_centre_offset is None:
        from fyp.shared.helpers.config import get_config
        max_centre_offset = float(
            get_config()["architecture_a"]["perception"]["max_centre_offset_mm"]) / 1000.0

    out: list[LocatedObject] = []
    for d in detections:
        u, v = d["center_uv"]
        z = depth_at(depth, u, v, radius=radius, max_depth=max_depth)
        p = pixel_to_camera(u, v, z, intr) if np.isfinite(z) else None

        offset = 0.0
        box = d.get("box_xyxy")
        if p is not None and box is not None:
            w_m = (box[2] - box[0]) * z / intr.fx
            h_m = (box[3] - box[1]) * z / intr.fy
            offset = min(0.5 * min(w_m, h_m), max_centre_offset)
            p = surface_to_centroid(p, offset)

        out.append(LocatedObject(query=d["query"], score=d.get("score", float("nan")),
                                 center_uv=(u, v), depth_m=z, p_cam=p,
                                 centre_offset_m=float(offset)))
    return out


def find_duplicates(located: list[LocatedObject],
                    separation_mm: float | None = None) -> list[tuple]:
    """
    It takes located objects and gives you the pairs that are suspiciously close
    together, which are the same object detected twice.

    `separation_mm` defaults to `architecture_a.perception.duplicate_separation_mm`
    in config.yaml. It is read there rather than kept as a constant here because
    it depends on how small the objects on the bench are, so it is a per-setup
    number. Read lazily, so importing this module still costs nothing.
    """
    if separation_mm is None:
        from fyp.shared.helpers.config import get_config
        separation_mm = float(
            get_config()["architecture_a"]["perception"]["duplicate_separation_mm"])
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

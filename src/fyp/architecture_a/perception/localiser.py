"""localiser.py

This file converts the detections + depth into 3D object positions.

Two methods, chosen by `architecture_a.perception.localiser` in config, and
`localise()` is the one entry point, called through `scene.Perception.see`, so a
measurement by a script is a measurement by the pipeline:

    box_depth   the box centre's depth, pushed back along the ray by half the
                box (`locate`). The original method.
    mask_ray    SAM mask, then the ray through the mask's centre cut at the
                object's centre height above the known table (`locate_on_table`).
                Depth only measures how TALL the object is. 13 Oct 2026.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fyp.architecture_a.perception.pixel_to_3d import (CameraIntrinsics, depth_at,
                                                       pixel_to_camera, ray_to_plane,
                                                       surface_to_centroid)



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
    p_base: np.ndarray | None = None  # the object's centre in base frame, unclamped
    centre_offset_m: float = 0.0   # how far p_cam was pushed from the seen surface
    depth_from: str = "centre"     # "centre" patch, "box" when the centre had no depth, or "mask_ray"
    height_m: float = float("nan")  # mask_ray: object height above the table it used
    height_from: str = ""           # mask_ray: "depth" (top of the mask's points) or "mask_size"
    mask_px: int = 0                # mask_ray: object pixels after erosion

    @property
    def valid(self) -> bool:
        """
        fn valid is written like a method but will be accessed like a field same as those above.
        This valid function is never stored but computed from the point, therefore its a computed attribute.
        """
        p = self.position
        return p is not None and bool(np.all(np.isfinite(p)))

    @property
    def position(self) -> np.ndarray | None:
        """
        It takes nothing and gives you the object's point in base frame when the
        localiser made one, else in camera frame.

        `mask_ray` works in base frame and fills only `p_base`. `box_depth` works
        in camera frame, and `localise()` fills `p_base` for it. The camera-frame
        fallback is for callers of `locate()` that have no transform.
        """
        return self.p_base if self.p_base is not None else self.p_cam


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
        source = "centre"
        box = d.get("box_xyxy")
        if not np.isfinite(z) and box is not None:
            # The centre patch can land on a hole: glare on a flat white lid, or
            # the dark floor inside a container. 13 Oct 2026: a white box was
            # found by OWLv2 every time and dropped every time for this. The
            # middle half of its box is still the object, so use the median of
            # whatever valid depth is there, and say so in the trace.
            z = _box_depth(depth, box, max_depth)
            source = "box"
        p = pixel_to_camera(u, v, z, intr) if np.isfinite(z) else None

        offset = 0.0
        if p is not None and box is not None:
            w_m = (box[2] - box[0]) * z / intr.fx
            h_m = (box[3] - box[1]) * z / intr.fy
            offset = min(0.5 * min(w_m, h_m), max_centre_offset)
            p = surface_to_centroid(p, offset)

        out.append(LocatedObject(query=d["query"], score=d.get("score", float("nan")),
                                 center_uv=(u, v), depth_m=z, p_cam=p,
                                 centre_offset_m=float(offset), depth_from=source))
    return out


def _box_depth(depth: np.ndarray, box, max_depth: float | None) -> float:
    """
    It takes a depth image and a detection box and gives you the median valid
    depth over the middle half of the box, or NaN when it has none.

    The middle half, not the whole box: a box includes background at its edges,
    and that is the table behind the object, not the object.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    H, W = depth.shape
    ya, yb = max(0, int(y0 + h / 4)), min(H, int(np.ceil(y1 - h / 4)))
    xa, xb = max(0, int(x0 + w / 4)), min(W, int(np.ceil(x1 - w / 4)))
    patch = np.asarray(depth[ya:yb, xa:xb], dtype=float)
    ok = np.isfinite(patch) & (patch > 0)
    if max_depth is not None:
        ok &= patch < max_depth
    return float(np.median(patch[ok])) if ok.any() else float("nan")


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
                sep = float(np.linalg.norm(a.position - b.position)) * 1000.0
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


# ------------------------------------------------------------------ mask_ray


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """
    It takes a boolean mask and gives you it shrunk by `px` pixels all round.

    The outline of an object is exactly where the stereo depth blends it with
    the table behind, so those pixels are the ones not to trust.
    """
    m = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(px))):
        inner = m.copy()
        inner[1:, :] &= m[:-1, :]
        inner[:-1, :] &= m[1:, :]
        inner[:, 1:] &= m[:, :-1]
        inner[:, :-1] &= m[:, 1:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        m = inner
    return m


def locate_on_table(detections: list[dict], masks: list, depth: np.ndarray,
                    intr: CameraIntrinsics, T_base_cam, table_z: float, *,
                    height_from: str = "depth", erode_px: int = 3,
                    percentile: float = 95.0, min_points: int = 30,
                    resting_gap: float = 0.025,
                    min_height: float = 0.01, max_height: float = 0.30,
                    max_depth: float | None = None) -> list[LocatedObject]:
    """
    It takes detections, one mask per detection, the depth image, the camera
    model and transform, and the table height, and gives you LocatedObjects whose
    point is the object's centre.

    For each object:
      1. its pixel centre is the mean of its mask, shrunk by `erode_px` so the
         outline, where depth blends the object into the table, is dropped
      2. the height of its centre, by `height_from`:
           depth      from its own depth points in base frame (percentiles, so
                      stray pixels cannot set them). If its lowest points come
                      within `resting_gap` of the table it is RESTING there, and
                      its centre is halfway from the table to its top. If not
                      (a drawer handle), its centre is halfway between its own
                      lowest and highest points. Not the midpoint for a resting
                      object: the camera looks down and sees more of the top than
                      the bottom, so that midpoint sits high (15 to 18 mm off for
                      the orange in the synthetic test)
           mask_size  the table plus half its size, size = the full mask's
                      smaller pixel extent times its camera depth over fx. Uses
                      no object depth, so a depth bias cannot reach it, but it
                      is only right for compact things RESTING on the table
         `depth` falls back to `mask_size` when the object has too few depth
         points.
      3. the point is where the centre pixel's ray reaches that height

    A height error h moves the point sideways by about h / tan(ray angle),
    roughly 1.5 to 1.9 h for this camera. That is the only way depth error gets
    in, where the old method put the whole of it along the line of sight.

    `p_base` is the result. `p_cam` stays None: the point is not converted back
    to camera frame, because nothing downstream needs it there.
    """
    if height_from not in ("depth", "mask_size"):
        raise ValueError(f"height_from is {height_from!r}; use 'depth' or 'mask_size'")
    T = np.asarray(T_base_cam, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.linalg.inv(T)
    table_z = float(table_z)
    out: list[LocatedObject] = []

    for det, mask in zip(detections, masks):
        if mask is None or not np.asarray(mask).any():
            # No mask: the box itself, which is the old method's footprint.
            mask = np.zeros(depth.shape, dtype=bool)
            x0, y0, x1, y1 = (int(round(v)) for v in det["box_xyxy"])
            mask[max(0, y0):y1, max(0, x0):x1] = True
        full = np.asarray(mask, dtype=bool)
        m = erode(full, erode_px)
        if m.sum() < 10:
            m = full
        vs, us = np.nonzero(m)
        uc, vc = float(us.mean()), float(vs.mean())

        centre_z, height, source = None, float("nan"), "mask_size"
        if height_from == "depth":
            d = np.asarray(depth, dtype=float)[vs, us]
            ok = np.isfinite(d) & (d > 0)
            if max_depth is not None:
                ok &= d < max_depth
            if ok.sum() >= min_points:
                P = np.stack([(us[ok] - intr.cx) / intr.fx * d[ok],
                              (vs[ok] - intr.cy) / intr.fy * d[ok], d[ok]], axis=1)
                z = P @ R.T[:, 2] + t[2]
                z_hi = float(np.percentile(z, percentile))
                z_lo = float(np.percentile(z, 100.0 - percentile))
                if z_lo - table_z <= resting_gap:
                    centre_z, source = (table_z + z_hi) / 2.0, "depth_top"
                else:
                    centre_z, source = (z_lo + z_hi) / 2.0, "depth_mid"
                centre_z = max(centre_z, table_z + min_height / 2.0)
                height = z_hi - table_z

        if centre_z is None:
            # Size from the FULL mask: shrinking it would shrink the object.
            fv, fu = np.nonzero(full)
            size_px = float(min(fu.max() - fu.min() + 1, fv.max() - fv.min() + 1))
            height = 0.05
            for _ in range(3):        # the depth to the object depends on its height
                p = ray_to_plane(uc, vc, intr, T, table_z + height / 2.0)
                if p is None:
                    break
                z_cam = float((T_inv @ np.append(p, 1.0))[2])
                height = float(np.clip(size_px * z_cam / intr.fx, min_height, max_height))
            centre_z = table_z + height / 2.0

        p_base = ray_to_plane(uc, vc, intr, T, centre_z)
        # Camera z of the point, only so `depth_m` still reports what the camera
        # would have measured.
        z_cam = float("nan") if p_base is None else float((T_inv @ np.append(p_base, 1.0))[2])
        out.append(LocatedObject(
            query=det["query"], score=det.get("score", float("nan")),
            center_uv=(round(uc, 1), round(vc, 1)),
            depth_m=z_cam, p_cam=None, p_base=p_base, depth_from="mask_ray", height_m=round(float(height), 4),
            height_from=source, mask_px=int(m.sum())))
    return out


def localise(records: list[dict], frame, T_base_cam, cfg: dict) -> list[LocatedObject]:
    """
    It takes the kept detections as records, one camera frame, the transform and
    the config, and gives you LocatedObjects by the method config names.

    The single entry point, so every caller measures the same way. Every valid
    result has `p_base` filled, whichever method made it.
    """
    a = cfg["architecture_a"]
    p = a["perception"]
    max_depth = float(cfg["camera"]["max_depth"])
    method = p["localiser"]
    if method == "box_depth":
        found = locate(records, frame.depth, frame.intrinsics, max_depth=max_depth,
                       radius=int(cfg["camera"]["depth_patch_radius"]))
        T = np.asarray(T_base_cam, dtype=float)
        for o in found:
            if o.valid:
                o.p_base = T[:3, :3] @ o.p_cam + T[:3, 3]
        return found
    if method == "mask_ray":
        from PIL import Image

        from fyp.architecture_a.perception.segmenter import segment
        seg = segment(Image.fromarray(frame.rgb), [r["box_xyxy"] for r in records],
                      a["segmenter"]["model"])
        return locate_on_table(
            records, [m for m, _ in seg], frame.depth, frame.intrinsics, T_base_cam,
            float(a["calibration"]["table_z"]),
            height_from=str(p["height_from"]),
            erode_px=int(p["mask_erode_px"]),
            percentile=float(p["height_percentile"]),
            resting_gap=float(p["resting_gap_mm"]) / 1000.0,
            min_points=int(p["min_mask_points"]),
            min_height=float(p["min_object_height_mm"]) / 1000.0,
            max_height=float(p["max_object_height_mm"]) / 1000.0,
            max_depth=max_depth)
    raise ValueError(f"architecture_a.perception.localiser is {method!r}; "
                     "use 'mask_ray' or 'box_depth'")


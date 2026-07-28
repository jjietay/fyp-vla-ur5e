"""Detection filtering: IoU, non-maximum suppression, per-query top-1.

Pure list-and-arithmetic code with NO torch import, deliberately split out of
`detector.py`. Two reasons:

  1. it is the only part of stage 2 that survives swapping the detection model,
  2. it can be unit-tested from the FYP venv, which has no torch at all.

A detection is a plain `(query: str, score: float, box_xyxy: list[float])`
triple throughout.
"""
from __future__ import annotations


def iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms(dets, iou_thresh: float = 0.5, per_query: bool = False):
    """Greedy non-maximum suppression over (query, score, box) triples.

    Class-AGNOSTIC by default (per_query=False), which is the behaviour we want:
    OWLv2 will happily fire "red cube" AND "yellow cube" on the same physical
    block. Per-query NMS would keep both (different labels never compete) and
    depth-to-3D would then emit two 3D points for one object. Class-agnostic
    suppression keeps only the higher-scoring label per image region.

    Set per_query=True only if the scene genuinely has different objects that
    overlap in the image (e.g. a block sitting inside the bin).
    """
    keep: list = []
    for det in sorted(dets, key=lambda d: -d[1]):     # highest score first
        q, _score, box = det
        if any(iou(box, kbox) >= iou_thresh
               for kq, _ks, kbox in keep
               if not per_query or kq == q):
            continue                                   # suppressed by a better box
        keep.append(det)
    return keep


def top1_per_query(dets):
    """Keep only the single highest-scoring box per query string.

    Valid when the scene contains at most one instance of each query - which is
    exactly the 4-cube workspace. Do NOT use once duplicate objects appear.
    """
    best: dict = {}
    for det in dets:
        q, score, _box = det
        if q not in best or score > best[q][1]:
            best[q] = det
    return sorted(best.values(), key=lambda d: -d[1])


def apply_filters(dets, threshold: float, nms_iou: float = 0.5,
                  per_query_nms: bool = False, keep_top1: bool = False) -> tuple:
    """Run the full filter chain and report the count after each step.

    ORDER MATTERS, and it is threshold -> NMS -> optional top-1:
      - threshold first so NMS never wastes a slot on a junk box;
      - NMS before top-1 so a duplicate cannot win its query on a spurious label.

    Returns (kept, counts) where counts is (n_raw, n_threshold, n_nms, n_final),
    which is what the diagnostic print in the CLI reports.
    """
    n_raw = len(dets)
    keep = [d for d in dets if d[1] >= threshold]
    n_thresh = len(keep)

    if nms_iou > 0:
        keep = nms(keep, iou_thresh=nms_iou, per_query=per_query_nms)
    n_nms = len(keep)

    if keep_top1:
        keep = top1_per_query(keep)

    return keep, (n_raw, n_thresh, n_nms, len(keep))


def to_json_records(dets) -> list[dict]:
    """Serialise kept detections, precomputing the centre pixel.

    `center_uv` is exactly the pixel the localiser samples depth at, so writing
    it here keeps that choice in one place rather than recomputing it downstream.
    """
    return [
        {"query": q, "score": round(s, 4), "box_xyxy": box,
         "center_uv": [round((box[0] + box[2]) / 2, 1),
                       round((box[1] + box[3]) / 2, 1)]}
        for q, s, box in dets
    ]

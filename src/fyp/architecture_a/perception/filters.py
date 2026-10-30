""" filters.py

This file filters out irrelevant detections using a few key methods:
    1) intersection over union (iou)
    2) non-maximum suppresion (nms)
    3) per-query top-1

It only contains lists and airthmetic with no torch import and split from
detector.py for 2 main reasons:
    
    1) it can survive the swapping the detection model
    2) it can be unit-tested from the FYP venv, which does not rely on torch
"""
from __future__ import annotations


def iou(a: list[float], b: list[float]) -> float:
    """
    Intersection over union (IOU) divides the intersection between 2 bounding boxes 
    by the total area of the two boxes.
    """
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
    """
    Non-maximum suppression (NMS) keeps the highest scoring bounding box
    in the region, delete the rest.
    """
    keep: list = []
    for det in sorted(dets, key=lambda d: -d[1]):
        q, _score, box = det
        if any(iou(box, kbox) >= iou_thresh
               for kq, _ks, kbox in keep
               if not per_query or kq == q):
            continue
        keep.append(det)
    return keep


def top1_per_query(dets):
    """
    It takes detections and gives you the single best scoring box per label,
    returned in descending score order.

    For a scene known to hold at most one of each thing. NMS cannot do this,
    because two boxes on genuinely different objects do not overlap and so never
    suppress each other. Leave it off the moment a scene can hold two red cubes.
    """
    return top_k_per_query(dets, 1)


def top_k_per_query(dets, k: int | dict[str, int]):
    """
    It takes detections and a count k and gives you the k best scoring boxes per
    label, returned in descending score order.

    k above 1 is for a scene that can hold two of a thing, such as two bottles.
    Each extra box is also a chance of a false detection, so keep k small. A dict
    gives each label its own k, and a label missing from it keeps 1.
    """
    limit = (lambda q: k.get(q, 1)) if isinstance(k, dict) else (lambda q: k)
    kept: list = []
    count: dict[str, int] = {}
    for det in sorted(dets, key=lambda d: -d[1]):
        if count.get(det[0], 0) < limit(det[0]):
            count[det[0]] = count.get(det[0], 0) + 1
            kept.append(det)
    return kept


def apply_filters(dets, threshold: float, nms_iou: float = 0.5,
                  per_query_nms: bool = False, keep_top1: bool = False,
                  keep_top_k: int | dict[str, int] | None = None) -> tuple:
    """
    this function requires these few inputs:
    - detections
    - threshold for each individual scores
    - nms_iou which is threshold for maximum overlap of default 50%
    - per_query_nms only let boxes wih the same label to suppress each other (off by default)
    - keep_top1 keeps only the single best-scoring box per label (off by default)
    - keep_top_k keeps the k best-scoring boxes per label (None keeps all), or a
      dict of k per label. keep_top1 is keep_top_k=1
    """
    n_raw = len(dets)
    keep = [d for d in dets if d[1] >= threshold]
    n_thresh = len(keep)

    if nms_iou > 0:
        keep = nms(keep, iou_thresh=nms_iou, per_query=per_query_nms)
    n_nms = len(keep)

    if keep_top1:
        keep_top_k = 1
    if keep_top_k is not None:
        keep = top_k_per_query(keep, keep_top_k)

    return keep, (n_raw, n_thresh, n_nms, len(keep))


def to_json_records(dets) -> list[dict]:
    """
    this takes in a tuple (dets format) and outputs the query, score, bounding box loc,
    pixels at center of box as a dictionary in a list for all detections.
    """
    return [
        {"query": q, "score": round(s, 4), "box_xyxy": box,
         "center_uv": [round((box[0] + box[2]) / 2, 1),
                       round((box[1] + box[3]) / 2, 1)]}
        for q, s, box in dets
    ]

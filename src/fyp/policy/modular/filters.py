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
    Arrange the query in ascending scores.
    """
    best: dict[str, tuple] = {}
    for det in dets:
        q, score, _box = det
        if q not in best or score > best[q][1]:
            best[q] = det
    return sorted(best.values(), key=lambda d: -d[1])


def apply_filters(dets, threshold: float, nms_iou: float = 0.5,
                  per_query_nms: bool = False, keep_top1: bool = False) -> tuple:
    """
    this function requires these few inputs:
    - detections
    - threshold for each individual scores
    - nms_iou which is threshold for maximum overlap of default 50%
    - per_query_nms only let boxes wih the same label to suppress each other (off by default)
    - keep_top1 keeps only the single best-scoring box per label (off by default)
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

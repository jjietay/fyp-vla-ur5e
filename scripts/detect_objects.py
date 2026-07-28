"""Open-vocabulary object detection on a rendered frame (Architecture A, stage 1).

Two stages that can run together or split across environments:

  1. RENDER  - needs MuJoCo (your FYP venv). Renders the fixed_cam view of
               scene_gripper.xml at the home keyframe and saves a PNG.
  2. DETECT  - needs torch + transformers + pillow (your lerobot venv). Runs
               OWL-ViT with text queries, thresholds, draws boxes, saves overlay.

Because MuJoCo and torch live in different venvs here, the safe workflow is to
split them:

    # in the FYP venv (has mujoco):
    uv run python scripts/detect_objects.py --render-only --out data/frames/frame.png

    # in the lerobot venv (has torch + transformers); add pillow if missing:
    PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
    uv run --extra ... python /home/jj/Documents/NTU/Y4S1/FYP/scripts/detect_objects.py \
        --image data/frames/frame.png \
        --queries "red block" "green block" "blue block" "yellow block" "bin" \
        --threshold 0.1 --out data/frames/detections.png

If a single venv has BOTH mujoco and torch, just omit --image/--render-only and
it renders + detects in one shot.

Boxes are returned in pixel xyxy (x0, y0, x1, y1) - exactly the format stage 2
(depth-to-3D) will consume via the box centre.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def render_frame(width: int, height: int, camera: str | None = None) -> np.ndarray:
    """Render a camera view at the home keyframe. Returns (H,W,3) uint8.

    camera: MuJoCo camera name to render from; defaults to config's sim.camera.name.
    """
    import mujoco
    from fyp.config import get_config, resolve

    sim = get_config()["sim"]
    scene = resolve(sim["scene"])
    cam = camera or sim["camera"]["name"]

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    # Reset to the 'home' keyframe so the blocks sit at their authored poses.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()          # (H, W, 3) uint8
    renderer.close()
    return img


def detect(pil_img, queries, model_name, threshold):
    """Run OWL-ViT. Returns list of (query, score, [x0,y0,x1,y1])."""
    import torch
    # Generic zero-shot detection API: works for OWL-ViT, OWLv2, and GroundingDINO,
    # so --model can be swapped without changing loader classes.
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    inputs = processor(text=[queries], images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes wants (height, width); PIL .size is (width, height).
    target_sizes = torch.tensor([pil_img.size[::-1]], device=device)

    # transformers renamed this: newer builds expose post_process_grounded_object_detection
    # (and may return text_labels instead of integer label indices). Handle both.
    # Use a near-zero floor so we can SEE the raw scores; caller filters by threshold.
    post = getattr(processor, "post_process_grounded_object_detection", None) \
        or processor.post_process_object_detection
    try:
        results = post(outputs, threshold=0.0, target_sizes=target_sizes,
                       text_labels=[queries])[0]
    except TypeError:
        results = post(outputs, threshold=0.0, target_sizes=target_sizes)[0]

    text_labels = results.get("text_labels")          # newer API: list[str] or None
    int_labels = results.get("labels")                # older API: tensor of indices

    dets = []
    for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
        if text_labels is not None and text_labels[i] is not None:
            q = text_labels[i]
        else:
            q = queries[int(int_labels[i])]
        dets.append((q, float(score), [round(float(v), 1) for v in box.tolist()]))
    return sorted(dets, key=lambda d: -d[1])


def _iou(a: list[float], b: list[float]) -> float:
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
        if any(_iou(box, kbox) >= iou_thresh
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", nargs="+",
                   default=["red block", "green block", "blue block", "yellow block", "bin"],
                   help="open-vocab text prompts")
    p.add_argument("--image", default=None,
                   help="detect on this image instead of rendering (skips MuJoCo)")
    p.add_argument("--render-only", action="store_true",
                   help="render a frame and save it, then exit (skips torch)")
    p.add_argument("--camera", default=None,
                   help="MuJoCo camera to render from (default: config sim.camera.name)")
    p.add_argument("--model", default="google/owlvit-base-patch32")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="class-agnostic NMS IoU threshold; 0 disables NMS")
    p.add_argument("--per-query-nms", action="store_true",
                   help="only suppress boxes sharing the same query label")
    p.add_argument("--top1-per-query", action="store_true",
                   help="keep at most one box per query (scene has <=1 of each)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out", default="data/frames/detections.png")
    p.add_argument("--json", default=None,
                   help="also write the kept detections here as JSON (feeds depth-to-3D)")
    args = p.parse_args()

    from PIL import Image, ImageDraw

    # ---- get the frame ----
    if args.image:
        img = np.array(Image.open(args.image).convert("RGB"))
    else:
        img = render_frame(args.width, args.height, camera=args.camera)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        Image.fromarray(img).save(args.out)
        print(f"saved frame -> {args.out}  ({img.shape[1]}x{img.shape[0]})")
        return

    # ---- detect + overlay ----
    pil = Image.fromarray(img)
    all_dets = detect(pil, args.queries, args.model, args.threshold)  # sorted, all scores

    # Diagnostic: show the strongest raw candidates even if below threshold.
    print(f"top raw candidates (model={args.model}):")
    for q, s, box in all_dets[:8]:
        print(f"  {q:14s} score={s:.3f}  box(xyxy)={box}")

    # ---- filter: threshold -> NMS -> optional per-query top-1 ----
    # Order matters. Threshold first so NMS never wastes a slot on a junk box;
    # NMS before top-1 so a duplicate can't win its query on a spurious label.
    keep = [d for d in all_dets if d[1] >= args.threshold]
    n_thresh = len(keep)

    if args.nms_iou > 0:
        keep = nms(keep, iou_thresh=args.nms_iou, per_query=args.per_query_nms)
    n_nms = len(keep)

    if args.top1_per_query:
        keep = top1_per_query(keep)

    print(f"\nfilter: {len(all_dets)} raw -> {n_thresh} >= {args.threshold}"
          f" -> {n_nms} after NMS(iou={args.nms_iou})"
          f" -> {len(keep)} final")

    draw = ImageDraw.Draw(pil)
    for q, s, (x0, y0, x1, y1) in keep:
        cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        # Mark the centre pixel: this is the exact (u,v) depth-to-3D will sample.
        draw.line([cu - 4, cv, cu + 4, cv], fill=(0, 255, 255), width=1)
        draw.line([cu, cv - 4, cu, cv + 4], fill=(0, 255, 255), width=1)
        draw.text((x0, max(0, y0 - 11)), f"{q} {s:.2f}", fill=(255, 255, 0))
    pil.save(args.out)
    print(f"saved overlay -> {args.out}")

    if args.json:
        import json
        payload = [
            {"query": q, "score": round(s, 4), "box_xyxy": box,
             "center_uv": [round((box[0] + box[2]) / 2, 1),
                           round((box[1] + box[3]) / 2, 1)]}
            for q, s, box in keep
        ]
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"saved detections -> {args.json}")


if __name__ == "__main__":
    main()

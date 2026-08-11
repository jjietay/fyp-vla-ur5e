"""Open-vocabulary object detection on an image (Architecture A, stage 2).

It takes a PNG and a list of text prompts, and gives you an overlay image plus a
JSON of kept boxes. That JSON is the handoff to stage 3, which samples depth at
each `center_uv`.

Runs in the lerobot venv, because it needs torch + transformers + pillow.

    cd ~/lerobot && uv run python \
        /path/to/scripts/check_detector.py \
        --image frame.png --queries "red cube" "metal tray" \
        --model google/owlv2-base-patch16-ensemble --threshold 0.3 \
        --json detections.json

--model and --threshold default to the values in config.yaml under
architecture_a.detector, so this script and the pipeline always agree about what
the detector is.

Boxes are pixel xyxy (x0, y0, x1, y1).

TODO (W3): --image is currently the only frame source. When the RealSense lands,
add a live-grab branch here rather than a separate script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", nargs="+", required=True,
                   help="open-vocab text prompts, e.g. \"red cube\" \"metal tray\"")
    p.add_argument("--image", required=True,
                   help="image to run detection on")
    p.add_argument("--model", default=None,
                   help="detector checkpoint; defaults to architecture_a.detector.model in config")
    p.add_argument("--threshold", type=float, default=None,
                   help="score threshold; defaults to architecture_a.detector.threshold in config")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="class-agnostic NMS IoU threshold; 0 disables NMS")
    p.add_argument("--per-query-nms", action="store_true",
                   help="only suppress boxes sharing the same query label")
    p.add_argument("--top1-per-query", action="store_true",
                   help="keep at most one box per query (scene has <=1 of each)")
    p.add_argument("--out", default="data/cache/detections.png")
    p.add_argument("--json", default=None,
                   help="also write the kept detections here as JSON (feeds stage 3)")
    args = p.parse_args()

    # Defaults come from config rather than being hardcoded here. They used to
    # disagree with detector.py's documented setup (OWL-ViT at 0.1 versus OWLv2
    # at 0.3), so running with defaults exercised the combination already known
    # not to work and made the detector look worse than it is.
    from fyp.shared.helpers.config import get_config
    det = get_config()["architecture_a"]["detector"]
    model = args.model or det["model"]
    threshold = args.threshold if args.threshold is not None else float(det["threshold"])

    from PIL import Image, ImageDraw


    img = np.array(Image.open(args.image).convert("RGB"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)


    from fyp.architecture_a.perception.detector import detect
    from fyp.architecture_a.perception.filters import apply_filters, to_json_records

    pil = Image.fromarray(img)
    all_dets = detect(pil, args.queries, model)


    print(f"top raw candidates (model={model}, threshold={threshold}):")
    for q, s, box in all_dets[:8]:
        print(f"  {q:14s} score={s:.3f}  box(xyxy)={box}")

    keep, (n_raw, n_thresh, n_nms, n_final) = apply_filters(
        all_dets,
        threshold=threshold,
        nms_iou=args.nms_iou,
        per_query_nms=args.per_query_nms,
        keep_top1=args.top1_per_query,
    )

    print(f"\nfilter: {n_raw} raw -> {n_thresh} >= {threshold}"
          f" -> {n_nms} after NMS(iou={args.nms_iou})"
          f" -> {n_final} final")

    draw = ImageDraw.Draw(pil)
    for q, s, (x0, y0, x1, y1) in keep:
        cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        draw.line([cu - 4, cv, cu + 4, cv], fill=(0, 255, 255), width=1)
        draw.line([cu, cv - 4, cu, cv + 4], fill=(0, 255, 255), width=1)
        draw.text((x0, max(0, y0 - 11)), f"{q} {s:.2f}", fill=(255, 255, 0))
    pil.save(args.out)
    print(f"saved overlay -> {args.out}")

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(to_json_records(keep), indent=2))
        print(f"saved detections -> {args.json}")


if __name__ == "__main__":
    main()

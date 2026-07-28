"""Open-vocabulary object detection on a rendered frame (Architecture A, stage 2).

Two halves that can run together or split across environments:

  1. RENDER  - needs MuJoCo (your FYP venv). Renders a camera view of the scene
               at the home keyframe and saves a PNG.
  2. DETECT  - needs torch + transformers + pillow (your lerobot venv). Runs the
               detector, filters, draws boxes, saves overlay + JSON.

Because MuJoCo and torch live in different venvs here, the safe workflow is to
split them:

    # in the FYP venv (has mujoco):
    uv run python scripts/check_detector.py --render-only --out data/frames/frame.png

    # in the lerobot venv (has torch + transformers):
    cd ~/lerobot && uv run python \
        /home/jj/Documents/NTU/Y4S1/FYP/scripts/check_detector.py \
        --image .../frame.png --queries "red cube" ... --json .../detections.json

If a single venv has BOTH mujoco and torch, just omit --image/--render-only and
it renders + detects in one shot.

Boxes are pixel xyxy (x0, y0, x1, y1); the JSON also carries `center_uv`, which
is exactly the pixel stage 3 samples depth at.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
                   help="also write the kept detections here as JSON (feeds stage 3)")
    args = p.parse_args()

    from PIL import Image, ImageDraw


    if args.image:
        img = np.array(Image.open(args.image).convert("RGB"))
    else:
        from fyp.hardware.sim.renderer import render_rgbd
        img, _depth, _intr = render_rgbd(args.width, args.height, camera=args.camera)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        Image.fromarray(img).save(args.out)
        print(f"saved frame -> {args.out}  ({img.shape[1]}x{img.shape[0]})")
        return


    from fyp.policy.modular.detector import detect
    from fyp.policy.modular.filters import apply_filters, to_json_records

    pil = Image.fromarray(img)
    all_dets = detect(pil, args.queries, args.model)


    print(f"top raw candidates (model={args.model}):")
    for q, s, box in all_dets[:8]:
        print(f"  {q:14s} score={s:.3f}  box(xyxy)={box}")

    keep, (n_raw, n_thresh, n_nms, n_final) = apply_filters(
        all_dets,
        threshold=args.threshold,
        nms_iou=args.nms_iou,
        per_query_nms=args.per_query_nms,
        keep_top1=args.top1_per_query,
    )

    print(f"\nfilter: {n_raw} raw -> {n_thresh} >= {args.threshold}"
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

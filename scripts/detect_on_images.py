"""detect_on_images.py

It takes saved colour images and gives you OWLv2 run on each one, with an overlay
PNG per image and one CSV of the best score per query.

    # every pre-run snapshot in the trace folder, each with the queries its own run used
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/detect_on_images.py

    # only today's, with your own queries instead
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/detect_on_images.py \
        "logs/traces/20261013*_colour.png" --queries orange "white box" "red tray"

    # any image at all
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/detect_on_images.py \
        data/cache/brown_paper.png --queries orange box

    # IMAGE QUERY: find things that look like a crop, e.g. the RG2 fingertips.
    # Without --query-box a window opens: click the crop's top-left, then bottom-right.
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/detect_on_images.py \
        data/cache/brown_paper.png "logs/traces/20261013*_colour.png" \
        --image-query data/cache/brown_paper.png

Runs in the lerobot venv, because OWLv2 needs PyTorch.

Image query. `--image-query` replaces text with an example picture, cut from that
image by `--query-box x0 y0 x1 y1` (pixels) or by two clicks. Keep the crop TIGHT
on the thing, fingertips only, because the model treats the whole crop as the
object. The crop is saved as `query_crop.png` in the output folder so you can see
what was used. Image-query scores are on a different scale from text, so the
config threshold is NOT applied: each image shows its best three boxes (green is
the best, grey the next two) and you judge them. When the query image is also
one of the images searched, the script prints how well the best box overlaps the
crop, as a check that the boxes land where they should.

Queries. With `--queries`, every image uses them. Without it, an image saved by
`run_architecture_a.py` (`<run_id>_colour.png`) uses the queries from the
grounding stage of `<run_id>.jsonl`, which is exactly what Architecture A looked
for on that run. A run that repeated an earlier instruction has no grounding
stage (the pipeline reused its cached queries), so the queries come from the
earlier run with the same instruction. An image with no trace at all, and no
`--queries`, is skipped and says so.

Filtering is the pipeline's own (`apply_filters` with the configured threshold
and NMS, top-1 per query), so a box kept here is a box A would have kept.

The overlay draws two things:
    green  kept: at or above the threshold, what A would act on
    grey   the best box BELOW the threshold for any query that kept nothing,
           with its score, so a near miss (0.27 against 0.30) is visible
           instead of looking like the object was never there
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    from fyp.shared.helpers.config import get_config, resolve

    cfg = get_config()
    det = cfg["architecture_a"]["detector"]
    trace_dir = resolve(cfg["architecture_a"]["trace"]["dir"])

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="*",
                   help="image paths or quoted glob patterns; default: every "
                        "*_colour.png in the trace folder")
    p.add_argument("--queries", nargs="+", default=None,
                   help="text prompts for every image; default: each run's own "
                        "grounding queries from its trace")
    p.add_argument("--threshold", type=float, default=None,
                   help="score threshold; default architecture_a.detector.threshold")
    p.add_argument("--model", default=None,
                   help="checkpoint; default architecture_a.detector.model")
    p.add_argument("--out", default=None,
                   help="output folder; default <trace dir>/detections")
    p.add_argument("--image-query", default=None,
                   help="search with a crop of this image instead of text")
    p.add_argument("--query-box", nargs=4, type=float, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="crop of --image-query in pixels; omit to click two corners")
    args = p.parse_args()

    model = args.model or det["model"]
    threshold = float(det["threshold"]) if args.threshold is None else args.threshold
    nms_iou = float(det["nms_iou"])
    out_dir = Path(args.out) if args.out else trace_dir / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _expand(args.images or [str(trace_dir / "*_colour.png")])
    if not paths:
        print("no images matched")
        return 1

    from PIL import Image

    from fyp.architecture_a.perception.detector import detect, detect_by_image
    from fyp.architecture_a.perception.filters import apply_filters, iou

    if args.image_query:
        q_path = Path(args.image_query)
        q_box = args.query_box or _pick_box(q_path)
        crop = Image.open(q_path).convert("RGB").crop(tuple(int(round(v)) for v in q_box))
        crop.save(out_dir / "query_crop.png")
        print(f"image query: {q_path.name} box {[round(v) for v in q_box]} "
              f"({crop.size[0]}x{crop.size[1]} px) -> {out_dir / 'query_crop.png'}")
        print(f"{len(paths)} image(s), {model}, NMS IoU {nms_iou}, no threshold")
        print("loading OWLv2 (first image only) ...")
        rows = []
        for path in paths:
            pil = Image.open(path).convert("RGB")
            raw = detect_by_image(pil, crop, model, label="query")
            top, _ = apply_filters(raw, threshold=0.0, nms_iou=nms_iou)
            top = top[:3]
            overlay = out_dir / f"{path.stem}_imgquery.png"
            _draw(pil, top[:1], top[1:], overlay)
            print(f"\n{path.name}")
            for rank, (_, sc, b) in enumerate(top, 1):
                print(f"  #{rank} score {sc:.3f}  box {[round(v) for v in b]}")
            if path.resolve() == q_path.resolve() and top:
                print(f"  self-check: best box overlaps the crop at IoU "
                      f"{iou(top[0][2], list(q_box)):.2f} (near 1 = boxes land right)")
            print(f"  -> {overlay}")
            for rank, (_, sc, b) in enumerate(top, 1):
                rows.append({"image": path.name, "rank": rank,
                             "score": round(sc, 4), "box_xyxy": [round(v, 1) for v in b]})
        _write_summary(out_dir / "summary_imgquery.csv", rows)
        return 0

    print(f"{len(paths)} image(s), {model}, threshold {threshold}, NMS IoU {nms_iou}")
    print("loading OWLv2 (first image only) ...")

    rows = []
    for path in paths:
        instruction, queries = _queries_for(path, args.queries)
        if not queries:
            print(f"\n{path.name}: skipped, no --queries and no trace with grounding queries")
            continue

        pil = Image.open(path).convert("RGB")
        raw = detect(pil, queries, model)
        kept, counts = apply_filters(raw, threshold=threshold, nms_iou=nms_iou,
                                     keep_top1=True)

        # Best raw box per query, whatever its score, for the near-miss column.
        best = {}
        for q, s, box in raw:
            if q not in best:
                best[q] = (q, s, box)
        kept_names = {d[0] for d in kept}
        near = [best[q] for q in queries if q not in kept_names and q in best]

        overlay = out_dir / f"{path.stem}_owlv2.png"
        _draw(pil, kept, near, overlay)

        print(f"\n{path.name}" + (f'   "{instruction}"' if instruction else ""))
        for q in queries:
            s = best[q][1] if q in best else float("nan")
            mark = "KEPT " if q in kept_names else "miss "
            print(f"  {mark} {q:<20} best {s:.3f}")
        print(f"  -> {overlay}")

        for q in queries:
            rows.append({"image": path.name, "instruction": instruction or "",
                         "query": q,
                         "best_score": round(best[q][1], 4) if q in best else "",
                         "kept": q in kept_names,
                         "box_xyxy": best[q][2] if q in best else "",
                         "threshold": threshold})

    if rows:
        n_kept = sum(r["kept"] for r in rows)
        print(f"\n{n_kept}/{len(rows)} query-image pairs kept at {threshold}")
        _write_summary(out_dir / "summary.csv", rows)
    return 0


def _write_summary(path: Path, rows: list[dict]) -> None:
    """It takes result rows and gives you them written as a CSV at `path`."""
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"summary -> {path}")


def _pick_box(path: Path) -> list[float]:
    """
    It takes an image path and gives you the pixel box you clicked on it:
    top-left first, then bottom-right.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(path).convert("RGB")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title("click the TOP-LEFT, then the BOTTOM-RIGHT of the fingertips (tight)")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise SystemExit("need two clicks; nothing was run")
    (xa, ya), (xb, yb) = pts
    return [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]


def _expand(patterns: list[str]) -> list[Path]:
    """It takes paths or glob patterns and gives you the matching files, sorted, once each."""
    found = []
    for pat in patterns:
        hits = glob.glob(pat) if any(c in pat for c in "*?[") else [pat]
        found += [Path(h) for h in hits if Path(h).is_file()]
    return sorted(set(found))


def _queries_for(path: Path, fixed: list[str] | None) -> tuple[str | None, list[str]]:
    """
    It takes an image path and gives you (instruction, queries) for it.

    `--queries` wins. Otherwise `<run_id>_colour.png` is matched to
    `<run_id>.jsonl` beside it, and the grounding stage's queries are used.
    """
    run_id = path.stem.removesuffix("_colour")
    trace = path.with_name(f"{run_id}.jsonl")
    instruction, queries = None, []
    if trace.is_file():
        for line in trace.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "run_start":
                instruction = rec.get("instruction")
            if rec.get("stage") == "grounding" and not queries:
                queries = list(rec.get("data", {}).get("queries", []))
    if not queries and instruction and not fixed:
        # A repeated instruction reuses the pipeline's cached queries and writes
        # no grounding stage, so borrow them from a run that did ground it.
        queries = _grounded(path.parent).get(instruction, [])
    return instruction, (list(fixed) if fixed else queries)


_GROUNDED: dict[Path, dict[str, list[str]]] = {}


def _grounded(folder: Path) -> dict[str, list[str]]:
    """It takes the trace folder and gives you instruction -> grounding queries from every trace in it."""
    if folder not in _GROUNDED:
        found: dict[str, list[str]] = {}
        for trace in sorted(folder.glob("*.jsonl")):
            instruction = None
            for line in trace.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "run_start":
                    instruction = rec.get("instruction")
                elif rec.get("stage") == "grounding" and instruction:
                    found.setdefault(instruction, list(rec.get("data", {}).get("queries", [])))
                    break
        _GROUNDED[folder] = found
    return _GROUNDED[folder]


def _draw(pil, kept, near, out: Path) -> None:
    """It takes the image and the boxes and gives you the overlay saved at `out`."""
    from PIL import ImageDraw, ImageFont

    img = pil.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow older than 10.1 has one fixed tiny font
        font = ImageFont.load_default()

    def box(q, s, b, colour, width, suffix):
        x0, y0, x1, y1 = b
        d.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
        d.line([cu - 6, cv, cu + 6, cv], fill=colour, width=2)
        d.line([cu, cv - 6, cu, cv + 6], fill=colour, width=2)
        label = f"{q} {s:.2f}{suffix}"
        tx, ty = x0, max(0, y0 - 22)
        d.rectangle(d.textbbox((tx, ty), label, font=font), fill=(0, 0, 0))
        d.text((tx, ty), label, fill=colour, font=font)

    for q, s, b in near:
        box(q, s, b, (170, 170, 170), 2, " (below)")
    for q, s, b in kept:
        box(q, s, b, (0, 230, 0), 3, "")
    img.save(out)


if __name__ == "__main__":
    sys.exit(main())

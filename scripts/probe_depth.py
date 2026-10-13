"""probe_depth.py  (diagnostic, 13 Oct 2026)

It takes the live camera and a query, and gives you the depth ON the object
against the depth of the table just around it, so you can see whether the
camera actually measures the object or smears it into the table.

    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/probe_depth.py
    ... --laser max --preset "High Accuracy"     # try better depth settings

Nothing moves.
"""
import argparse
import numpy as np
import pyrealsense2 as rs
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--query", default="orange")
p.add_argument("--laser", default=None, help="laser power in mW, or 'max'")
p.add_argument("--preset", default=None, help="e.g. 'High Accuracy', 'High Density', 'Default'")
p.add_argument("--frames", type=int, default=15)
a = p.parse_args()

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
prof = pipe.start(cfg)
dev = prof.get_device().first_depth_sensor()
scale = dev.get_depth_scale()
if a.preset:
    names = {dev.get_option_value_description(rs.option.visual_preset, i): i
             for i in range(int(dev.get_option_range(rs.option.visual_preset).max) + 1)}
    dev.set_option(rs.option.visual_preset, names[a.preset])
if dev.supports(rs.option.emitter_enabled):
    dev.set_option(rs.option.emitter_enabled, 1)
if a.laser:
    r = dev.get_option_range(rs.option.laser_power)
    dev.set_option(rs.option.laser_power, r.max if a.laser == "max" else float(a.laser))
print("preset", dev.get_option_value_description(rs.option.visual_preset,
                                                  dev.get_option(rs.option.visual_preset)),
      "| laser", dev.get_option(rs.option.laser_power), "mW",
      "| emitter", dev.get_option(rs.option.emitter_enabled))

align = rs.align(rs.stream.color)
for _ in range(30):
    pipe.wait_for_frames()
depths, rgb = [], None
for _ in range(a.frames):
    f = align.process(pipe.wait_for_frames())
    d = np.asarray(f.get_depth_frame().get_data(), dtype=np.float32) * scale
    d[d <= 0] = np.nan
    depths.append(d)
    rgb = np.asarray(f.get_color_frame().get_data())
pipe.stop()
single = depths[-1]
depth = np.nanmedian(np.stack(depths), axis=0)

from fyp.architecture_a.perception.detector import detect
from fyp.architecture_a.perception.filters import apply_filters
kept, _ = apply_filters(detect(Image.fromarray(rgb), [a.query], "google/owlv2-base-patch16-ensemble"),
                        threshold=0.2, nms_iou=0.5, keep_top1=True)
if not kept:
    raise SystemExit(f"no {a.query} found")
_q, score, (x0, y0, x1, y1) = kept[0]
cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
print(f"\n{a.query}: score {score:.2f}, box {w:.0f} x {h:.0f} px at ({cx:.0f}, {cy:.0f})")

def med(img, xa, xb, ya, yb):
    patch = img[int(max(ya, 0)):int(yb), int(max(xa, 0)):int(xb)]
    return np.nanmedian(patch), np.isnan(patch).mean() * 100

inner = (cx - w / 4, cx + w / 4, cy - h / 4, cy + h / 4)
rows = [("object centre (5x5)", (cx - 2, cx + 3, cy - 2, cy + 3)),
        ("object middle half", inner),
        ("table ABOVE (behind)", (cx - w / 4, cx + w / 4, y0 - h, y0 - 4)),
        ("table BELOW (in front)", (cx - w / 4, cx + w / 4, y1 + 4, y1 + h)),
        ("table LEFT", (x0 - w, x0 - 4, cy - h / 4, cy + h / 4)),
        ("table RIGHT", (x1 + 4, x1 + w, cy - h / 4, cy + h / 4))]
print(f"\n{'region':<24}{'median of 15':>14}{'one frame':>12}{'holes %':>9}")
for name, box in rows:
    m, holes = med(depth, *box)
    s, _ = med(single, *box)
    print(f"{name:<24}{m:>13.3f}m{s:>11.3f}m{holes:>8.0f}")
side = np.nanmean([med(depth, *rows[4][1])[0], med(depth, *rows[5][1])[0]])
print(f"\nobject centre is {1000 * (side - med(depth, *rows[0][1])[0]):+.0f} mm nearer than the "
      "table beside it.\nA solid ~55 mm object seen from ~35 deg above should be roughly +40 to "
      "+70 mm.\nNear 0 means the camera is not resolving the object: its depth is the table's.")

"""level_to_table.py

It takes the current `config/calibration/T_base_cam.json` and the camera's view
of the table, and gives you that transform corrected so the table comes out flat
at its known height.

    uv run python scripts/level_to_table.py --dry-run   # report only, write nothing
    uv run python scripts/level_to_table.py             # correct and write

Run it straight after `calibrate_ball.py`, with the camera on its tape marks.
NOTHING MOVES. It only reads the camera. Objects may stay on the table: they are
not part of the plane and are thrown out, but more bare paper in view is better.

## Why it exists (13 Oct 2026)

Two ball fits made with different grids agreed with each other to 19 mm, and both
put a real orange about 35 mm too low. A constant error like that is invisible to
the fit's own RMS: a ball that sits higher in the fingers than
`calibration.ball_offset_tool_m` says, or depth that reads a few percent long,
shifts every point the same way and the fit absorbs it.

The table fixes this, because its height in base frame is known exactly
(`calibration.table_z`, touched by the fingertips on 23 Sep 2026). The camera sees
thousands of table points. If the calibration were right they would all come out
at table_z and dead flat. Whatever height and tilt they come out at instead IS the
calibration error, measured where the objects actually sit, and
`hand_eye.level_to_plane` removes it.

It corrects height and tilt only. Sideways position and yaw are left as the ball
fit found them, because a flat table carries no information about them.

## Reading the output

  table offset    how far the table sat from table_z before the fix. This is the
                  number that was making objects read too low or too high
  table tilt      degrees the table plane was tilted. Each degree is about 17 mm
                  of height error across a metre of table
  flatness        spread of the table points about the fitted plane after the
                  fix. A few mm is paper creases and depth noise. Much more means
                  the depth itself is bending, and the camera wants an on-chip
                  calibration in realsense-viewer

The previous file is kept as `T_base_cam.<time>.json`, never overwritten.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="report the correction, write nothing")
    p.add_argument("--force", action="store_true",
                   help="write even when the correction is larger than the config limits")
    p.add_argument("--note", default="", help="free text kept in the file")
    args = p.parse_args()

    from fyp.architecture_a.calibration.hand_eye import fit_plane, level_to_plane
    from fyp.architecture_a.calibration.store import CalibrationStore
    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.safety import WorkspaceEnvelope
    from fyp.shared.helpers.config import get_config

    cfg = get_config()
    cc = cfg["architecture_a"]["calibration"]
    table_z = float(cc["table_z"])
    store = CalibrationStore(cfg)
    T, _ = store.current()

    n_frames = int(cc["level_frames"])
    print(f"reading {n_frames} depth frames ...")
    with RealSenseCamera(cfg) as cam:
        frames = [cam.capture() for _ in range(n_frames)]
    intr = frames[0].intrinsics
    depth = _median_depth([f.depth for f in frames], float(cfg["camera"]["max_depth"]))

    pts = _table_candidates(depth, intr, T, table_z, float(cc["level_band_m"]),
                            WorkspaceEnvelope(cfg))
    if len(pts) < 500:
        print(f"only {len(pts)} table points in view, need 500. Is the table bare enough, "
              "and is the current calibration roughly right (within level_band_m)?",
              file=sys.stderr)
        return 1

    normal, centroid, keep = fit_plane(pts)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    offset_mm = (centroid[2] - table_z) * 1000.0
    T_new = level_to_plane(T, normal, centroid, table_z)

    after = (T_new[:3, :3] @ np.linalg.solve(T[:3, :3], (pts[keep] - T[:3, 3]).T)).T + T_new[:3, 3]
    flat_mm = float(np.std(after[:, 2])) * 1000.0
    moved_mm = float(np.linalg.norm(T_new[:3, 3] - T[:3, 3])) * 1000.0

    print(f"\n  table points   {int(keep.sum())} on the plane, of {len(pts)} near it")
    print(f"  table offset   {offset_mm:+.1f} mm  (+ means it read ABOVE {table_z:.3f})")
    print(f"  table tilt     {tilt:.2f} deg")
    print(f"  after the fix  mean z {after[:, 2].mean():.4f}, flatness {flat_mm:.1f} mm")
    print(f"  camera moves   {moved_mm:.0f} mm in the transform")

    too_big = (tilt > float(cc["level_max_tilt_deg"])
               or abs(offset_mm) > float(cc["level_max_shift_mm"]))
    if too_big and not args.force:
        print(f"\n  REFUSED: larger than level_max_tilt_deg ({cc['level_max_tilt_deg']}) or "
              f"level_max_shift_mm ({cc['level_max_shift_mm']}). A correction this size means "
              "the ball fit or the camera is badly off. Redo calibrate_ball.py, or pass --force.",
              file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n  dry run, nothing written")
        return 0

    archived = store.replace(T_new, "levelled_to_table", {
        "table_z": table_z,
        "offset_before_mm": round(offset_mm, 1),
        "tilt_before_deg": round(tilt, 3),
        "flatness_after_mm": round(flat_mm, 1),
        "n_points": int(keep.sum()),
        "note": args.note,
    })
    print(f"\n  previous calibration kept as {archived.name}\n  wrote {store.path}")
    return 0


def _median_depth(depths, max_depth: float) -> np.ndarray:
    """
    It takes several depth images and gives you one, each pixel the median of its
    valid readings, NaN where there were none.

    One frame jumps by centimetres at this range. The median of 15 does not.
    """
    import warnings
    stack = np.stack([np.asarray(d, dtype=np.float32) for d in depths])
    stack[(stack <= 0) | ~np.isfinite(stack) | (stack >= max_depth)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def _table_candidates(depth, intr, T, table_z: float, band: float, envelope,
                      step: int = 4) -> np.ndarray:
    """
    It takes the depth image and the current transform and gives you, in base
    frame, the points that could be table: inside the workspace in x and y, and
    within `band` of table_z.

    Every `step`-th pixel only. Thousands of points are plenty for one plane.
    """
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[v, u]
    ok = np.isfinite(z)
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok].astype(float)
    p_cam = np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], axis=1)
    p = p_cam @ T[:3, :3].T + T[:3, 3]
    x, y = p[:, 0], p[:, 1]
    inside = envelope.contains_xy(x, y) & (np.abs(p[:, 2] - table_z) < band)
    return p[inside]


if __name__ == "__main__":
    sys.exit(main())

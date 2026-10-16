"""place_and_look.py

It takes an object held in the gripper and gives you the miss Architecture A
would make, measured on the table with A's own perception code, and (unless
--dry-run) `T_base_cam.json` shifted to remove it.

    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/place_and_look.py --dry-run
    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/place_and_look.py

The lerobot venv, because it runs OWLv2. Start with the arm AT HOME
(go_home.py): home is where A looks from, so it is where this looks from.

THIS MOVES THE ARM and places and picks the object several times. Clear the
table apart from the object, keep a hand on the e-stop.

## Why (13 Oct 2026)

The ball fit measures the ball HELD IN THE FINGERS, in the air. A measures
objects SITTING ON THE TABLE with nothing in the way. Those are two different
measurements, and whatever differs between them becomes a constant error the
fit cannot see. After the fit and `level_to_table.py`, heights were right but
three picks in a row closed on nothing, every time BEHIND the orange as the
camera sees it.

So this measures the error the way it actually happens. The arm puts the object
down at a spot it knows exactly, goes home, and A's perception (OWLv2, depth,
centre push, transform, the same functions the pipeline calls) says where it
thinks the object is. The difference is A's miss at that spot. After several
spots the mean miss is removed from the transform as a sideways shift.

Height is not touched: the table levelling already owns z. The object's
sideways position under the TCP is taken as exact, which is true to the few mm
the fingers centre it.

## Reading the output

Each spot prints the miss in base x and y. If they all point the same way and
agree to within about a centimetre, a shift is the right fix and the line
`spread after the shift` will be small. If they point in different directions,
the error turns or scales across the table, and a shift is not enough.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    p.add_argument("--note", default="", help="free text kept in the file")
    args = p.parse_args()

    from PIL import Image

    from fyp.architecture_a.calibration.store import CalibrationStore
    from fyp.architecture_a.perception.detector import detect
    from fyp.architecture_a.perception.filters import apply_filters, to_json_records
    from fyp.architecture_a.perception.localiser import localise
    from fyp.architecture_a.planner import describe_objects
    from fyp.architecture_a.skills import GRIPPER_OPEN, Skills
    from fyp.shared.hardware.camera import RealSenseCamera
    from fyp.shared.hardware.safety import WorkspaceEnvelope
    from fyp.shared.hardware.ur5e_controller import URController
    from fyp.shared.helpers.config import get_config

    cfg = get_config()
    a = cfg["architecture_a"]
    cc = a["calibration"]
    env = WorkspaceEnvelope(cfg)
    store = CalibrationStore(cfg)
    T, _ = store.current()
    query = str(cc["ball_query"])
    # The object's centre above the fingertips, as the ball fit uses it. For
    # an object resting on the table that is also its centre above the table.
    half_h = -float(cc["ball_offset_tool_m"][2])
    table_z = float(cc["table_z"])

    spots = _spots(cc["grid_centre_xy"], float(cc["grid_half_extent_m"]) * 0.75,
                   int(cc["place_look_n"]), env)
    print(f"{len(spots)} spots on the table, object '{query}'")
    for i, s in enumerate(spots, 1):
        print(f"  {i}. ({s[0]:.3f}, {s[1]:.3f})")

    ctrl = URController()
    skills = Skills(ctrl, cfg, env)
    misses = []
    try:
        ctrl.gripper_start()
        park = list(ctrl.get_state()["tcp_pose"])
        if not _near_home(ctrl, cfg):
            print("the arm is not at home. Run go_home.py first: home is where A looks "
                  "from, so it is where this has to look from.", file=sys.stderr)
            return 1
        ctrl.gripper_toggle(GRIPPER_OPEN)
        input(f"\nPut the {query} between the fingers, bottom flush with the fingertips, "
              "and press Enter. The fingers will close.")
        width = float(ctrl.gripper_toggle(0))
        if width < float(a["skills"]["min_grasp_width_mm"]):
            print("nothing held, stopping", file=sys.stderr)
            return 1

        with RealSenseCamera(cfg) as cam:
            for i, (x, y) in enumerate(spots, 1):
                print(f"\n  {i}/{len(spots)} ({x:.3f}, {y:.3f})", end="  ", flush=True)
                truth = np.array([x, y, table_z + half_h])
                r = skills.place([x, y, env.z[0]], release_clearance=0.005)
                if not r.ok:
                    print(f"place failed: {r.message}. Stopping here.")
                    break
                ctrl.move_to_pose(env.check_pose(park, "park"), *env.clamp_motion(
                    float(a["skills"]["speed"]), float(a["skills"]["acc"])))
                time.sleep(float(cc["settle_s"]))

                seen = _perceive(cam, query, T, a, cfg, detect, apply_filters,
                                 to_json_records, localise, describe_objects, Image)
                if seen is None:
                    print("not detected", end="")
                else:
                    miss = seen[:2] - truth[:2]
                    misses.append((truth, seen))
                    print(f"A sees ({seen[0]:.3f}, {seen[1]:.3f}, {seen[2]:.3f})  "
                          f"miss dx {miss[0] * 1000:+.0f} dy {miss[1] * 1000:+.0f} mm  "
                          f"(|{np.linalg.norm(miss) * 1000:.0f}| mm)", end="")

                # Pick back up from where it was PUT, not where it was seen. The
                # put position is the one thing known exactly.
                r = skills.pick(truth)
                if not r.ok:
                    print(f"\n  could not pick it back up: {r.message}. Stopping here.")
                    break
        ctrl.move_to_pose(env.check_pose(park, "park"), *env.clamp_motion(
            float(a["skills"]["speed"]), float(a["skills"]["acc"])))
        input(f"\n\nHold the {query} and press Enter. The fingers will open.")
        ctrl.gripper_toggle(GRIPPER_OPEN)
    finally:
        ctrl.close()

    if len(misses) < 3:
        print(f"\nonly {len(misses)} spots measured, need 3. Nothing written.", file=sys.stderr)
        return 1

    truth = np.array([t for t, _ in misses])
    seen = np.array([s for _, s in misses])
    miss = seen[:, :2] - truth[:, :2]
    shift = -miss.mean(axis=0)
    after = miss + shift
    print(f"\n{'=' * 70}\n  {len(misses)} spots, localiser "
          f"{a['perception']['localiser']}")
    print(f"  mean miss       dx {-shift[0] * 1000:+.1f}  dy {-shift[1] * 1000:+.1f} mm  "
          f"(|{np.linalg.norm(shift) * 1000:.1f}| mm)")
    print(f"  spread after the shift  RMS {np.sqrt((after ** 2).sum(1).mean()) * 1000:.1f} mm, "
          f"worst {np.linalg.norm(after, axis=1).max() * 1000:.1f} mm")
    print(f"  heights A saw   {seen[:, 2].min():.3f} to {seen[:, 2].max():.3f} "
          f"(true centre {table_z + half_h:.3f})")

    if args.dry_run:
        print("\n  dry run, nothing written")
        return 0

    T_new = T.copy()
    T_new[0, 3] += shift[0]
    T_new[1, 3] += shift[1]
    archived = store.replace(T_new, "place_and_look", {
        "shift_mm": [round(float(v) * 1000, 1) for v in shift],
        "spread_after_rms_mm": round(float(np.sqrt((after ** 2).sum(1).mean()) * 1000), 1),
        "n_spots": len(misses),
        # A shift is only right for the localiser it was measured with.
        "localiser": a["perception"]["localiser"],
        "note": args.note,
    })
    print(f"\n  previous calibration kept as {archived.name}\n  wrote {store.path}")
    return 0


def _spots(centre_xy, half: float, n: int, env) -> list[tuple[float, float]]:
    """It takes the grid settings and gives you table spots inside the reach, with margin."""
    cx, cy = (float(v) for v in centre_xy)
    out = []
    for x in np.linspace(cx - half, cx + half, n):
        for y in np.linspace(cy - half, cy + half, n):
            if (np.hypot(x, y) <= env.max_radius - 0.03
                    and env.contains([x, y, env.z[0] + 0.15])):
                out.append((round(float(x), 3), round(float(y), 3)))
    return out


def _near_home(ctrl, cfg) -> bool:
    """It takes the controller and gives you whether every joint is within 2 deg of home."""
    q = np.asarray(ctrl.get_state()["joint_pos"], dtype=float)
    home = np.asarray(cfg["robot"]["home_q"], dtype=float)
    return bool(np.max(np.abs(np.degrees(q - home))) < 2.0)


def _perceive(cam, query, T, a, cfg, detect, apply_filters, to_json_records, localise,
              describe_objects, Image):
    """
    It takes the camera and gives you where Architecture A puts the object in
    base frame, or None. The same calls, in the same order, with the same
    settings as `pipeline.perceive`.
    """
    frame = cam.capture(depth_frames=int(a["perception"]["depth_frames"]))   # as pipeline.perceive
    raw = detect(Image.fromarray(frame.rgb), [query], a["detector"]["model"])
    kept, _ = apply_filters(raw, threshold=float(a["detector"]["threshold"]),
                            nms_iou=float(a["detector"]["nms_iou"]), keep_top1=True)
    if not kept:
        return None
    # localise() is the pipeline's own call, so the method in config is measured.
    located = localise(to_json_records(kept), frame, T, cfg)
    valid = [o for o in located if o.valid]
    if not valid:
        return None
    return np.asarray(describe_objects(valid, T)[0]["xyz"], dtype=float)


if __name__ == "__main__":
    sys.exit(main())

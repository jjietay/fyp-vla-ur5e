"""write_calibration.py

It takes the mount measurements you made with a rule and gives you
`config/calibration/T_base_cam.json`, the file every later run loads.

    uv run python scripts/write_calibration.py \
        --xyz 0.62 -0.31 0.74 --rpy 0 45 180

That file holds one 4x4 matrix, `T_base_cam`, which converts a point the camera
saw into a point the arm can reach for. Stage 3 of Architecture A gives a
camera-frame XYZ, the pick and place primitives only speak base frame, and this
transform is the whole of the bridge between them. Without it the detector can
see an object perfectly and the arm has no idea where it is.

The camera stands at roughly 45 degrees to the side, taped in place because the
lab is shared, so the transform holds for one session. Re-measure and re-run this
each time the camera is set up again. `Calibration Marker` closed on no marker,
which means these six numbers are MEASURED rather than fitted, and this script
is that decision made runnable.

What the six numbers are:

  --xyz   where the camera's OPTICAL ORIGIN sits in the robot base frame, metres
  --rpy   how the camera is turned relative to the base frame, degrees

A measured transform reports no residual of its own, so nothing here can tell you
whether the numbers are right. `check_calibration.py` is the only thing that can,
and it is not optional.

Two ways to get this wrong that cost a day each:

  Measure to the case front rather than the optical origin. The RealSense colour
  sensor sits a few millimetres inside the housing, and a datum error is pure
  bias on every point forever.

  Confuse the base frame with a world frame. There is no world frame on the real
  cell. `get_state` reports the TCP in base frame and `moveL` expects targets in
  it, so measure against the arm base, not against the table or the room.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xyz", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
                   help="camera OPTICAL ORIGIN in the robot base frame, metres")
    p.add_argument("--rpy", nargs=3, type=float, required=True,
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="camera orientation relative to the base frame, degrees by default")
    p.add_argument("--radians", action="store_true",
                   help="read --rpy as radians instead of degrees")
    p.add_argument("--out", default=None,
                   help="output path; defaults to the calibration_dir in config.yaml")
    p.add_argument("--note", default="",
                   help="free text kept in the file, e.g. which bench and which mount")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing calibration")
    args = p.parse_args()

    from fyp.architecture_a.calibration.store import CalibrationStore
    from fyp.shared.helpers.config import get_config
    from fyp.shared.helpers.rotations import euler_to_R

    store = CalibrationStore(get_config(), args.out)
    out = store.path

    # Refusing rather than overwriting, because you must be able to say which
    # calibration produced which result, and a silently replaced file makes every
    # earlier number in the logbook unattributable.
    if store.exists() and not args.force:
        print(f"{out} already exists. Pass --force if you have just "
              f"re-measured the mount. The old file is kept as T_base_cam.<time>.json.", file=sys.stderr)
        return 1

    rpy = np.asarray(args.rpy, dtype=float)
    rpy_rad = rpy if args.radians else np.deg2rad(rpy)

    T = np.eye(4)
    T[:3, :3] = euler_to_R(rpy_rad)
    T[:3, 3] = np.asarray(args.xyz, dtype=float)

    store.replace(T, "measured_mount", restart=True, meta={
        "method": "measured mount geometry, no marker",
        "xyz_m": [float(v) for v in args.xyz],
        "rpy_deg": [float(v) for v in (rpy if not args.radians else np.rad2deg(rpy))],
        "euler_convention": "intrinsic Z-Y-X, matching helpers.rotations.euler_to_R",
        "note": args.note,
    })

    print(f"wrote {out}")
    print(np.array2string(T, precision=4, suppress_small=True))
    print("\nNot validated. A measured transform has no residual of its own, so "
          "this file is a guess until you run:")
    print("    uv run python scripts/check_calibration.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

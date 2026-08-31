"""check_camera.py

It takes nothing and gives you proof that the RealSense works, plus the numbers
you need before calibrating.

    uv run python scripts/check_camera.py

Run this FIRST at the bench, before the arm, before calibration, before anything
else that involves the camera. Every later stage assumes frames arrive and that
the intrinsics are right, and each of them fails in a way that looks like a
different problem when the camera is the real cause.

What it reports, and why each one is on the list:

  the device model and serial     so a swapped camera in a shared lab is obvious
  the USB descriptor              a USB 2 cable enumerates fine and then silently
                                  caps resolution and frame rate. This is the
                                  single most likely thing to waste a lab session
  the negotiated stream format    what you asked for in config.yaml against what
                                  the camera actually agreed to
  the intrinsics from the SDK     fx, fy, cx, cy at THIS resolution. They change
                                  with resolution, so a hardcoded set is wrong
  the depth scale                 the metres-per-unit the driver reports
  depth coverage                  the share of pixels with a valid reading, which
                                  is how you find out the scene is too dark, too
                                  shiny, or too close
  the working distance            min, median and max depth in view, checked
                                  against the D435i's 0.3 m minimum

It saves a colour frame and a colourised depth frame into the cache directory so
you can see what the camera sees, and so `check_detector.py` has an image to run
against without needing the camera attached again.

Nothing here moves the arm and nothing here needs a calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MIN_Z_M = 0.3   # D435i datasheet minimum depth. Closer than this reads as noise.


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=30,
                   help="frames to discard before measuring, letting auto-exposure settle")
    p.add_argument("--prefix", default="frame_top",
                   help="basename for the saved images, in the cache directory")
    args = p.parse_args()

    from fyp.shared.helpers.config import get_config, resolve

    cfg = get_config()
    cache = resolve(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    if not _report_device(cfg):
        return 1

    from fyp.shared.hardware.camera import RealSenseCamera

    print("\nstarting the stream ...")
    with RealSenseCamera(cfg) as cam:
        # Auto-exposure needs a few frames. Measuring the first one reports a
        # dark image and a depth map full of holes, and blames the scene.
        for _ in range(max(0, args.frames)):
            cam.capture()
        frame = cam.capture()

    _report_frame(frame, cfg)
    _report_backprojection(frame)
    _save(frame, cache, args.prefix)
    return 0


def _report_device(cfg) -> bool:
    """
    It takes the config and gives you whether a camera is present, printing what
    it is and how it is plugged in.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not installed. Run: uv sync", file=sys.stderr)
        return False

    devices = list(rs.context().devices)
    if not devices:
        print("NO REALSENSE FOUND.\n"
              "  check the cable is in a USB 3 port, both ends\n"
              "  on Ubuntu you may need the librealsense udev rules, or the device\n"
              "  is found and no frames arrive\n"
              "  `lsusb | grep -i intel` says whether the OS sees it at all",
              file=sys.stderr)
        return False

    print(f"{len(devices)} device(s) found")
    for d in devices:
        name = _info(d, rs.camera_info.name)
        serial = _info(d, rs.camera_info.serial_number)
        usb = _info(d, rs.camera_info.usb_type_descriptor)
        print(f"  {name}  serial {serial}  firmware {_info(d, rs.camera_info.firmware_version)}")
        print(f"  USB {usb}")
        # The quiet killer. A USB 2 cable connects, streams, and caps what you
        # can negotiate, and the symptom is a resolution you did not ask for.
        if usb.startswith("2"):
            print("  WARNING: this is a USB 2 connection. Resolution and frame rate\n"
                  "           will be capped silently. Change the cable or the port\n"
                  "           before trusting anything below.")
    return True


def _report_frame(frame, cfg) -> None:
    """It takes one captured frame and gives you the numbers worth writing down."""
    rgb, depth, intr = frame.rgb, frame.depth, frame.intrinsics
    want = cfg["camera"]

    print(f"\nasked for {want['width']}x{want['height']} at {want['fps']} fps")
    print(f"got       {rgb.shape[1]}x{rgb.shape[0]} colour, {depth.shape[1]}x{depth.shape[0]} depth")
    if (rgb.shape[1], rgb.shape[0]) != (want["width"], want["height"]):
        print("  MISMATCH: the camera did not agree to the requested format. On a\n"
              "            USB 2 link this is the expected outcome.")
    if rgb.shape[:2] != depth.shape[:2]:
        print("  MISMATCH: colour and depth differ in size, so alignment is not on.\n"
              "            Every back-projected point will land centimetres away.")

    print(f"\nintrinsics at this resolution, from the SDK:")
    print(f"  fx={intr.fx:.2f}  fy={intr.fy:.2f}  cx={intr.cx:.2f}  cy={intr.cy:.2f}")

    valid = depth[np.isfinite(depth) & (depth > 0)]
    coverage = 100.0 * valid.size / depth.size if depth.size else 0.0
    print(f"\ndepth coverage {coverage:.1f}% of pixels")
    if coverage < 50:
        print("  LOW. Usually a dark, shiny or transparent scene, or a surface closer\n"
              "  than the sensor's minimum. Matte objects read far better.")

    if valid.size:
        print(f"  range {valid.min():.3f} to {valid.max():.3f} m, median {np.median(valid):.3f} m")
        if valid.min() < MIN_Z_M:
            print(f"  NOTE: readings closer than {MIN_Z_M} m are below the D435i minimum\n"
                  f"        and are noise. Move the camera back, or expect that region\n"
                  f"        of the table to be unusable.")
    else:
        print("  no valid depth anywhere. Check the scene is lit and not a blank wall.")


def _report_backprojection(frame) -> None:
    """
    It takes a frame and gives you the centre pixel as a 3D point, which is how
    you confirm the axis convention at the bench.

    This exists because the convention was wrong until 31 Aug 2026: the maths
    used MuJoCo's axes while the camera reports OpenCV's. It was invisible,
    because the round trip between pixel and point negated the error twice. The
    only way to see it is to look at an actual number from an actual camera.
    """
    from fyp.architecture_a.perception.pixel_to_3d import depth_at, pixel_to_camera

    u, v = frame.intrinsics.cx, frame.intrinsics.cy
    z = depth_at(frame.depth, u, v, radius=2)
    if not np.isfinite(z) or z <= 0:
        print("\nno depth at the image centre, so the axis check is skipped. "
              "Aim the camera at something matte and run again.")
        return

    p = pixel_to_camera(u, v, z, frame.intrinsics)
    print(f"\ncentre pixel back-projects to camera-frame {np.round(p, 4).tolist()} m")
    if p[2] > 0:
        print("  z is positive, so the axis convention is right (OpenCV, "
              "z forward). This is what you want.")
    else:
        print("  z is NEGATIVE. The OpenGL convention has come back into "
              "pixel_to_3d.py, and every calibrated point will land behind the\n"
              "  camera. Stop and read A12 in Open Actions before calibrating.")


def _save(frame, cache: Path, prefix: str) -> None:
    """It takes a frame and gives you two PNGs you can look at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour_path = cache / f"{prefix}.png"
    depth_path = cache / f"{prefix}_depth.png"

    plt.imsave(colour_path, frame.rgb)

    d = frame.depth.copy()
    d[~np.isfinite(d)] = 0.0
    plt.imsave(depth_path, np.ma.masked_where(d <= 0, d), cmap="viridis")

    print(f"\nsaved {colour_path}")
    print(f"saved {depth_path}")
    print("\nThe colour frame is what check_detector.py wants:")
    print(f"    --image {colour_path}")


def _info(device, field) -> str:
    """It takes a device and a field and gives you the string, or a placeholder."""
    try:
        return device.get_info(field)
    except Exception:
        return "?"


if __name__ == "__main__":
    sys.exit(main())

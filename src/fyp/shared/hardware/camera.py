"""camera.py

It takes nothing and gives you aligned colour and depth frames plus the real
camera intrinsics, from the fixed RealSense looking at the table.

Shared, because both architectures need images. Architecture A additionally
reads the depth channel to back-project a detection into 3D; Architecture B
consumes only the colour frame, because SmolVLA has no depth input. Same
hardware, same driver, different amount of it used.

Two things here are not optional:

  Depth is aligned to colour. Without alignment the pixel a detector found in
  the colour image indexes a different point in the depth image, and every
  object lands a few centimetres from where it really is. The error is small
  enough to look like bad calibration and waste a day.

  Intrinsics are read from the device, never hardcoded. They change with
  resolution, and a mismatched principal point tilts the whole workspace.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fyp.shared.hardware.intrinsics import CameraIntrinsics
from fyp.shared.helpers.config import get_config


@dataclass(frozen=True)
class Frame:
    """One synchronised capture: colour, depth in metres, and the intrinsics they share."""
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, metres, 0.0 where invalid
    intrinsics: CameraIntrinsics


class RealSenseCamera:
    """
    It takes the camera block from config and gives you a frame source.

    Use as a context manager so the pipeline cannot leave the device streaming
    after a crash:

        with RealSenseCamera() as cam:
            frame = cam.capture()
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or get_config()
        cam = cfg["camera"]
        self.width = int(cam["width"])
        self.height = int(cam["height"])
        self.fps = int(cam["fps"])
        self.align_depth = bool(cam["align_depth_to_color"])
        self._pipeline = None
        self._align = None
        self._depth_scale = None

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        """It takes nothing and gives you a streaming device, or a clear error if there is none."""
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 is not installed. `uv add pyrealsense2`, and note it "
                "needs the librealsense udev rules on Linux or the device will "
                "enumerate but refuse to stream."
            ) from e

        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._align = rs.align(rs.stream.color) if self.align_depth else None

        # The first few frames come out before auto-exposure settles and are
        # visibly darker. Detecting on one costs you a false negative that looks
        # like a detector problem.
        for _ in range(10):
            self._pipeline.wait_for_frames()

    def stop(self) -> None:
        """It takes nothing and gives you a released device. Safe to call twice."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def capture(self) -> Frame:
        """
        It takes nothing and gives you one synchronised Frame.

        Depth is converted to metres here rather than left in raw units, so that
        nothing downstream has to remember the device scale factor.
        """
        if self._pipeline is None:
            raise RuntimeError("camera is not started; call start() or use it as a context manager")

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("incomplete frameset from the camera")

        bgr = np.asanyarray(color_frame.get_data())
        rgb = bgr[:, :, ::-1].copy()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale

        i = color_frame.profile.as_video_stream_profile().intrinsics
        intr = CameraIntrinsics(fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy,
                                width=i.width, height=i.height)
        return Frame(rgb=rgb, depth=depth, intrinsics=intr)

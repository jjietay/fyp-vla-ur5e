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
        # Depth has its own size (13 Oct 2026): 848x480 is Intel's recommended
        # depth resolution for the D435. Colour stays at width x height, and the
        # depth is aligned (resampled) onto it, so every consumer still gets
        # colour-sized depth on colour pixels and nothing downstream changes.
        self.depth_width = int(cam.get("depth_width", self.width))
        self.depth_height = int(cam.get("depth_height", self.height))
        self.fps = int(cam["fps"])
        self.align_depth = bool(cam["align_depth_to_color"])
        self.cfg_depth = {"visual_preset": cam.get("visual_preset"),
                          "laser_power_mw": cam.get("laser_power_mw")}
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
        cfg.enable_stream(rs.stream.depth, self.depth_width, self.depth_height,
                          rs.format.z16, self.fps)

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = sensor.get_depth_scale()
        self._set_depth_options(sensor, rs)
        self._align = rs.align(rs.stream.color) if self.align_depth else None

        # The first few frames come out before auto-exposure settles and are
        # visibly darker. Detecting on one costs you a false negative that looks
        # like a detector problem.
        for _ in range(10):
            self._pipeline.wait_for_frames()

    def _set_depth_options(self, sensor, rs) -> None:
        """
        It takes the depth sensor and gives it the preset and laser power from
        config, every time the stream starts.

        A RealSense keeps these settings until it is unplugged, so a camera that
        never sets them runs with whatever the last program left behind. On 13
        Oct 2026 a diagnostic left "High Accuracy" on, which throws away any depth
        it is unsure of: the next run lost the whole top of a white box and half
        the table, and the box was dropped for having no depth.
        """
        preset = self.cfg_depth.get("visual_preset")
        if preset and sensor.supports(rs.option.visual_preset):
            r = sensor.get_option_range(rs.option.visual_preset)
            names = {sensor.get_option_value_description(rs.option.visual_preset, i): i
                     for i in range(int(r.min), int(r.max) + 1)}
            if preset not in names:
                raise RuntimeError(f"camera.visual_preset {preset!r} is not one of "
                                   f"{sorted(names)}")
            sensor.set_option(rs.option.visual_preset, names[preset])
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1)
        power = self.cfg_depth.get("laser_power_mw")
        if power is not None and sensor.supports(rs.option.laser_power):
            sensor.set_option(rs.option.laser_power, float(power))

    def stop(self) -> None:
        """It takes nothing and gives you a released device. Safe to call twice."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def capture(self, depth_frames: int = 1) -> Frame:
        """
        It takes how many frames to average the depth over, and gives you one
        synchronised Frame.

        Depth is converted to metres here rather than left in raw units, so that
        nothing downstream has to remember the device scale factor.

        `depth_frames` > 1 takes that many framesets back to back and gives each
        pixel the MEDIAN of its valid readings (13 Oct 2026). That lowers the
        frame-to-frame noise; it does nothing for a reading that is wrong the same
        way every frame. Colour is the last frame's. Only for a still scene: 5
        frames is about 0.17 s at 30 fps, so Architecture A uses it while the arm
        is at rest, and anything that streams (Architecture B) keeps the default 1.
        """
        if self._pipeline is None:
            raise RuntimeError("camera is not started; call start() or use it as a context manager")

        depths = []
        for _ in range(max(1, int(depth_frames))):
            frames = self._pipeline.wait_for_frames()
            if self._align is not None:
                frames = self._align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("incomplete frameset from the camera")
            depths.append(np.asanyarray(depth_frame.get_data()).astype(np.float32)
                          * self._depth_scale)

        bgr = np.asanyarray(color_frame.get_data())
        rgb = bgr[:, :, ::-1].copy()
        depth = median_depth(depths)

        i = color_frame.profile.as_video_stream_profile().intrinsics
        intr = CameraIntrinsics(fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy,
                                width=i.width, height=i.height)
        return Frame(rgb=rgb, depth=depth, intrinsics=intr)


def median_depth(depths: list[np.ndarray]) -> np.ndarray:
    """
    It takes depth images of the same still scene and gives you one, where each
    pixel is the median of its valid (above 0) readings, and 0 where it had none.

    Invalid readings are left out rather than counted as 0, or a pixel that
    drops out in two of five frames would be pulled towards the camera.
    """
    if len(depths) == 1:
        return depths[0]
    stack = np.stack(depths).astype(np.float32)
    stack[~(stack > 0)] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN pixel is expected
        out = np.nanmedian(stack, axis=0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


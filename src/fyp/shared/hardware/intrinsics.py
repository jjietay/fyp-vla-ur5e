"""intrinsics.py

It takes nothing and gives you the container that describes a camera's geometry.

Lives in `shared/` rather than with Architecture A's perception because a Frame
carries its intrinsics, and both architectures receive Frames. Architecture B
ignores them; Architecture A back-projects with them. The maths that uses this
(pixel_to_camera, depth_at) is Architecture A only and stays there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    This class contains the variables of the Camera's Intrinsic properties.
    
    1) fx/fy
        - represents focal length which is the distance between lens/pinhole and the sensor
        - scaled by pixel size
        - affects fov, and magnification
    
    2) cx/cy
        - reprsents principal point
        - it is the pixel coordinate where the central optical axis hits the image sensor
    
    3) width/height
        - this is the image sizes

    4) Camera Intrinsic Matrix, K
        - this is represented as:
            [fx   0  cx]
            [0   fy  cy]
            [0    0   1]
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    def __repr__(self) -> str:
        return (f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")

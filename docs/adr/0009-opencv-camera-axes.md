# pixel_to_3d.py uses the OpenCV camera frame: x right, y down, z forward positive

Fixed 31 Aug 2026 to match `pyrealsense2` and its intrinsics. The code had carried MuJoCo's OpenGL axes (y and z negated) from the simulation, and the round trip through `camera_to_pixel` hid it. A measured transform cannot absorb an axis flip, so every point would have landed mirrored and behind the camera. `tests/unit/test_perception.py` pins the signs and `check_camera.py` reports whether z is positive; negative z means the convention has come back, and the fix is never to negate downstream.

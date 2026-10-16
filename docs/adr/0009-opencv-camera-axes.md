# pixel_to_3d.py uses the OpenCV camera frame: x right, y down, z forward positive

Fixed 31 Aug 2026. That is what `pyrealsense2` deprojects into and what the SDK intrinsics assume.

Before 31 Aug the code negated y and z, which is the OpenGL convention MuJoCo uses, inherited from the simulation this project started in. Nothing caught it, because `camera_to_pixel` negated them back and the round trip stayed exact.

It would not have stayed harmless. A fitted `T_base_cam` absorbs a fixed axis flip silently, because Kabsch finds whatever map takes one cloud to the other. A measured transform cannot absorb a flip that is not physically there, so every back projected point would have landed mirrored and behind the camera, reading as a badly wrong calibration.

## Consequences

Two guards exist, because the original bug was invisible for months. `tests/unit/test_perception.py` pins the signs directly, and `check_camera.py` prints a real back projected point and says whether z is positive. If a future session sees negative z out of `pixel_to_camera`, the OpenGL convention has come back, and the fix is not to negate downstream.

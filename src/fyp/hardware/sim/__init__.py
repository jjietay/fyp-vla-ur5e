"""MuJoCo development substrate. NOT a deliverable.

Everything in this package exists to let the pipeline be developed and
diagnosed before the real UR5e is available. When hardware lands, this whole
directory is deleted and `hardware/ur5e_controller.py` +
`hardware/realsense_camera.py` take over.

What does NOT survive the move, and why:
  - mujoco_controller.py : the real arm is driven over RTDE
  - renderer.py          : intrinsics come from calibration, not from fovy
  - scene.py             : ground truth read from a model has no real-world analogue
  - server.py/client.py  : real teleop is a different input device entirely
"""

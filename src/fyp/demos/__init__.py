"""Demonstration capture and export — the data path for Architecture B.

Deliberately backend-agnostic: the recorder is handed state by whatever is
driving the robot (sim teleop today, a real UR5e later) and never reads the
robot itself. That is why the same HDF5 schema, and therefore the same LeRobot
export, works for both.
"""

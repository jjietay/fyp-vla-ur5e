"""The data path: teleoperated demonstrations into a trainable dataset.

Deliberately backend-agnostic. The recorder is handed state by whatever is
driving the robot and never reads the robot itself, so the same HDF5 schema and
the same LeRobot export work regardless of the teleop device.

Two things that must hold, or the comparison is unfair:

  action space   must match what Architecture A's skills command. If A commands
                 TCP poses and B is trained on joint positions, the two are not
                 solving the same problem.
  instruction    varies per episode, drawn from a paraphrase set rather than one
                 fixed template, so phrasing robustness is testable afterwards.
"""

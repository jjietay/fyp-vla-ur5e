"""The physical robot cell: arm, camera, gripper.

`ur5e_controller.py` is the real backend. `sim/` is the MuJoCo development
substrate and is deliberately a subfolder, not a peer: the sim is not a
deliverable, and the whole directory is deleted once real hardware is
available. Both backends expose the same public API — move_joints,
move_to_pose, gripper_toggle, get_state, close — so downstream code does not
know which one it is holding.
"""

"""The physical robot cell: arm, cameras, gripper.

`ur5e_controller.py` drives the real arm over RTDE and is the ONLY path to the
robot. Both architectures command through it: Architecture A from its scripted
skills, Architecture B from its predicted action chunks.

That single shared surface is deliberate. If the two architectures reached the
arm by different routes, any difference in measured success could be the route
rather than the policy, and the comparison would be worthless.

The MuJoCo backend that used to live here as `sim/` was quarantined to
`simulation/fyp_sim/` on 11 Aug 2026. Nothing in this package imports mujoco.
"""

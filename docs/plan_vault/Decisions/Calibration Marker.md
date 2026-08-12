---
status: decided
decided: 2026-08-12
needed_by: W3
priority: high
---

# Calibration Marker

**No marker. `T_base_cam` comes from the measured mount geometry instead.** Decided 12 Aug 2026, closing a decision that had been open since July.

## The decision

The fixed RGB-D camera is bolted at a known offset from the arm base, so `T_base_cam` is one constant transform. Rather than recover it by observing a target, it gets written directly from the measured mount geometry into `config/calibration/T_base_cam.json`.

Nothing in the pipeline changes. `hand_eye.py` already loads the transform from JSON and never cares how the numbers got there.

## Why no marker

Two reasons, and the second is the one that decided it.

* the mount is rigid and the transform is genuinely constant, so there is a geometric answer available without observing anything
* Prof Cheah's view is that a printed board in the cell looks out of place against a real deployment, and the project is arguing for an interface a person could actually use

## The cost, recorded honestly

`T_base_cam` is six numbers, three translation and three rotation, and the rotation is the exposed one. At a 1 m standoff, **1 degree of camera orientation error displaces every back projected point by about 17 mm**. Holding the 5 mm target in [[Risks]] would need the mount known to better than 0.3 degrees, which a bracket and a rule will not give.

So the realistic expectation is a systematic offset of roughly 1 to 3 cm, in one consistent direction rather than randomly per object. Grasp tolerance is what absorbs it.

This has to be measured, not assumed, which is why validation in [[W3 Camera and Calibration]] survives the decision and matters more than it did before. A hand written transform has no residual of its own, so the reach test against known points is the only number there is.

## Two things that make it work better

* measure to the camera's optical origin, not the case front, since the RealSense colour frame sits a few millimetres inside the housing and a datum error here is pure bias
* re measure after any knock, because there is no fit to notice that the camera moved

## The fallback if the offset is too large to grasp with

Ball on the gripper, Kabsch, no fiducial. A matte sphere of known radius held in the jaws, the arm driven to 15 to 20 poses at fixed down orientation, sphere centres paired against `tcp_pose` and fed to the solver that is already written and tested. Tag free, so the objection above still holds, and it reuses `depth_at`, `pixel_to_camera`, `surface_to_centroid`, `solve_rigid_transform` and `report` without new maths.

Kept as rung two rather than the plan, because the measured transform may well be good enough and costs an hour instead of an afternoon.

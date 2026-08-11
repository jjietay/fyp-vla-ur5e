---
status: blocked
needs_lab: true
week: 4
---

# W3 Camera and Calibration

The single biggest unblocker in the project. Nothing downstream can address a real object without it.

## Steps

- [ ] real camera driver behind the same interface the MuJoCo renderer satisfies, with real intrinsics from the SDK replacing `intrinsics_from_fovy`
- [ ] decide the marker, see [[Calibration Marker]], and stop deferring it
- [ ] collection script: move the arm to 15 to 20 poses across the workspace, recording TCP pose and marker centroid at each
- [ ] solve and save `T_base_cam` using the existing `solve_rigid_transform` and `save`
- [ ] validate against a measured physical point and **report residual RMS in mm**

## The trap

The arm base carries `quat="0 0 0 -1"`, so **base is not world**. A world frame target fed through a base frame transform looks entirely plausible and is wrong. This is documented in the module header and is still the easiest way to lose a day.

## Why validation is not optional

A calibration you did not check numerically is a calibration you do not have, and every downstream failure will get blamed on it. If the residual exceeds about 5 mm, collect more poses before moving on, per [[Risks]].

The solver is already written and tested. Only the procedure is missing.

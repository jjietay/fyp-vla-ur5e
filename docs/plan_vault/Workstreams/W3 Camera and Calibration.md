---
status: driver written, rest blocked on lab access
needs_lab: true
week: 4
---

# W3 Camera and Calibration

The single biggest unblocker in the project. Nothing downstream can address a real object without it.

## Steps

- [x] real camera driver, with intrinsics read straight from the RealSense SDK, in `shared/hardware/camera.py`
- [ ] decide the marker, see [[Calibration Marker]], and stop deferring it
- [ ] collection script: move the arm to 15 to 20 poses across the workspace, recording TCP pose and marker centroid at each
- [ ] solve and save `T_base_cam` using the existing `solve_rigid_transform` and `save`
- [ ] validate against a measured physical point and **report residual RMS in mm**

## The trap

The arm base carries `quat="0 0 0 -1"`, so **base is not world**. A world frame target fed through a base frame transform looks entirely plausible and is wrong. This is documented in the module header and is still the easiest way to lose a day.

## Why validation is not optional

A calibration you did not check numerically is a calibration you do not have, and every downstream failure will get blamed on it. If the residual exceeds about 5 mm, collect more poses before moving on, per [[Risks]].

The solver is already written and tested. Only the procedure is missing.

## Status 11 Aug 2026

The camera driver is written but has never seen a device. It is in `shared/` rather than with Architecture A, because both architectures need frames: A reads the depth channel, B uses only the colour one.

Two things in it are not optional and are easy to get wrong later:

* depth is aligned to colour, or the pixel the detector found indexes a different point in the depth image and every object lands centimetres off
* intrinsics come from the SDK at runtime, never hardcoded, because they change with resolution

Everything else here still waits on lab access, and [[Calibration Marker]] still blocks the collection script.

---
status: driver written, rest blocked on lab access
needs_lab: true
week: 4
---

# W3 Camera and Calibration

The single biggest unblocker in the project. Nothing downstream can address a real object without it.

## Steps

- [x] real camera driver, with intrinsics read straight from the RealSense SDK, in `shared/hardware/camera.py`
- [x] decide the marker, see [[Calibration Marker]]: **no marker, the transform is measured from the mount**
- [ ] measure the mount: camera optical origin relative to the arm base, translation and orientation both
- [ ] write `config/calibration/T_base_cam.json` from those numbers, in the format `hand_eye.save` produces
- [ ] validate: reach for 5 or more known points across the workspace and **report the error in mm**

## The trap

The arm base carries `quat="0 0 0 -1"`, so **base is not world**. A world frame target fed through a base frame transform looks entirely plausible and is wrong. This is documented in the module header and is still the easiest way to lose a day.

## Why validation is not optional

A calibration you did not check numerically is a calibration you do not have, and every downstream failure will get blamed on it.

This got sharper once the transform became hand measured rather than fitted. A fit reports its own residual and tells you when it is wrong. A measured transform reports nothing, so the reach test against known points is the **only** number in the project that says whether stage 4 works at all. Expect a systematic offset in one direction rather than scatter. If it is too large to grasp with, take the fallback in [[Calibration Marker]].

The solver is already written and tested and stays in the repo for exactly that fallback.

## Status 11 Aug 2026

The camera driver is written but has never seen a device. It is in `shared/` rather than with Architecture A, because both architectures need frames: A reads the depth channel, B uses only the colour one.

Two things in it are not optional and are easy to get wrong later:

* depth is aligned to colour, or the pixel the detector found indexes a different point in the depth image and every object lands centimetres off
* intrinsics come from the SDK at runtime, never hardcoded, because they change with resolution

Everything else here still waits on lab access.

## Status 12 Aug 2026

[[Calibration Marker]] is closed: no marker, no collection script, `T_base_cam` written straight from the measured mount. That removes the last decision blocking W3, so the only remaining gate on this workstream is physical access to the cell.

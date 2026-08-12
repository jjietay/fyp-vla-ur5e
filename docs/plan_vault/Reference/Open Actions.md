---
tags: [tracker]
---

# Open Actions

Things that block other things. The top two are the highest priority items in the whole project.

- [ ] **A1, week 1.** Confirm lab access date and out of hours access to Robotics I. Demo capture is multi hour and will not fit inside office hours. Everything on the critical path waits on this
- [ ] **A2, week 1.** Confirm who pays for the wrist camera and teleop device, then order, see [[Shopping List]]
- [ ] **A3, week 1.** Confirm the existing lab RGB-D model and its SDK, and whether a mount point already exists
- [ ] **A4, week 2.** Confirm GPU availability and hours, and run the throwaway fine tune against `ep_001.h5` to check 12 GB is enough
- [x] **A5, done 12 Aug.** Marker decided: **none**. `T_base_cam` is measured from the mount geometry, see [[Calibration Marker]]. Validation in [[W3 Camera and Calibration]] is now the only check on stage 4
- [ ] **A6, week 4, hard deadline.** Ask whether the StaffLink project title and summary need updating for the widened scope. Changes lock around week 8 and **examiners are allocated based on the title and summary**
- [ ] **A7, Jan 2027.** Agree the demonstration date with the examiner
- [x] **A8, done 11 Aug.** `README.md` and `pyproject.toml` updated. The stale handover, ursim reference and old plan versions were deleted rather than maintained, so `docs/fyp_plan.md` is now the only plan
- [ ] **A9, fortnightly.** Send Prof Cheah a short written progress update. This was skipped through July and it is worth 8 marks, see [[Assessment Mapping]]

## Why A6 has a real deadline

An examiner picked for a pick and place project will be assessing a project about spoken interaction, clarification dialogue and dynamic tasks. That mismatch is avoidable and only avoidable before week 8.

## Two more, added 11 Aug 2026

- [ ] **A10.** Set `ANTHROPIC_API_KEY`. Architecture A will not run a single instruction without it
- [ ] **A11.** Back up `data/raw/` somewhere outside the repo before recording starts. It is gitignored, and the demonstrations are 15 hours of lab time that git will not protect

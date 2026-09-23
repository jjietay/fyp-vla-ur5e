---
tags: [tracker]
---

# Open Actions

Things that block other things. Lab access is handled by JJ directly and is not tracked here.

## Live

- [ ] **A14, next week.** Ask Koji about moving the arm closer to the table and about the new layout. The arm base is about 40 cm from the useful table area. Measure `workspace.x` and `workspace.y` only after the layout is fixed
- [ ] **A15, before demonstration capture.** Confirm the 3D jog device that will drive the demonstrations, see [[Decisions to Make#Teleoperation method]]
- [ ] **A4.** Confirm GPU availability and hours. The throwaway fine tune that was going to check whether 12 GB is enough now waits for the first real episode, because no episode data exists
- [ ] **A9, fortnightly.** Send Prof Cheah a short written progress update. Skipped through July, and worth **8 marks**, see [[FYP Submissions#Where the marks actually are]]
- [ ] **A10.** Set `ANTHROPIC_API_KEY`. Architecture A will not run a single instruction without it
- [ ] **A11.** Back up `data/raw/` somewhere outside the repo before recording starts. It is gitignored, and the demonstrations are 15 hours of lab time that git will not protect
- [ ] **A7, Jan 2027.** Agree the demonstration date with the examiner, rather than leaving it to April

## Closed

- **A1.** Lab access. Handled directly, not tracked here. Arm access from about 22 Sep
- **A2, 13 Sep.** Both cameras are in hand, so the camera items on the shopping list are closed
- **A3, 1 Sep.** The fixed camera is a RealSense D435i. There is no permanent mount: the lab is shared, so the camera is taped in place each session
- **A13, 23 Sep.** `robot.host` set to the arm's address, `10.149.230.1`
- **A5, 12 Aug.** Calibration marker decided: none. `T_base_cam` is measured from the mount, see [[Decisions to Make#Calibration marker]]
- **A6, 31 Aug.** Prof Cheah updated the StaffLink title, closing the examiner allocation risk before the week 8 lock. The new title is in every file that carried the old one
- **A8, 11 Aug.** `README.md` and `pyproject.toml` updated, and the stale plan versions deleted rather than maintained
- **A12, 31 Aug.** The camera axis convention. `pixel_to_3d.py` used MuJoCo's axes against a RealSense that reports OpenCV's, and a measured transform could never have absorbed the flip. Fixed the same day with two guards, see [[Workstreams#The axis convention, fixed 31 Aug]]

## Shopping list

Camera reasoning is in [[Decisions to Make#Camera]].

| Item | Priority | Approx | Note |
|---|---|---|---|
| wrist camera, RGB, USB | must | bought | in hand since 13 Sep |
| wrist camera mount | must | 0 to 60 SGD | must not foul the gripper or the wrist 3 joint |
| cable routing and strain relief | must | 20 SGD | a cable snagging mid demo ends the demo |
| headset or lapel microphone | must | 30 to 80 SGD | the laptop mic will not survive a noisy lab |
| 3D jog device | strong | to confirm | see [[Decisions to Make#Teleoperation method]] |
| tray, glasses, sealable bottles, food colouring | must | 40 SGD | tinted water only |
| small drawer unit | must | 40 to 100 SGD | handle must be graspable by the gripper |
| footswitch for push to talk | nice | 30 SGD | frees both hands, a keyboard key also works |
| wrist RGB-D, short range | optional | 400 SGD and up | only if the lab pays, and nothing may depend on it |

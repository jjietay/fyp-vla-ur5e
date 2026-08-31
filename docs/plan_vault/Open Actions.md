---
tags: [tracker]
---

# Open Actions

Things that block other things. Lab access is handled by JJ directly and is not tracked here.

## Live

- [ ] **A13, before any hardware run.** Set `robot.host` in `config.yaml` to the arm's IP. It is still `127.0.0.1`, which was the simulator's address, and everything in `tests/hardware/` targets whatever that value says
- [ ] **A2.** Confirm who pays for the wrist camera and the teleop device, then order, see the shopping list below
- [ ] **A3.** Whether a camera mount point already exists in the cell. The camera itself is settled: a RealSense D435i, driven by `pyrealsense2`, which the existing driver already supports with no code change
- [ ] **A4.** Confirm GPU availability and hours. The throwaway fine tune that was going to check whether 12 GB is enough now waits for the first real episode, because no episode data exists
- [ ] **A9, fortnightly.** Send Prof Cheah a short written progress update. Skipped through July, and worth **8 marks**, see [[Deliverables#Where the marks actually are]]
- [ ] **A10.** Set `ANTHROPIC_API_KEY`. Architecture A will not run a single instruction without it
- [ ] **A11.** Back up `data/raw/` somewhere outside the repo before recording starts. It is gitignored, and the demonstrations are 15 hours of lab time that git will not protect
- [ ] **A7, Jan 2027.** Agree the demonstration date with the examiner, rather than leaving it to April

## Closed

- **A1.** Lab access. Handled directly, not tracked here
- **A5, 12 Aug.** Calibration marker decided: none. `T_base_cam` is measured from the mount, see [[Decisions#Calibration marker]]
- **A6, 31 Aug.** Prof Cheah updated the StaffLink title, closing the examiner allocation risk before the week 8 lock. The new title is in every file that carried the old one
- **A8, 11 Aug.** `README.md` and `pyproject.toml` updated, and the stale plan versions deleted rather than maintained
- **A12, 31 Aug.** The camera axis convention. `pixel_to_3d.py` used MuJoCo's axes against a RealSense that reports OpenCV's, and a measured transform could never have absorbed the flip. Fixed the same day with two guards, see [[Workstreams#The axis convention, fixed 31 Aug]]

## Shopping list

Confirm who is paying before ordering. Camera reasoning is in [[Decisions#Camera]].

| Item | Priority | Approx | Note |
|---|---|---|---|
| wrist camera, RGB, USB | must | 50 to 120 SGD | global shutter preferred, rolling shutter smears during motion |
| wrist camera mount | must | 0 to 60 SGD | must not foul the gripper or the wrist 3 joint |
| cable routing and strain relief | must | 20 SGD | a cable snagging mid demo ends the demo |
| headset or lapel microphone | must | 30 to 80 SGD | the laptop mic will not survive a noisy lab |
| SpaceMouse Compact | strong | 200 to 500 SGD | see [[Decisions#Teleoperation method]] |
| tray, glasses, sealable bottles, food colouring | must | 40 SGD | tinted water only |
| small drawer unit | must | 40 to 100 SGD | handle must be graspable by the gripper |
| footswitch for push to talk | nice | 30 SGD | frees both hands, a keyboard key also works |
| wrist RGB-D, short range | optional | 400 SGD and up | only if the lab pays, and nothing may depend on it |

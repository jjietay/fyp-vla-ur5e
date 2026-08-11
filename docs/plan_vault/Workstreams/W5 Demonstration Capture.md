---
status: blocked
needs_lab: true
week: 6
---

# W5 Demonstration Capture

Critical path, longest tail, and the workstream most likely to be underestimated. Demo collection is hours of wall clock time and it gates training, which gates the whole comparison.

## Steps

- [ ] teleoperation rig working, see [[Teleoperation Method]]
- [ ] lock the dataset conventions and **write down the reasons**, see [[Action Space]]
- [ ] hardware recorder: threaded, clock gated state polling with camera frames captured in sync
- [ ] record, using a paraphrase from [[W2b Speech Front End]] as each episode instruction
- [ ] export to LeRobot, confirm it loads and the fps assertion passes

## Do not port the sim recorder

Real `moveJ` and `moveL` **block**, unlike the sim stepping loop. The sim recorder counts ticks inside the loop and still carries a cadence bug that was fixed in the server and never back ported. Write the hardware recorder fresh against `DemoRecorder`, which already accepts an explicit timestamp.

## Budget

Roughly 50 episodes per task as a baseline. Density matters more than raw count: 50 episodes spread across 30 cm has been reported to fail where 75 across 10 cm succeeded, so **constrain the workspace for the first fine tune** and widen it afterwards.

At 1.5 to 2 minutes per episode including reset, 250 episodes is 8 to 10 hours before retakes. Budget 15 hours and book it as two or three dedicated blocks, not scattered hours.

Inspect the frames of the first five episodes before recording two hundred. Hands in frame is the failure mode that costs the most to discover late.

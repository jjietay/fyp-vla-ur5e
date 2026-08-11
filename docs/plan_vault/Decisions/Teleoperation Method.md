---
status: decided
---

# Teleoperation Method

**Freedrive to prove the loop, SpaceMouse to record the real dataset.**

The most underestimated risk in the project, because it silently determines whether the training data is usable.

| Option | Cost | Problem |
|---|---|---|
| UR5e freedrive, hand guiding | none | **your hands and arms appear in both camera views on every frame** |
| SpaceMouse to TCP velocity to `servoL` | 200 to 500 SGD | nothing in frame, but a 6 DOF input has a learning curve |
| leader follower arm | 150 SGD and up | the LeRobot native pattern, but a 5 DOF leader onto a 6 DOF UR5e is not a solved mapping |

## The contamination problem

A visuomotor policy trained on kinesthetic teaching can key on the human arm that is present in every training frame. At test time that arm is absent, and the policy has learned a feature that no longer exists.

Discovering this after 200 episodes costs weeks. Inspect the frames of the first five episodes before recording the rest, per [[W5 Demonstration Capture]].

## The plan

* prove the recording loop with freedrive in the first lab session, since it needs no purchase and no wiring
* order a SpaceMouse in week 1 anyway
* record the real training set with the SpaceMouse

If freedrive data does end up shipping, say so in the report and treat it as a named confound rather than hoping nobody notices.

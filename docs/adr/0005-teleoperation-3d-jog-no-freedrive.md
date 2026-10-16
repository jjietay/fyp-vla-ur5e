# Demonstrations are driven with a 3D jog device through servo_to_pose, never freedrive

Decided 23 Sep 2026. The exact device is still to confirm (open action A15).

The most underestimated risk in the project, because it silently determines whether the training data is usable. A visuomotor policy trained on kinesthetic teaching can key on the human arm present in every training frame. At test time that arm is absent, and the policy has learned a feature that no longer exists. Discovering this after 200 episodes costs weeks, so inspect the frames of the first five episodes before recording the rest.

## Considered options

| Option | Cost | Problem |
|---|---|---|
| UR5e freedrive, hand guiding | none | **your hands and arms appear in both camera views on every frame** |
| SpaceMouse to TCP velocity to `servoL` | 200 to 500 SGD | nothing in frame, but a 6 DOF input has a learning curve |
| leader follower arm | 150 SGD and up | the LeRobot native pattern, but a 5 DOF leader onto a 6 DOF UR5e is not a solved mapping |

The first plan was to prove the recording loop with freedrive and record the real set with a SpaceMouse. On 23 Sep that changed to no freedrive at all, so no hand is in any frame, including the first test episodes.

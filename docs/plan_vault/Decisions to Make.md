---
tags: [overview]
---

# Decisions

Choices that shape the project, each with the reasoning attached. These become the methodology chapter of [[FYP Submissions#D4 Final Report]], so the reason written down at the time is worth more than it looks.

| Decision | State |
|---|---|
| [[Decisions to Make#Camera|Camera]] | decided, both cameras in hand 13 Sep |
| [[Decisions to Make#Speech stack|Speech stack]] | decided, working on the GPU 31 Aug |
| [[Decisions to Make#Detector query format|Detector query format]] | decided, forced by how OWLv2 works |
| [[Decisions to Make#Teleoperation method|Teleoperation method]] | decided 23 Sep, 3D jog device, no freedrive |
| [[Decisions to Make#Action space|Action space]] | frame decided 31 Aug, rotation format decided 23 Sep |
| [[Decisions to Make#Calibration marker|Calibration marker]] | decided 12 Aug, no marker. Ball fit each session since 13 Oct |
| [[Decisions to Make#Architecture B inference|Architecture B inference]] | decided 31 Aug |

A decision recorded without its reason is not much use six months later when an examiner asks why.

---

## What config.yaml holds, and what it does not

`config/config.yaml` is the single source of truth for every **tunable** value, so that Architecture A and Architecture B cannot be constrained differently. A number living in a module is invisible to whoever is reviewing the workspace bounds, and invisible to every test.

Six things stay in code on purpose. Each would be made worse by living in an editable file.

* **`STATE_NAMES`, `ACTION_NAMES`, `REQUIRED_DATASETS`** in `lerobot_export.py`. The format contract with a trained checkpoint. Editing one line in config would silently break every checkpoint trained before the edit, with no error, and the symptom would be a policy sending roll to z
* **`TASKS` and `CHANGELOG`** in `evaluation/suite.py`. The experimental design, frozen before either pipeline was tuned. That freeze is what the evaluation chapter claims, and a file anyone can edit weakens the claim
* **`SYSTEM_PROMPT` and `GROUNDING_PROMPT`** in `planner.py`. Multi-paragraph text that YAML handles badly, and "behaviour changes by editing prompts" is a stated property of Architecture A. They are the architecture rather than settings for it
* **`TOOLS`, `TOOL_NAMES`, `MOTION_TOOLS`** in `tools.py`. Derived from the skill signatures with a test enforcing the match. In config they could disagree with the code
* **`GRIPPER_CLOSED` and `GRIPPER_OPEN`** in `skills.py`. An API contract, not a tunable, so these never change with the cell
* **The RG2 protocol numbers** in `ur5e_controller.py`, agreed 23 Sep 2026. The word formula (the 9.2 offset, the 110 mm clamp, the 3 to 40 N force range), the width formula and the sync cycle waits. They are the gripper's protocol copied from `docs/rg2.script`, not tuning, and a changed number sends a wrong command or reads a wrong width. The tunables (open and close width, force, tolerance, resends) are in `robot.gripper`

`PROJECT_ROOT`, `DEFAULT_CONFIG_PATH` and `_CACHE` in `helpers/config.py` are the machinery that finds and reads the config, so they cannot be read from it.

The file itself stays lean: values, with at most one line of comment each. Anything that needs a paragraph belongs here or in `docs/commands.md`.

---

## Camera

**A fixed RGB-D plus a wrist RGB camera, the fixed one square on to the table and 0.7 to 0.8 m above it.**

The lab unit is a **RealSense D435i**, confirmed 31 Aug. Its SDK is `pyrealsense2`, already a dependency, and `shared/hardware/camera.py` already drives it.

### Does the wrist camera need depth

No. SmolVLA consumes RGB images, a state vector and a language string, and has no depth input at all. Architecture A needs depth only from the **fixed** camera, because that is where detection happens and where pixels are back projected into 3D.

| Camera | Type | Architecture A | Architecture B |
|---|---|---|---|
| fixed, side view | RGB-D, the lab D435i | detection, depth to 3D, base frame | first RGB view |
| wrist mounted | RGB is sufficient | optional close range re-detect | second RGB view |

**Hard rule: nothing may depend on wrist depth.** Design as if it is RGB. If depth arrives it is a branch that can be deleted without touching anything else. If the lab pays, specify a short minimum range model, because a D435 class sensor cannot focus closer than about 0.3 m and is useless at grasp distance.

Two cameras is evidence rather than preference. Reference SmolVLA fine tuning setups use two views, and overhead plus wrist produces noticeably smoother trajectories than wrist only. A single camera chosen against the reference configuration would compromise [[Evaluation of Architecture]].

### Where the fixed camera stands

Square on to the flat side of the table, arm at the far end, workspace between the two.

- **azimuth: square on**, optical axis along the base x axis rather than diagonally across a corner
- **lateral: centred**, camera at `y = 0` in base frame, position `(x_c, 0, z_c)` with `x_c` beyond the 0.75 m reach limit measured on 23 Sep, looking back along minus x
- **elevation: 0.7 to 0.8 m above the table**, about 35 to 39 degrees at a 1 m standoff

Two D435i limits that constrain this: min-Z is about **0.3 m**, so nothing may sit closer than that, and it needs a **USB 3** cable, because a USB 2 cable enumerates and then silently caps resolution.

### Why square on and centred

Range error lands along the camera ray. Pointing that ray down the base x axis keeps almost all of it in one axis, so the measured `T_base_cam` bias and the surface-to-centroid offset collapse into a single constant the reach test can measure. From a diagonal the same error smears across x and y and needs two.

Centring is not cosmetic: a lateral offset tilts the optical axis and range error starts splitting into x and y, the same cross talk as an azimuth mounting error but caused by position. With both, the error decomposes cleanly. y carries no range error at all, since it comes purely from pixel column. x and z carry all of it, roughly 80 percent x and 20 percent z at 39 degrees, and z is absorbed by `pick` descending from its approach height.

Occlusion also drops. Against the frozen layouts in `evaluation/suite.py`, square on puts 15 of 100 at risk against 25 of 100 at a diagonal, because the layouts spread objects over 40 cm in y and only 25 cm in x.

One consequence to expect rather than fix: the arm column becomes the backdrop behind the workspace, so OWLv2 sees objects against metal. If detection scores look soft on the real cell, rule this out before blaming the threshold.

### Why 0.7 to 0.8 m

A near object hides a region behind it of length `s = d * H / (h - H)`, for standoff `d`, object height `H` and camera height `h`. It diverges as `h` approaches `H`, which is why a camera near table level is unusable.

At `d` of 1 m, counting a hit when one object falls in another's shadow within 5 cm laterally:

| camera height | elevation | 5 cm objects | 20 cm objects |
|---|---|---|---|
| 0.30 m | 17 deg | 12/100 | 15/100 |
| 0.60 m | 31 deg | 0/100 | 15/100 |
| 0.80 m | 39 deg | 0/100 | 15/100 |
| 1.50 m | 56 deg | 0/100 | 6/100 |

Short object occlusion disappears past a knee at 0.6 m, and 0.7 to 0.8 m is that knee plus margin. Measure `d` on the cell and re-run the formula before settling the tripod. Going higher is not free: more elevation foreshortens the vertical faces OWLv2 needs to find a drawer handle.

**What elevation cannot fix** is tall objects. A 20 cm bottle still occludes at 1.0 m. Tier 1 and Tier 2 layouts must place tall objects so nothing sits behind them, and [[Tasks#Tier 2 Ambiguity]] is the exposed one: a hidden second candidate turns a clarification trial into Architecture A confidently picking the visible object, which scores a mount choice as an architecture finding.

### Deciding at the bench

The placement above assumes the camera can stand at the far end of the table. If the mounting options do not allow that, here is what each alternative costs. Whichever you pick, **tape the tripod position**, because the lab is shared and the camera comes down between sessions.

The measurement procedure is identical in all three cases. Only the numbers change: `--xyz` is the camera's optical centre in the base frame, `--rpy` is how it is turned relative to that frame.

**Far end, looking back at the arm.** The plan. Nothing stands between the camera and the objects, and range error collapses into one axis so the calibration bias is a single constant the reach test can measure. Take this if it is available.

**Off to one side, looking across.** The acceptable fallback. Occlusion is moderate and the geometry still works, but range error now splits between x and y, so the bias is two constants instead of one. The reach test still measures it, and the correction is less clean to state in the report.

**Behind the arm, looking forward over it.** The one to avoid if there is any choice, because the arm stands between the camera and everything it is trying to see, and unlike a static obstacle **it moves**, so the occlusion changes shape during the task.

Two concrete harms rather than a general worry. The detector can lose the target mid-task when the arm passes in front of it. And `depth_at` can sample the ARM's depth rather than the object's, which puts a grasp point somewhere on the gripper, and nothing downstream would notice because the number is perfectly valid.

If it is the only option, three things make it survivable:

* mount as high as the tripod allows, so the sight line clears the arm's body. This trades against the vertical faces OWLv2 needs for a drawer handle, so it is a real cost rather than a free fix
* **re-perceive only from the home pose**, never mid-motion. Architecture A re-perceives before each planning turn, and if that happens while the arm is extended, the arm is in the frame. Retracting first costs a second per turn and removes the failure entirely
* say so in the report. A known dynamic occlusion, stated with its mitigation, reads as understanding the limitation. Discovered by an examiner in a failure video, it does not

### The residual, accepted knowingly

Square on foreshortens the reach axis, and foreshortening is the weakest cue in an RGB image. That cost falls on Architecture B alone, since B never forms a 3D point and cannot benefit from the geometry square on buys.

Accepted for two reasons. The wrist camera reads approach distance directly, and the reference configuration is overhead plus wrist, where the descent runs straight at the overhead camera and is foreshortened the same way. Occlusion, by contrast, costs both architectures equally, so a diagonal is not a neutral compromise: it pays a penalty on both to buy back what the wrist camera already covers.

State the placement and this residual in [[FYP Submissions#D2 Interim Report and Video]]. Documenting the direction of a known bias is what makes the comparison defensible.

---

## Speech stack

**`faster-whisper` running locally, push to talk, `compute_type="float16"`.**

Speech recognition is an off the shelf component here, like the camera driver. Say so explicitly in [[FYP Submissions#D4 Final Report]], because claiming it invites an examiner to probe an area with no novel work in it.

### Why local rather than a cloud API

Cost is not the reason. Total spoken audio across the project is roughly 500 utterances at 5 seconds, under a dollar at any provider.

The reason is the demonstration. A local model cannot fail because the campus network did, and [[FYP Submissions#D5 Demonstration]] happens in front of an examiner. Keep a cloud path behind a config flag as a fallback.

### Two CUDA traps, both hit

`compute_type` must be **float16**. CTranslate2's default int8 crashes on sm_120 with `CUBLAS_STATUS_NOT_SUPPORTED`, because Blackwell int8 tensor cores need padding older builds do not emit.

CTranslate2 also needs **cuBLAS for CUDA 12 and cuDNN 9**, and dlopens them by bare name so it searches only the loader path. This venv has no torch by design, so nothing pulled them in, and the symptom is dishonest: the model loads, reports `cuda/float16`, and fails on the first utterance. Both are now dependencies, and `shared/speech.py` preloads them by absolute path so no `LD_LIBRARY_PATH` export is needed.

VRAM is not a constraint. `large-v3` at float16 needs about 3.4 GB, so use the largest model.

### Push to talk, not always on

Capturing a complete 3 to 5 second clip sidesteps endpointing, voice activity detection and partial hypotheses entirely. A robotics lab is noisy, and wake word detection is a research project of its own.

Enter starts, Enter stops, rather than a held key. A held key needs a real display session and breaks the moment the cell is driven over SSH.

### The rule that protects the comparison

**Both architectures get the same raw transcript.** Architecture A absorbs disfluent speech natively because it is an LLM. Architecture B takes the string conditioned on its training distribution and has no way to normalise wording. Cleaning the transcript for B inserts an LLM into B and it stops being end to end. That asymmetry is measured rather than patched, see [[Hypotheses for Comparison]].

---

## Detector query format

**OWLv2 takes short noun phrases, never sentences. This is why Architecture A needs a language model in front of its detector.**

OWLv2 encodes text with CLIP's text encoder, so every query becomes a single embedding matched against per patch image embeddings. What follows:

- a sentence produces one embedding for the whole sentence, and no region of the image corresponds to it
- there is no mechanism for splitting a sentence into the several objects it mentions
- there is no mechanism for ignoring the verbs, articles and destinations a sentence also contains
- queries are embedded independently, so a relationship like "the cube that is inside the tray" cannot be represented

CLIP's 77 token context is a real limit but not the binding one. A spoken command fits inside it and still fails, because the problem is what the embedding means rather than how long it is.

### Why it matters to the project

Something has to convert `"place that red cube into that metal tray"` into `["red cube", "metal tray"]` before OWLv2 can be asked anything. That is the grounding call in `planner.py::extract_queries`, and it is a genuine architectural requirement.

Worth stating plainly in the report, because it cuts both ways. Architecture A needs a second language model call purely to bridge from human language into detector language, and that bridge is a failure point, recorded as H9 in [[Hypotheses for Comparison]]. A wrong noun phrase kills the run before perception begins. Architecture B pays nothing, because SmolVLA takes the raw string into the policy and never converts it.

So the modular architecture is not simply more interpretable at a small cost. It requires a translation step the end to end model does not, and that step is one of the places it breaks.

### Practical rules

- keep distinguishing words the user actually said, so "red cube" not "cube"
- do not invent distinguishing words, so "the cube" stays "cube"
- include the destination, since the robot has to find that too
- cap the number of queries, because more queries slow OWLv2 and add false positives

---

## Teleoperation method

**A 3D jog device through `servo_to_pose`, and no freedrive at all.** Decided 23 Sep 2026. The exact device is still to confirm.

The most underestimated risk in the project, because it silently determines whether the training data is usable.

| Option | Cost | Problem |
|---|---|---|
| UR5e freedrive, hand guiding | none | **your hands and arms appear in both camera views on every frame** |
| SpaceMouse to TCP velocity to `servoL` | 200 to 500 SGD | nothing in frame, but a 6 DOF input has a learning curve |
| leader follower arm | 150 SGD and up | the LeRobot native pattern, but a 5 DOF leader onto a 6 DOF UR5e is not a solved mapping |

A visuomotor policy trained on kinesthetic teaching can key on the human arm present in every training frame. At test time that arm is absent, and the policy has learned a feature that no longer exists. Discovering this after 200 episodes costs weeks, so inspect the frames of the first five episodes before recording the rest.

The first plan was to prove the recording loop with freedrive and record the real set with a SpaceMouse. On 23 Sep that changed to no freedrive at all, so no hand is in any frame, including the first test episodes.

---

## Action space

**TCP pose deltas. Seven numbers: a base frame translation delta, a rotation delta, and an absolute gripper target.** Frame decided 31 Aug 2026. Rotation format decided 23 Sep 2026, see below.

An earlier lean towards joint positions predated `skills.py`. Architecture A's primitives command TCP poses through `move_to_pose`, and the deciding constraint in this project is that the action space must match what A commands, or the two architectures are not solving the same problem and the comparison is unfair in a way a careful examiner will spot.

### Why TCP poses

- Architecture A commands them, so both architectures speak one language
- the policy learns the task only, because the controller's IK supplies the kinematics. In joint space it would have to learn both, and would underperform for a reason that is not architectural
- generalisation across placement is cheaper, because an object 5 cm further right means roughly 5 cm more action wherever it sits
- `WorkspaceEnvelope` bounds x, y and z, so the safety check is direct and testable with no arm attached
- the detector, the depth back projection and the hand-eye transform all already produce base frame positions, and calibration error is reported in millimetres of hand position

### What it costs

Inverse kinematics is in the loop, so near a singularity a small hand movement demands very large joint speeds. Keep the workspace away from full extension and from directly over the base. One hand pose also has several joint solutions, so the arm can reconfigure at a branch boundary.

### Rotation format, settled 23 Sep 2026

The down pose has a rotation vector of magnitude pi (`down_rotvec` is `[0.580, 3.087, 0.0]`). That is the boundary of **both** rotation formats in use. A turn of pi about an axis equals a turn of pi about the opposite axis, so the rotation vector can flip sign, and converting to Euler gives roll of pi, which is the wrap point, so roll can flip too. Same physical pose, opposite numbers.

`scripts/check_orientation.py` on the lab arm on 23 Sep 2026 showed roll flipping between -pi and +pi. So `observation.state` now stores rotation as the first two columns of the rotation matrix: ten numbers, being x, y, z, six matrix entries and the gripper. They change smoothly everywhere, and the third column is their cross product, so nothing is lost. A test puts two poses either side of the flip and checks the state barely changes.

The action is unchanged: seven numbers with an Euler rotation delta. Action deltas were always immune, because they are computed through rotation matrices and a delta near zero is far from any wrap point. No episode had been recorded, so no data was lost.

### Also settled

Architecture B's per step ceiling is DERIVED as `workspace.max_speed * robot.servo.time` rather than written down. It had been a literal 0.05 m, which at 20 Hz is 1.0 m/s against Architecture A's 0.25 m/s ceiling. Both architectures are constrained identically again, and a test asserts it.

Also to lock before recording: the observation set of two RGB views plus the state vector, the resolution and control rate, and the per episode instruction string drawn from the paraphrase set rather than a fixed template.

---

## Calibration marker

**No marker. `T_base_cam` comes from the measured mount geometry instead.** Decided 12 Aug 2026.

Changed in part on 13 Oct 2026: still no marker, but the transform is now fitted each session with a ball, see [[Decisions to Make#Update 13 Oct, the ball fit became the plan]]. The reasoning below is kept as the record of the first choice.

The fixed camera stands at a measured offset from the arm base, so `T_base_cam` is one transform. Rather than recover it by observing a target, it is written directly from the measured position into `config/calibration/T_base_cam.json`. The lab is shared, so the camera is taped in place and taken down between sessions, and the transform is re-measured at the start of each session. Nothing in the pipeline changes, because `hand_eye.py` loads the transform from JSON and never cares how the numbers got there.

Two reasons, and the second decided it. Within a session the camera does not move, so a geometric answer exists without observing anything. And Prof Cheah's view is that a printed board in the cell looks out of place against a real deployment, while the project argues for an interface a person could actually use.

### The cost, recorded honestly

`T_base_cam` is six numbers, and the rotation is the exposed one. At a 1 m standoff, **1 degree of camera orientation error displaces every back projected point by about 17 mm**. Holding a 5 mm target would need the mount known to better than 0.3 degrees, which a bracket and a rule will not give.

So expect a systematic offset of roughly 1 to 3 cm, in one consistent direction rather than randomly per object. Grasp tolerance absorbs it.

This has to be measured, not assumed, which is why validation matters more than it did before. A hand written transform has no residual of its own, so the reach test against known points is the only number there is.

Two things that make it work better: measure to the camera's **optical origin**, not the case front, since the colour sensor sits a few millimetres inside the housing and a datum error is pure bias; and re-measure after any knock, because there is no fit to notice the camera moved.

### The fallback if the offset is too large

Ball on the gripper, Kabsch, no fiducial. A matte sphere of known radius held in the jaws, the arm driven to 15 to 20 poses at fixed down orientation, sphere centres paired against `tcp_pose` and fed to the solver that is already written and tested. Tag free, so the objection above still holds, and it reuses `depth_at`, `pixel_to_camera`, `surface_to_centroid`, `solve_rigid_transform` and `report` without new maths.

Kept as rung two rather than the plan, because the measured transform may well be good enough and costs an hour instead of an afternoon.

### Update 13 Oct, the ball fit became the plan

On 1 Oct the measured transform needed a hand correction of (+22, +62, +23) mm, about 70 mm, before the orange could be grasped. That is well past the 1 to 3 cm the cost section expected. The ball fit is now the per session procedure, in `scripts/calibrate_ball.py`.

- it carries the ball to 16 poses at two heights, fits with `solve_rigid_transform`, drops one wrong detection if there is one, and reports the RMS
- it costs one to two minutes a session instead of an afternoon, so the original cost argument no longer holds
- it still uses no marker, so the objection to a printed board in the cell still holds
- its blind spot is the ball offset in the gripper. Every pose has the same tool orientation, so an offset error is the same shift at every point and the fit absorbs it. The reach test checks that once, and again if the ball or its placement changes
- measuring the mount by hand is now the fallback

---

## Architecture B inference

**Two processes, both LeRobot's own.** `policy_server` holds the checkpoint and does the forward pass. `robot_client` holds the action queue and drives the robot. They talk over gRPC. We supply only the missing piece, a UR5e adapter conforming to LeRobot's `Robot` interface, as a separate installable package at `integrations/lerobot_robot_ur5e/`. Nothing in LeRobot is forked or patched.

### Why not write our own loop

The action queue is the part that matters, and it lives in `robot_client`. It keeps a buffer of predicted actions, executes from it, requests a fresh chunk before it empties, and merges the overlap. Without it the arm stops dead every time the policy thinks.

The argument against writing it ourselves is not effort, it is the comparison. If B underperforms A, the first question an examiner asks is whether the inference loop was correct. Using the reference implementation removes that question; a hand written queue keeps it open and would have to be defended with timing measurements there is no time to collect.

### What it costs

The client process needs `lerobot`, `torch`, `ur_rtde` and `pyrealsense2` together, so Architecture B runs entirely out of `~/lerobot/.venv` with the two hardware drivers added. The FYP venv stays free of torch and keeps running Architecture A. Splitting the processes does **not** remove the need for `ur_rtde` in the lerobot environment, because the client is the half that touches hardware.

### The two contracts

Both read out of the LeRobot 0.6.1 source rather than the documentation.

**State names.** LeRobot builds `observation.state` by looking each dataset feature name up in the dict the robot returns. So the adapter's observation keys are literally `x`, `y`, `z`, `r00`, `r10`, `r20`, `r01`, `r11`, `r21`, `gripper`, and the camera key is `top`. `rIJ` is row I, column J of the rotation matrix. Euler `roll`, `pitch`, `yaw` were dropped on 23 Sep 2026, see the rotation format section above.

**Action order.** `robot_client` unpacks the policy's action tensor by the **order** of `action_features`, by index, with no names involved. A reordering produces no error and sends roll to z.

Both tuples live in `lerobot_export.py` as `STATE_NAMES` and `ACTION_NAMES`, and the adapter imports them. Retyping them is how the two halves would drift apart.

### What it exposed

`URController` had only `move_to_pose` and `move_joints`, which block until the motion finishes. A chunked policy at 20 Hz needs a new target every 50 ms without waiting, so `servo_to_pose` and `servo_stop` were added on top of `servoL`, with the gains in `config.yaml` under `robot.servo`.

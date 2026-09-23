---
tags: [overview]
---

# Workstreams

Build order, arranged by dependency rather than by date. [[Weekly Focus]] places these in time.

Critical path to a result: lab access, then W3, then W5, then W6, then W7. Everything in Architecture A sits off that path, which is why it went first.

| Workstream | Needs lab | State, 23 Sep 2026 |
|---|---|---|
| [[Workstreams#W2 Architecture A software\|W2 Architecture A software]] | no | code complete. Controller motion and the gripper protocol proved on the arm, skills not yet run |
| [[Workstreams#W2b Speech front end\|W2b Speech front end]] | no | **working**, wired into the runner |
| [[Workstreams#W3 Camera and calibration\|W3 Camera and calibration]] | yes | orientation check run on the arm. Camera not yet on the cell |
| [[Workstreams#W4 Architecture A end to end\|W4 Architecture A end to end]] | yes | code complete, unrun on hardware |
| [[Workstreams#W5 Demonstration capture\|W5 Demonstration capture]] | yes | not started, no data exists |
| [[Workstreams#W6 Architecture B training\|W6 Architecture B training]] | partly | robot side built, no checkpoint |
| [[Workstreams#W7 Evaluation and write up\|W7 Evaluation and write up]] | yes | harness built, trials blocked |

**Treat every "code complete" as unverified** until it has run on the arm. What has met hardware so far: speech, the controller's moves and servo stream (hardware tests 7 of 7, twice, on 23 Sep), and the RG2 protocol. The rewritten gripper code with its resend has not run on the arm yet.

---

## W2 Architecture A software

The whole modular pipeline except the camera. Built against saved PNGs as fixtures, so it did not wait on lab access.

- [x] `architecture_a/skills.py` with `pick`, `place`, `pour` and `open_drawer`
- [x] tool schemas in `tools.py`, matched to the signatures by an automated check
- [x] `ask_user(question, options)` exposed alongside the motion primitives
- [x] `planner.py`: the system prompt lists skills plus detected objects with base frame positions, and the model returns an ordered tool call plan
- [x] plan validation **before** dispatch, rejecting unknown skills, out of workspace coordinates and malformed arguments
- [x] clarification loop: on `ask_user`, surface the question, block for input, append the answer, re-plan
- [x] query grounding, so the instruction chooses the detector vocabulary

### Rules

Primitives are written against the **shared controller interface only**. A primitive reaching past the controller for a `ur_rtde` attribute means the comparison starts measuring plumbing instead of architectures. A test enforces this.

Approach height, descent speed, grasp width, tilt angle and tilt rate are all parameterised, because the planner will want to vary them.

### Do not skip validation

A plan that reaches the controller unvalidated is how an arm gets damaged. This is the one place in the project where a software bug has a physical consequence.

### Query grounding

The instruction chooses the detector vocabulary rather than it being passed on the command line. The planner turns the transcript into short noun phrases and OWLv2 searches for those, using the same model as the planner so no new component enters Architecture A. It is its own trace stage, because a vocabulary error is not a detection error and recording it as one would corrupt the per stage breakdown. Extraction failure raises rather than falling back to a default list, since a silent fallback would restore the advantage this removes.

Why it matters: Architecture A now starts from exactly the string Architecture B gets. Before, a human supplied the object list, which was information B never received and which weakened [[Tasks#Tier 4 Generalisation]].

### What still gates a first run

- ~~measure `down_rotvec`, `robot.home_q` and the table height~~, done 23 Sep
- measure `workspace.x` and `workspace.y`, after the table layout is agreed with Koji
- write `T_base_cam.json` from the measured camera position, since the pipeline refuses to start without it
- set `ANTHROPIC_API_KEY`

---

## W2b Speech front end

The user speaks, the system acts. ASR sits **outside both architectures** and is identical for both: one microphone, one model, one transcript, handed to A and to B unchanged. Reasoning in [[Decisions to Make#Speech stack]].

- [x] `shared/speech.py`: push to talk capture, then `faster-whisper`, then text, and nothing else
- [x] `compute_type="float16"` set explicitly, since the default int8 crashes on RTX 50 series
- [x] runnable alone as `scripts/listen.py`, so a bad transcript can be pinned on the microphone rather than the architecture
- [x] `--text` flag, so either architecture can be driven from the keyboard while debugging
- [x] every transcript logged with its audio, timestamped
- [x] paraphrase sets, living in `evaluation/suite.py` as `train_utterances` and `heldout_utterances` for all five tiers
- [x] wired into `scripts/run_architecture_a.py`, and **speech is the default**
- [ ] optional text to speech for spoken clarification, if [[Tasks#Tier 2 Ambiguity]] is working by W9

**Working end to end on the GPU.** It transcribed a real voice correctly on its first attempt and logged the audio beside the transcript.

### The deadline hidden in here

The paraphrase sets must exist **before** demonstration capture starts. The instruction string is per episode metadata, so varying it during recording is free, and retrofitting it means re-recording everything. Miss this and the phrasing robustness hypothesis becomes untestable. They are done.

### Three things that were decisions rather than typing

- the recogniser is opened **before** the arm and the camera, because loading `large-v3` takes seconds and fails outright on a missing PortAudio, and neither is worth discovering with a live arm connected and a gripper powered up
- the transcript is echoed before the pipeline runs, because the arm is about to move on the strength of that string
- the wiring is in the runner, not the pipeline. `ArchitectureA.run` still takes a plain string, because a pipeline that owned a microphone could quietly clean up its own input

### Two things to expect on a new machine

`sounddevice` needs PortAudio present, `sudo apt install libportaudio2` on Ubuntu. And a USB headset does not become the default input just by being plugged in, so `--devices` is the first thing to try when a clip comes back silent.

`--cpu` works everywhere and measured a real time factor of 2.50, so it is for testing and not for the demo.

---

## W3 Camera and calibration

The single biggest unblocker in the project. Nothing downstream can address a real object without it.

- [x] real camera driver with intrinsics read from the SDK, in `shared/hardware/camera.py`
- [x] camera confirmed as a **RealSense D435i**, which that driver already supports
- [x] marker decided: **no marker, the transform is measured from the mount**
- [x] `scripts/check_camera.py`, which proves the device works and reports intrinsics, coverage and the USB link
- [x] `scripts/write_calibration.py`, which writes `T_base_cam.json` from six measured numbers
- [x] `scripts/check_calibration.py`, which reaches for N points and reports the error in mm
- [x] `check_orientation.py` run on the arm, 23 Sep
- [ ] run the rest, which needs the camera on the cell

### Bench order

Each of these fails in a way that looks like the next one's problem, so doing them out of order costs a session.

1. `check_camera.py`. Proves frames arrive, reports intrinsics at this resolution, warns on a USB 2 link, and prints the centre pixel as a 3D point so the axis convention is visible
2. `check_orientation.py`, arm parked pointing down, read only. Run on 23 Sep: Euler roll flipped, which settled the rotation format in [[Decisions to Make#Action space]]. It now checks that the 6D state is stable
3. measure the mount, camera optical origin to arm base, translation and orientation
4. `write_calibration.py` with those six numbers
5. `check_calibration.py`, the only evidence stage 4 works

Depth coverage below 50 percent in step 1 means the scene, not the camera. Matte objects read far better than shiny ones, and anything closer than 0.3 m is below the sensor minimum and returns noise.

### The trap

**Read poses from RTDE, never from the pendant.** On 23 Sep the pendant showed TCP z as -0.020 where RTDE reported 0.380, because the pendant shows a different frame. Everything in the code is in the robot base frame as RTDE reports it, and there is no world frame on the real cell. The old note about `quat="0 0 0 -1"` belonged to the MuJoCo model.

### Why validation is not optional

A calibration you did not check numerically is a calibration you do not have, and every downstream failure will get blamed on it. This got sharper once the transform became hand measured rather than fitted. A fit reports its own residual and tells you when it is wrong. A measured transform reports nothing, so the reach test is the **only** number that says whether stage 4 works. Expect a systematic offset in one direction rather than scatter.

### Two things in the driver that are not optional

Depth is aligned to colour, or the pixel the detector found indexes a different point in the depth image and every object lands centimetres off. And intrinsics come from the SDK at runtime, never hardcoded, because they change with resolution.

### The axis convention, fixed 31 Aug

`pixel_to_3d.py` uses the **OpenCV** camera frame: x right, y down, z forward positive. That is what `pyrealsense2` deprojects into and what the SDK intrinsics assume.

It did not, until 31 Aug. It negated y and z, which is the OpenGL convention MuJoCo uses, inherited from the simulation this project started in. Nothing caught it, because `camera_to_pixel` negated them back and the round trip stayed exact.

It would not have stayed harmless. A fitted `T_base_cam` absorbs a fixed axis flip silently, because Kabsch finds whatever map takes one cloud to the other. The no-marker decision means the transform is measured, and a measured transform cannot absorb a flip that is not physically there. Every back projected point would have landed mirrored and behind the camera, reading as a badly wrong calibration.

Two guards now exist, because the original bug was invisible for months. `tests/unit/test_perception.py` pins the signs directly, and `check_camera.py` prints a real back projected point and says whether z is positive. If a future session sees negative z out of `pixel_to_camera`, the OpenGL convention has come back, and the fix is not to negate downstream.

---

## W4 Architecture A end to end

Wiring the software and the camera into one runnable thing.

- [x] single entry point, in `architecture_a/pipeline.py` and `scripts/run_architecture_a.py`
- [x] per stage logging in `architecture_a/trace.py`, one JSONL file per run
- [x] closed loop by default, re-perceiving before each planning turn, with `--no-reperceive` to measure what that is worth
- [ ] reach 70 percent success on [[Tasks#Tier 0 Pick and Place]] over 20 trials

### Build the logging properly the first time

Per stage diagnosability is the entire argument for Architecture A. It is one of the [[Hypotheses for Comparison]], and this log becomes the evidence in the final report. Retrofitting it later means re-running every trial.

Closing the loop is also what makes [[Tasks#Tier 3 Drawer]] possible at all. Document the failure modes encountered while doing it, because they are results rather than bugs.

### The stage list

    capture, grounding, detection, depth, transform, planning, clarify, execution

`grounding` is the instruction becoming detector queries, and it is its own stage because a wrong noun phrase is not a detection error. `clarify` is the run pausing to ask the user, which is the Tier 2 mechanism. `ASR` sits outside the pipeline entirely, in the runner, per W2b.

---

## W5 Demonstration capture

Critical path, longest tail, and the workstream most likely to be underestimated. Demo collection is hours of wall clock time and it gates training, which gates the whole comparison.

- [ ] teleoperation rig working with the 3D jog device, see [[Decisions to Make#Teleoperation method]]
- [ ] lock the dataset conventions and **write down the reasons**, see [[Decisions to Make#Action space]]
- [ ] hardware recorder: threaded, clock gated state polling with camera frames captured in sync
- [ ] record, using a paraphrase from W2b as each episode instruction
- [ ] export to LeRobot, confirm it loads and the fps assertion passes

### What data exists

**None.** `data/raw/episodes/` holds only its `.gitkeep`. The `ep_001.h5` referred to in older commands, and the renders made from it, came from the MuJoCo simulation work and were deleted deliberately: recorded in sim, predating two fixes, and one episode is not a dataset. Every count below starts from zero.

### Do not port the sim recorder

Real `moveJ` and `moveL` **block**, unlike the sim stepping loop. Write the hardware recorder fresh against `DemoRecorder`, which already accepts an explicit timestamp.

### Budget

Roughly 50 episodes per task as a baseline. Density matters more than raw count: 50 episodes spread across 30 cm has been reported to fail where 75 across 10 cm succeeded, so **constrain the workspace for the first fine tune** and widen afterwards.

At 1.5 to 2 minutes per episode including reset, 250 episodes is 8 to 10 hours before retakes. Budget 15 hours and book it as two or three dedicated blocks, not scattered hours.

Inspect the frames of the first five episodes before recording two hundred. Hands in frame is the failure mode that costs the most to discover late.

Back up `data/raw/` outside the repo before recording starts. It is gitignored, and git will not protect 15 hours of lab time.

---

## W6 Architecture B training

Fine tuning SmolVLA and driving the arm from it.

- [x] inference: LeRobot's `policy_server` and `robot_client` plus our UR5e adapter, see [[Decisions to Make#Architecture B inference]]
- [x] safety envelope wired into B, in the adapter's `send_action`, using the same `WorkspaceEnvelope` Architecture A uses
- [x] asynchronous inference, which comes free with `robot_client`
- [ ] throwaway fine tune to find out whether 12 GB is enough, which now waits on the first real episode
- [ ] fine tune on Tier 0 first, logging config, seed, wall clock and GPU hours
- [ ] re-train per tier as demonstrations land

### Non negotiables

The safety envelope is not optional. A VLA given a novel observation can emit a chunk that slams the arm into the table. `WorkspaceEnvelope` lives in `shared/` precisely so both architectures get the identical clamp: two envelopes would mean the two architectures were constrained differently, which quietly breaks the comparison.

The clamp got stricter than planned. The envelope catches a pose outside the box and does not catch a plausible pose reached in one enormous step, so the adapter also scales any delta larger than the per step ceiling. Scaling the whole vector rather than clipping each axis, because per axis clipping bends the direction the policy asked for instead of only slowing it.

Keep the checkpoint and the exact dataset version together. A checkpoint whose dataset you cannot identify is not a result, it is an anecdote.

### VRAM

12 GB is workable but not roomy. Weights, gradients and optimiser states come to roughly 7 GB before activations, with two camera streams on top. If it does not fit: smaller batch with gradient accumulation, gradient checkpointing, an 8 bit optimiser, or freeze the vision backbone.

### What to watch on the first real run

`gripper_toggle` blocks until the fingers stop: about 0.2 s even when nothing moves, and about 1.5 s for a full stroke. It also ends the servo stream while it runs, because the RG2 is driven by a URScript that replaces the ur_rtde control script for that time. So the adapter only actuates on a change of decision, and each grasp and release will be a visible stop in the motion. Whether this hurts B's success rate is for the first real run to show.

---

## W7 Evaluation and write up

Where the project turns into a result.

- [x] harness built in `src/fyp/evaluation/`, importing neither architecture, with a test enforcing that
- [ ] run both architectures across every attempted tier
- [ ] **freeze the results before writing**, and tune nothing afterwards
- [ ] figures: success rate by tier and architecture, Architecture A failure stage breakdown, Tier 2 choice distribution, cost table
- [ ] record the interim video while the cell is still set up
- [ ] write the final report

### The discipline that matters

Once numbers start going into the report, stop tuning. Tuning after seeing results, then reporting the improved numbers, is the most common way an honest project becomes a dishonest one without anybody deciding to do that.

If something needs fixing after the freeze, re-run **every** trial for both architectures, or report both sets separately and say why.

The harness was exercised end to end with fake agents, so the four results tables are known to render. Everything else needs trials, which need the arm.

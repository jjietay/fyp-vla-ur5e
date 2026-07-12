
**FYP: A Large Language Model-Based Interface for Robotics Systems using UR5e**

This document maps the software structure of the project: what each module does,
how they connect, and the design decisions behind the layout. It covers the
simulation and control layers built during the prep period.

---

## Design principle: one interface, swappable backends

The central architectural idea is that **the same public API drives both the
MuJoCo simulation and the real UR5e**. Two controller classes expose identical
methods:

| Backend | File | Talks to |
|---|---|---|
| Real robot | `src/fyp/controller.py` (`URController`) | URSim / physical UR5e via `ur_rtde` |
| Simulation | `src/fyp/sim/mujoco_controller.py` (`URControllerMuJoCo`) | MuJoCo physics |

Both provide `move_joints`, `move_to_pose`, `get_state`, `gripper_toggle`,
`gripper_start`, and `close`, with matching signatures and return types.

Because everything above the controller depends on the *interface* rather than a
specific backend, higher-level code (scripted primitives, the LLM planner,
demonstration recording) runs unchanged against sim or hardware. Prototype in
simulation, deploy on the real robot, no rewrite.

Conventions shared across both backends:

- **Gripper state:** integer `0 = closed`, `1 = open`.
- **TCP pose:** `[x, y, z, rx, ry, rz]`, axis-angle rotation vector (matches RTDE).

---

## Module reference

### Core (`src/fyp/`)

**`config.py`** — Loads `config/config.yaml` (robot host, motion defaults). Path
is anchored via a resolve so it works regardless of the working directory.

**`transforms.py`** — Coordinate-transform and pose-conversion helpers supporting
the kinematics work.

**`controller.py`** — Real-robot backend. Wraps `ur_rtde` (`RTDEControlInterface`
for commands, `RTDEReceiveInterface` for state, `RTDEIOInterface` for the
gripper). `move_to_pose` uses `moveL`; `move_joints` uses `moveJ`. IK is solved by
the UR firmware, so no explicit IK solver is needed on this path.

### Simulation (`src/fyp/sim/`)

**`mujoco_controller.py`** — MuJoCo backend, mirroring the real controller's API.
Loads the scene, holds `model` (blueprint) and `data` (mutable state), drives
joints via velocity-limited interpolation, reads TCP pose as axis-angle, and
controls the Robotiq gripper by writing actuator index 6 (`0` = open,
`255` = closed). Auto-detects a gripper via `nu > 6`, so it works with both the
plain and gripper-equipped scenes.

**`ik.py`** — Damped least-squares Jacobian inverse kinematics, **simulation-only**.
MuJoCo has no built-in IK, so when `move_to_pose` is called on the sim controller
this converts the Cartesian target into joint angles (via `mujoco.mj_jacSite`,
with damping for stability near singularities). The real robot never uses this —
its firmware solves IK internally.

**`demo_recorder.py`** — Passive data logger for the imitation-learning
(Architecture B) data pipeline. Three parts:
- `TimestepSnapshot` — dataclass for one moment (timestamp, joints, TCP, gripper,
  image).
- `DemoRecorder` — buffers snapshots (`start_episode`, `record`, `save_episode`)
  and writes them to HDF5 as per-field arrays.
- `RateControl` — drift-free fixed-rate loop helper (deadline-based sleep).

It is deliberately **passive**: it never drives the robot or reads sensors. A
caller hands it state via `record(...)`; it only files data away. This keeps it
format-agnostic (works with sim or real state).

**`sim_server.py`** — Teleoperation-and-record host (runs in Terminal 1). Opens
the MuJoCo viewer with built-in control sliders, listens on a TCP socket
(port 5555), and holds a `DemoRecorder` plus an offscreen `mujoco.Renderer` for
the fixed camera. When recording is armed it captures a snapshot at ~20 Hz.
Commands accepted: `get_state`, `move_joints`, `move_to_pose`, `gripper_toggle`,
`home`, `start_recording`, `stop_and_save`.

**`sim_client.py`** — Thin command sender (runs in Terminal 2). Each method opens
a socket, sends one JSON line, reads the reply, and closes. Holds no state — a
remote control whose methods mirror the server commands.

---

## Threading model (server)

MuJoCo's `data` is **not thread-safe**, so all simulation mutation must happen on
one thread. The server enforces this:

- **Main thread** — steps physics, syncs the viewer, records frames, and executes
  commands.
- **Listener thread** — accepts socket connections, parses JSON commands, and
  places each on a thread-safe queue, then blocks until the main thread returns a
  result.

Each main-loop tick drains the queue, so commands are executed on the main thread
only. The socket thread never touches `data` directly.

---

## Call graph

```
              sim_client.py        (Terminal 2 — you send commands)
                    |  JSON over socket (port 5555)
                    v
              sim_server.py        (orchestrator: steps sim, records, dispatches)
               |     |      |
      ---------      |      ------------------
      v              v                        v
mujoco_controller  demo_recorder        mujoco.Renderer
  .get_state()       .record()           (fixed-camera frame)
  .gripper_toggle()  .save_episode()
      |
      v
    ik.py            (only when move_to_pose is called)
```

`sim_server.py` is the conductor: it calls the controller for state and gripper
control, calls the recorder to store data, and renders the camera. For motion it
runs its own stepping loop (so it can sync the viewer and record each tick) using
the same interpolation approach as the controller. The controller calls `ik.py`
only for Cartesian moves. The recorder is purely a sink.

---

## Data flow: recording an episode

```
1. Terminal 1:  start sim_server  (viewer + sliders + socket + recorder)
2. Terminal 2:  client.start_recording()          -> server arms recording
3. Terminal 1:  drag sliders to teleoperate (arm + gripper)
                  -> server logs a snapshot every ~20 Hz:
                     joints, TCP pose, gripper state (read from slider),
                     fixed-camera image, timestamp
4. Terminal 2:  client.stop_and_save("data/episodes/ep_XXX.h5")
                  -> server flushes the buffer to HDF5
```

Each episode HDF5 contains, for N frames:

| Field | Shape | Type |
|---|---|---|
| `timestamps` | (N,) | float64 |
| `joint_positions` | (N, 6) | float64 |
| `tcp_poses` | (N, 6) | float64 |
| `gripper_states` | (N,) | int8 |
| `images` | (N, 240, 320, 3) | uint8 |

These episodes are the demonstration dataset for fine-tuning the
vision-language-action model (Architecture B). A separate converter (planned)
will transform them into the RLDS / LeRobot format, computing action deltas from
consecutive absolute states and converting axis-angle to Euler as needed — kept
separate from the recorder to preserve format-agnosticism.

---

## Assets (`assets/mujoco/`)

- **`ur5e/`** — `ur5e.xml` (bare arm), `scene.xml` (arm + table + ground + skybox
  + lighting + fixed camera + gripper-compatible physics options),
  `scene_gripper.xml` (generated arm+gripper composition, the scene actually
  loaded), and `assets/` holding arm `.obj` meshes and the copied gripper `.stl`
  meshes.
- **`robotiq_2f85/`** — `2f85.xml` (Robotiq 2F-85 model) and its `assets/` STL
  meshes. Source for the attachment step.

`scripts/attach_gripper.py` composes the arm and gripper (via MuJoCo's `mjSpec`
API) into `scene_gripper.xml`. Re-run it after editing the scene or gripper.

---

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `attach_gripper.py` | Compose UR5e + Robotiq 2F-85 into `scene_gripper.xml` |
| `record_replay_episode.py` | Standalone record + save + replay (trajectory plot + GIF) |
| `replay_episode.py` | Render a recorded episode's frames to an MP4 |

---

## Environment notes

- Onscreen viewer requires the X11 prefix on this machine (Wayland fix):
  `XDG_SESSION_TYPE=x11`.
- Offscreen / headless rendering uses `MUJOCO_GL=egl`.
- The two are mutually exclusive per run: the viewer path uses X11, the pure
  headless render path uses EGL.
- Package and environment management via `uv` (venv `universalrobot`).
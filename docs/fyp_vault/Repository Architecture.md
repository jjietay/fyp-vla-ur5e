## 1 Introduction

### 1.1 Architecture Tree
```
FYP/
├── src/           # main package code for real and sim controller
├── scripts/       # testing scripts
├── assets/        # MuJoCo scene/robot/gripper XML + mesh files
├── config/        # config.yaml (single source of truth for sim and real)
├── data/          # episodes/ and logs/ — recorded demonstrations
├── docs/          # architecture, commands references, working plan
├── tests/         # unit tests - Pytests
├── notebooks/     # Jupyter explorations (future)
├── ursim/         # URSim source files
└── media/         # Screen Recordings and screenshots of tests results, etc
```

### 1.2 Here are 2 main paths for recreating UR5e Movement in Sim
1) Scripted path via `scripts/record_replay_eposide.py`
2) Teleoperation via `sim_server.py` + `sim_client.py` + `demo_recorder.py`

### 1.3 More important files
```
- demo_recorder.py (the recorder itself that is recording the episode)
- record_replay_episode.py (recording the standalone scripted path)
- replay_episode.py (just reads HDF5 back to MP4)
- mujoco_controller.py (the sim's backend where the recorder pull state)
- ik.py (only relevant when move_to_pose() is used)
- sim_server.py ()
- sim_client.py
```


## 2 Scripts

### 2.1 attach_gripper.py

#### 2.1.1 Purpose
- this is a one-time script is to attach the Robotiq 2F-85 gripper to a particular scene loaded

#### 2.1.2 Imports and Global Variables
- imports mujoco because we need the **MjSpec** which is MuJoCo's programmatic model-editing API
- **MJSpec** is used for its ability to attach and replicate meta-elements in MJCF
- Currently we have both `scene.xml` and `2f85.xml` which are the 2 things we wanna attach
- By using **MJSpec** API, we can easily attach these 2 parts together
- We define `REPO`, `ARM_SCENE`, `GRIPPER`, and `OUT` where we wanna output this new `.xml` file
- We also define 2 hardcoded paths `ARM_MESHDIR` and `GRIP_MESHDIR` 
#### 2.1.3 Main Code
1) First, we parse `scene.xml` and `2f85.xml` using `mujoco.MJSpec.from_file()`
2) We grab the arm's `attachment_site` and use `attach_body` to attach the body's base mount to the arms
3) We then compile the model so that it becomes a runnable MJModel using `arm.compile()`
4) We then use `to_xml()` to create a single merged MJCF string

### 2.2 record_replay_episode.py

#### 2.2.1 Purpose
- This script is used to drive the UR5e in MuJoCo using a scripted joint-space trajectory via DemoRecorder, record the episode, save it as a HDF5, and replay

#### 2.2.2 Imports and Global Variables
- important import is `from fyp.config import get_config, resolve` to allow us to access the data in `config.yaml`

#### 2.2.3 Functions

`def render_frame()`
- this function is to render frames using mujoco built-in renderer
- it takes in the `mujoco.Renderer` and `mujoco.MJData` and outputs an ndarray of size `(H, W, 3)` of `uint8`
- renderer updates the scene using `renderer.update_scene`

`def waypoints()`
- this function returns a few joint configs for the robotic arm to move through

`def main()`
- this main function first creates an instance of `class URControllerMuJoCo` in mujoco_controller.py file, creates an instance of `class DemoRecorder` in demo_recorder.py file, another instance of `mujoco.Renderer` 
- It then uses a for loop, to move the arm through these already pre-configured waypoints and it saved a snapshot every `record_every` 25th step
- It moves using a linear interpolation
- it calls `mujoco.mj_step() once`, therefore advancing the simulation by exactly one timestep every `control_dt = 0.002` (2ms)
- `ctrl.data.ctrl[:6] = q_start + alpha*delta` only sets a target
- `mj_step()` actually moves the joints towards the target and updates `qpos`
- It records at a fixed rate, using the fixed value from config
- It uses DemoRecorder to `record` the key values of `joint_positions`, `tcp_pose`, `gripper_state`, and `image` which is stored in a `_buffer`
- `mujoco.mj_step(model, data)` advances the physics by one timestep
- It also uses `render_frame` that captures the frame and subsequently passed to the `_buffer`
- saves the tcp trajectory as a png image using matplotlib's plot
- also reopens the HDF5 to read-only and read timestamps, tcp_poses and images back into arrays (basically checking to see if file saved correctly and can be read back)
- also replay saved as gif


### 2.3 replay_episode.py
#### 2.2.1 Purpose
- Replay a recorded episode as a video
- It takes in a h5 file and outputs a mp4 file
- this is only used for teleop path, i.e. `sim_server` + `sim_client` + `demo_recorder`


## 3 Src

### 3.1 config.py
#### 3.1.1 Purpose
- `load_config()` returns a dictionary
- it requires `import yaml` for reading the `config.yaml` file
- it reads using `yaml.safe_load()`
- `get_config()` is the main function we need as it uses the `load_config` function
- we need to pass the config path to `get_config`
- `resolve()` is a helper function
- it takes a path written relative to the project root, and turns it into a full abosulte path on the disk

### 3.2 controller.py
#### 3.2.1 Purpose
- this connects to `ur_rtde` in init
- functions include `gripper_start` which sends power to the digital output pins using `rtde_io.setDigitalOutState()`, also using `rtde_r.getDigitalOutState()` that reads the current state of the IO and checks to make sure it turns on
- `move_to_pose` is used to `moveL` the arm
- `move_joints` is used to `moveJ` the arm
- `gripper_toggle(1 or 0)` is used to open or close the gripper by sending signal to digital `pin_control`
- `get_state` returns `joint_pos, `tcp_pose`, `gripper_state`
- `close` just exits cleanly using `rtde_c.StopScript()`

### 3.3 transforms.py
- empty for now

## 4 Src/Sim

### 4.1 demo_recorder.py
#### 4.1.1 Purpose
- this is the data-logging layer for imitationg learning in Architecture B
- it is to capture a sychronised stream of robot state + camera into an episode, then write it into a hdf5
- each episode is a sequence of timesteps and at every timestep it records 4 things, `joint_positions`, `tcp_pose`, `gripper_state`, `image`, and `timestamp`
- the recorder doesn't move anything
- the caller (scripted ``record_replay_episode.py`` or teleop `sim_server`) hands it's state via `record(...)` to demo_recorder.py
- this file is format-agnostic where this same recorder works against a simulation or the rea UR5e

#### 4.1.2 Classes
1) **TimestepSnapshot**
	- this is a `@dataclass` representing one moment in time
	- its a plain data container with five fields
	- `@dataclass` means Python auto-generates the `--init__`, `__repr__`, etc from those field annotations
2) **DemoRecorder**
	- the recorder itself
	- it owns the in-memory buffer (a list of `TimestepSnapshot`) and knows how to flush that buffer to HDF5

3) **RateControl**
	- a small drift-free fixed-rate loop helper
	- its job is to hold a loop at a target frequency (e.g. 20 Hz) using deadline-based sleeping
	- Currently unused

### 4.2 mujoco_controller.py
#### 4.2.1 Purpose
- its MuJoCo backend for the controller
- It exposes the exact same public API as the real `ur_rtde` and `URController`
- the only difference is entirely the implementation behind the shared interface
- `controller.py` sends commands over the wire to the UR Firmware using `ur_rtde` as the API but this file `mujoco_controller.py` **is** the physics engine and does the work itself, so it has to calculate inverse kinematics, etc

#### 4.2.2 Classes
- 1 single class: `URControllerMuJoCo` 
- This `URControllerMuJoCo` class in `mujoco_controller.py` can be compared with `URController` in `controller.py` 
- they both defined methods with the same name and same arguments
- any higher level code can write `ctrl.move_joints(...)` and it works whether `ctrl` is real or the sim version

### 4.3 ik.py
#### 4.3.1 Purpose
- provides inverse kinematics for the MuJoCo sim
- given a target TCP pose, find the point angles that put the TCP there
- this exist because MuJoCo has no built-in IK
- On real robot, the UR firmware already solves the IK internally

#### 4.3.2 Functions
- this file is a stateless module of pure functions
- IK is a computation, not a thing with state to hold, so no classes just all functions
- 2 helper functions:
	1) `_quant_to_axis_angle(quant)`
		- converts a MuJoCo quartenion `[w,x,y,z]` into a rotation vector `[rx,ry,rz]`
	2) `_pose_error(model, data, site_id, target_pos, target_mat)`
		- computes a 6-vector position and rotation
		- this is the quantity that the solver drives towards 0
- `solve_ik(model,data,site_id,target_pos,target_mat, q_init,max_iters,tol_damping,step_scale)`
		- this is the iterative solver that solves IK for the TCP site

#### 4.3.3 solve_ik

**MJModel**
1) `model.nu` --> number of actuators (nu > 6 to detect for gripper)
2) `model.nv` --> number of velocity DOF
3) `model.jnt_range` --> joint limits
4) `model.opt.timestep` --> the physics step size

**MJData**
1) `data.qpos` --> joint position right now
2) `data.qvel` --> joint velocities right now
3) `data.ctrl` --> the actuator commands you set
4) `data.site_xpos`, `data.site_xmat` --> the computed world position/orientation of the TCP site
5) `data.time` --> the simulation clock


### 4.4 sim_client.py
#### 4.4.1 Purpose
- thin remote control for the sim server
- the **server** running in Terminal 1 owns the MuJoCo viewer and physics (MuJoCo frontend)
- the **client** in Terminal 2 just sends commands to it over a TCP socket on port 5555
- The client holds **no simulation state at all**
- The client sends a request, the server carries it out and reports it back

#### 4.4.2 Classes
- one class `SimClient`
- it stores `host` and `port` defaulted from config file same as client

#### 4.4.3 Methods
- `_send(request)` opens a TCP socket and connects to `(host, port)`, and then sends the request as **one newline-terminated JSON line**, `json.dumps(request) + "\n"`
- reads the reply in 4096-byte-chunks until it sees a `"\n"`
- parse that line back into a dict and returns it

### 4.5 sim_server.py
#### 4.5.1 Purpose
- the orchastrator
- it owns the entire simulation, the MuJoCo controller, the viewer with its sliders (GUI), the `DemoRecorder`, and an offscreen renderer for the fixed camera
- It runs the physics loop continuously
- listens on the socket for client command/requests
- auto-record snapshots when recording is armed
- The server gaurantees that with a listerner-thread-plus-queu design, a command arriving over the socket never touches the sim directly

#### 4.5.2 Classes
- 2 classes: `_Job` and `SimServer`

`_Job`
- `Job` holds the `request` dictionary, an empty `result` slot, and a `threading.Event` called `Done`
- Its a hand-off token between the 2 threads
- the listener thread create a `_Job` and the main thread fills in `result` and flips `done`
- The `Event` is what lets the listeners block until the work is finished
- This is the mechanism that makes "the client waits while the server serves it"

`SimServer`
- the main class that holds the controller, recorder, renderer, the job `queue`, a `_stop` event, and the recording state (`_recording`, `_tick`)

#### 4.5.3 Functions

`__init__`
	- builds the controller, create the job queue, the recorder, the offscreen renderer, and the recording flags

`_serve`
- listener thread
- opens the server socket, accept connections, reads newline-delimited JSON, wraps each command in a `_Job`, puts it on the queue then calls `job.done.wait()`
- when the main thread signals `done`, it sends the results back over the socket, it never touches the sim

`run`
- this is the main thread/main loop
- starts the listener thread, opens the MuJoCo viewer, then loops while the viewer is open
- Each 'loop', it will drain job queue, calls `mj_step`, calls `_maybe_record`, sync the viewer and sleeps to pace real time
- this is the only place the sim is ever mutated

`_execute(req, viewer)`
- the command dispatcher
- runs only on main thread
- it switches on `req["cmd"]` for `get_state`, `start_recording`, `stop_and_save`, `move_joints`, `move_to_pose`, `gripper_toggle`, `home`
- everything is wreapped in try/except so failure come back as `{"ok": False, "error": ...}` instead of crashing the server

`_interp_move(q_target, speed, viewer)`
- server's own motion loop for `move_joints`/`move_to_pose`, interpolate to the target, `mj_step` each increment, record if armed, sync the viewer, and sleep to real time

`_maybe_record`
- called every tick
- if recording is armed, it counts every tick and grabs a snapshot of state + rendered camera into the recorder

`_gripper_from_ctrl`
- infers the 0/1 gripper state from the actual actuator value using `ctrl[6]`







































### 3.1 demo_recorder.py

- to record every episode
- since every episode contains many timestamps
- record these **4** things at every timestamps:
	1) tcp_pose
	2) gripper_state
	3) camera_image
	4) joint_positions
- contains **3** classes:
	1) **TimestepSnapshot**
		- This is just a `@dataclass`
		- It contains timestamp, joint_positions, tcp_pose, gripper_state and image
	2) **DemoRecorder**
		- This is the main class that records the demo
		- It contains 3 methods (excluding the `__init__`)
		- `start_episode` flushes the buffer, and starts the timer
		- `record` takes in arguments: joint_positions, tcp_pose, gripper_state and image, then passes those arguments into the arguments of an instance of the TimestepSnapshot dataclass, then appends this as snapshot into the `_buffer`
		- `save_eposide` takes in a save path, and stack per field arrays inot the h5 file, can image the file as columns of: timestamps, joint_positions, tcp_pose, etc
	3) **RateControl**
		- This is to control rate of capturing/recording the required information from a single episode
		- This is a helper function that contains 2 methods (excluding `__init__`)
		- `start` just starts the timer
		- if the loop is faster than our stipulated rate that we set, `wait` will just `time.sleep()` the difference in duration, if our loop runs slower than the rate we set, we will loop again (no interference)

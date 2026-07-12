## 1 Introduction

### 1.1 Architecture Tree
```
FYP/
├── src/fyp/       # main package code (controllers and shared helpers)
├── scripts/       # testing scripts
├── assets/        # MuJoCo scene/robot/gripper XML + mesh files
├── config/        # config.yaml (robot host, motion defaults)
├── data/          # episodes/ and logs/ — recorded demonstrations
├── docs/          # architecture notes, setup references, plan PDFs
├── tests/         # unit tests — Pytest
├── notebooks/     # Jupyter explorations (future)
├── ursim/         # URSim source files
├── media/
├── main.py
├── pyproject.toml
└── README.md
```

### 1.2 Here are 2 main paths
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

## 2 Files by file breakdown

### 2.1 demo_recorder.py

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

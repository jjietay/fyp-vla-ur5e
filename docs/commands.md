# Commands

Live commands only, **one copy of each**, ordered by when you run them.

This file used to carry a quick list at the top and detailed sections below, which meant every
command existed twice and the two copies drifted. Anything worth explaining is now a note
under the single copy.

Commands run from the FYP repo root unless they say `cd ~/lerobot`. Paths into the FYP repo
from that side are written out in full, because a `$FYP` shell variable breaks the moment you
open a new terminal without exporting it.

---

## Setup, once per machine

```bash
# FYP environment
uv sync

# lerobot environment, which is where torch lives
cd ~/lerobot && uv sync --extra dataset --extra training --extra smolvla

# the UR5e adapter, plus the drivers the inference client needs
cd ~/lerobot
uv pip install ur-rtde pyrealsense2
uv pip install -e /home/jj/Documents/NTU/Y4S1/FYP/integrations/lerobot_robot_ur5e
```

Check the adapter registered before wiring anything to the arm. If `ur5e` is missing from the
list, LeRobot's plugin discovery did not import it, and the distribution name is the first
thing to look at.

```bash
cd ~/lerobot
PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
uv run python -c "
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.robots import RobotConfig
register_third_party_plugins()
print(sorted(RobotConfig.get_known_choices()))
"
```

---

## Tests

Split by whether a device is needed, not by what the test covers. `unit/` runs on any machine
and is expected green at all times. `hardware/` needs the real arm answering on the RTDE port
at `robot.host`, and skips itself when nothing answers. **Running those MOVES THE ARM.**

```bash
uv run pytest tests/unit          # no device needed, the everyday one
uv run pytest tests               # adds tests/hardware, skipped without a controller
uv run pytest tests --integration # force the hardware tests to run and fail loudly
```

A file added to `tests/hardware/` needs `pytestmark = pytest.mark.integration` at module
level, or it errors on a laptop instead of skipping.

---

## Speech

```bash
uv run python scripts/listen.py             # push to talk, prints the transcript
uv run python scripts/listen.py --devices   # start here if clips come back silent
uv run python scripts/listen.py --once      # one utterance, then exit
uv run python scripts/listen.py --file clip.wav   # no microphone at all
uv run python scripts/listen.py --cpu       # no GPU, or CUDA libraries missing
```

No robot and no API key. Running it alone is the point: when a trial goes wrong in the lab you
need to know whether the microphone or the architecture was at fault.

`--cpu` measured a real time factor of 2.50 on `large-v3`, so it is for testing and not for
the demo.

---

## At the bench, in this order

Each step fails in a way that looks like the next one's problem. Read-only steps come first,
motion comes after, and calibration comes last. Replace `<ARM_IP>` with the address Koji gives
you.

These values were set against the dead simulator, or never set at all. Steps (11) to (16) fill
them in:

```
config/config.yaml
   8:  host: "127.0.0.1"       # MEASURE. Still the old simulator's address
  11:    power_pin: 1          # WRONG for this gripper, see (5) to (8) and (17) to (18)
  12:    control_pin: 2        # WRONG for this gripper
  13:  # home_q: [ , , , , , ] # MEASURE. Six joint angles. skills.home() refuses without it

  ################# WORKPLACE SAFETY BOX ##########################
  23:  x: [-0.700, 0.700]      # MEASURE, and err tight
  24:  y: [-0.700, 0.700]      # MEASURE
  25:  z: [ 0.005, 0.700]      # MEASURE. z min sits just above the table

  ##################### ORIENTATION OF GRIPPER POINTING DOWN ########################
  80:    down_rotvec: [3.142, 0.0, 0.0]  # MEASURE. See scripts/check_orientation.py
```

**The gripper is an OnRobot (RG2 or RG6), not a Robotiq 2F-85.** Its cable goes to the green
M8 plug on the wrist tool connector. `gripper_start()` and `gripper_toggle()` write to control
box pins 1 and 2, which do not reach the tool connector, so the current gripper code cannot
move it. Do not run Architecture A or the gripper test until the gripper code is rewritten.


### Find Out

(3) The arm IP address, and a free static IP address for the laptop on the same subnet

(4) Is RTDE on in the pendant security settings, and can you use Remote Control mode? The
control interface needs Remote Control mode on e-Series. The receive interface does not

(5) Which OnRobot gripper is it? Read the label for RG2 or RG6

(6) Is there an OnRobot Compute Box, and what is its IP address?

(7) Is the OnRobot URCap installed? Ask him to open and close the gripper from the pendant

(8) Are the TCP offset and the payload in the installation file set for the gripper plus the
quick changer? `getActualTCPPose` reports the point the pendant defines, so a wrong TCP makes
every pose wrong, and the long RG fingers make the error large

(9) Where can the camera be taped? Ask him to show you the e-stop, freedrive and the speed
slider

### Connection and read-only work, about 45 min

(10) Static IP on the laptop, then check the arm answers

```bash
ip link                                  # find the Ethernet name, for example enp3s0
sudo ip addr add <LAPTOP_IP>/24 dev <ETH_NAME>
ping -c 3 <ARM_IP>
nc -zv <ARM_IP> 30004                    # RTDE port
```

(11) Set `robot.host: "<ARM_IP>"` in `config/config.yaml`

(12) Read the joint angles and the TCP pose. Receive interface only, so it cannot move the arm
and does not stop a pendant program. Steps (14) to (16) reuse this command

```bash
uv run python -c "import rtde_receive as r; x=r.RTDEReceiveInterface('<ARM_IP>'); print('q', x.getActualQ()); print('tcp', x.getActualTCPPose())"
```

Do not use `URController` for this. Its constructor opens the control interface too, which
uploads a script and takes the arm.

(13) Down pose. Freedrive on, hand-guide the gripper to point straight down in the pose you
would pick from, release freedrive so the arm holds, and keep your hands off it. The **length**
of RX, RY, RZ on the Move tab should be near 3.14, not any one of them

```bash
uv run python scripts/check_orientation.py --seconds 10
```

*Stable* means Euler is safe and nothing changes. *Unstable* means the rotation format has to
change before recording anything.

(14) `down_rotvec`. Take the middle of each range from (13), or the last three numbers from
(12) in the same pose, and paste them into `architecture_a.skills.down_rotvec`

(15) `home_q`. Freedrive to a safe home pose, run (12), and paste the six joint angles into
`robot.home_q`

(16) Workspace box. Freedrive to each corner of the usable table area, run (12) at each, and
write tighter x, y and z limits into `workspace`. z min sits just above the table

### Gripper facts, about 15 min

(17) If there is a Compute Box, check it answers on the Modbus port

```bash
ping -c 3 <BOX_IP>
nc -zv <BOX_IP> 502
```

(18) If there is no Compute Box, make a pendant program with one gripper open node and one
close node. Save it, then copy the `.urp` and the `.script` files to the USB stick. The
`.script` file holds the exact URScript the URCap sends, which the new gripper code will copy

### First motion, about 45 min

(19) Pendant in Remote Control mode, speed slider low, Koji at the e-stop

(20) Controller tests, without the gripper test

```bash
uv run pytest tests/hardware/test_controller.py -k "not gripper" -v
```

The `controller` fixture no longer starts the gripper, so with `-k "not gripper"` this run
writes to no digital output at all. Only the gripper tests ask for `gripper_start()`, and they
are expected to fail until the gripper code fits the OnRobot

(21) Servo test. First time `servoL` goes to a real arm, and Architecture B depends on it

```bash
uv run pytest tests/hardware/test_servo.py -v
```

### Camera and calibration, stretch, about 60 min

(22) Camera opposite the arm, centred, 0.7 to 0.8 m above the **table surface**, at about 45
degrees. USB 3 cable in a USB 3 port. Tape the tripod feet, because this comes down between
sessions

```bash
uv run python scripts/check_camera.py
```

Good: USB says **3.2**, format matches what you asked for, coverage above 50 percent, centre
pixel back-projects to **positive** z.

Negative z means the OpenGL convention is back in `pixel_to_3d.py`. Stop and fix it. Low
coverage is the scene, not the camera: use matte objects, and nothing under 0.3 m returns
anything.

Writes `data/cache/frame_top.png`, which the detector check needs later.

(23) Measure the mount

- **jog the TCP in +x and watch which way it goes** before measuring anything. Guessing the
  base axes is a 90 degree error that looks like a calibration fault
- measure from the arm base to the **middle lens**, the RGB one. The imager sits a few mm
  behind the glass and that offset is unpublished, so measure to the glass
- square the tripod to the table edge. One degree is about 17 mm at 1 m

(24) Write it. Metres and degrees. Refuses to overwrite without `--force`

```bash
uv run python scripts/write_calibration.py --xyz X Y Z --rpy ROLL PITCH YAW
```

(25) Prove it. **The real run moves the arm**, so Koji stays at the e-stop

```bash
uv run python scripts/check_calibration.py --dry-run   # compute targets, do not move
uv run python scripts/check_calibration.py --points 5
```

Run `--dry-run` first. If the printed positions are not somewhere on your table, (23) is wrong
and there is no point moving anything.

Then the real run: click an object, it hovers the TCP above where it thinks the object is, and
**you measure the miss with a rule and type it in**. Manual on purpose, because anything the
software measured would come from the transform being tested.

Expect 1 to 3 cm. Large mean with small scatter means the six numbers are slightly off, so
re-measure the angles. Large scatter means depth alignment or the datum, not the numbers.

If it is too large to grasp with, the fallback is the ping-pong ball in the gripper and a
fitted transform, reusing code that already exists.

### Last 15 min

(26) Photos of the taped positions, the gripper label, and the pendant TCP and payload screens

(27) Commit the config and calibration, then take the camera down

```bash
git add config/
git commit -m "Lab session: robot host, home_q, workspace, down_rotvec, calibration"
```

If (22) to (25) do not fit, do them first in recess week, before Architecture A. Architecture
A cannot run without `T_base_cam`.

## Architecture A

```bash
uv run python scripts/run_architecture_a.py    # speak to it. The default, and what a trial uses
uv run python scripts/run_architecture_a.py --text
uv run python scripts/run_architecture_a.py --instruction "place the red cube in the tray"
```

`--text` and `--instruction` skip loading the recogniser entirely, which is worth several
seconds per debugging run. Needs `ANTHROPIC_API_KEY` set.

Detector check, which runs in the lerobot venv because that is where torch lives. It needs an
image, so `check_camera.py` has to have run first.

```bash
cd ~/lerobot
uv run python /home/jj/Documents/NTU/Y4S1/FYP/scripts/check_detector.py \
  --image /home/jj/Documents/NTU/Y4S1/FYP/data/cache/frame_top.png \
  --queries "red cube" "metal tray" "orange juice" "water bottle" "glass" \
  --model google/owlv2-base-patch16-ensemble --threshold 0.3 \
  --nms-iou 0.5 --top1-per-query \
  --out  /home/jj/Documents/NTU/Y4S1/FYP/data/cache/detections_owlv2.png \
  --json /home/jj/Documents/NTU/Y4S1/FYP/data/cache/detections.json
```

- `--model` and `--threshold` are optional now. They default to
  `architecture_a.detector` in `config.yaml`, so the script and the pipeline cannot disagree
  about what the detector is. They are written out above only so the command records which
  combination produced a given overlay
- NMS is class-agnostic by default, since OWLv2 fires two labels on one object and per-query
  suppression would keep both
- drop `--top1-per-query` once a scene can hold more than one instance of a query

---

## Episodes

No episodes exist yet, so treat `ep_001.h5` as a template rather than a file.

```bash
uv run python scripts/replay_episode.py data/raw/episodes/ep_001.h5
xdg-open data/raw/episodes/ep_001.mp4
```

First and last frame plus gripper transitions, for a quick sanity look before trusting a
recording session:

```bash
uv run python -c "
import h5py, numpy as np
import matplotlib.pyplot as plt
with h5py.File('data/raw/episodes/ep_001.h5', 'r') as f:
    imgs = f['images'][:]; grip = f['gripper_states'][:]; tcp = f['tcp_poses'][:]
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(imgs[0]);  ax[0].set_title('frame 0');  ax[0].axis('off')
ax[1].imshow(imgs[-1]); ax[1].set_title('frame -1'); ax[1].axis('off')
plt.savefig('data/cache/ep_001_check.png', dpi=100)
print('gripper transitions at frames:', np.where(np.diff(grip) != 0)[0])
print('TCP z range:', round(float(tcp[:,2].min()),3), '->', round(float(tcp[:,2].max()),3))
"
```

Export to a LeRobot dataset:

```bash
cd ~/lerobot
PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
uv run --extra dataset --with h5py \
python /home/jj/Documents/NTU/Y4S1/FYP/scripts/export_lerobot.py \
  --episodes /home/jj/Documents/NTU/Y4S1/FYP/data/raw/episodes \
  --repo-id  jj/ur5e_pickplace \
  --task     "pick and place the block" \
  --camera   top \
  --root     /home/jj/Documents/NTU/Y4S1/FYP/data/processed/lerobot_ur5e
```

---

## Architecture B: training

`--steps=1` is the plumbing dry run. Raise it, and drop `--batch_size=2`, for a real run.

```bash
cd ~/lerobot && uv run --extra dataset --extra training lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=jj/ur5e_pickplace \
  --dataset.root=/home/jj/Documents/NTU/Y4S1/FYP/data/processed/lerobot_ur5e \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=1 \
  --output_dir=outputs/train/dryrun_smolvla \
  --job_name=dryrun \
  --policy.device=cuda \
  --wandb.enable=false
```

`smolvla_base` expects cameras named `camera1`, so a single camera needs the `--rename_map`.
`push_to_hub` defaults true and demands a destination repo, hence the explicit false.

---

## Architecture B: inference on the arm

Two processes, both LeRobot's own. `policy_server` runs the checkpoint, `robot_client` runs
the arm through our adapter. Reasoning in `docs/plan_vault/Decisions to Make.md`. Both run out of the
lerobot venv, because the client is the half that touches hardware.

Terminal 1, the policy. `--host` is the interface this server **binds** to, not a robot
address, and it stays loopback because both processes run on this machine.

```bash
cd ~/lerobot
uv run --extra smolvla python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=20
```

Terminal 2, the arm. **This moves a physical UR5e**, so read the `workspace` block in
`config.yaml` first and stand near the estop.

```bash
cd ~/lerobot
PYTHONPATH=/home/jj/Documents/NTU/Y4S1/FYP/src \
uv run --extra smolvla python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=ur5e \
  --robot.id=ur5e_lab \
  --robot.fyp_config=/home/jj/Documents/NTU/Y4S1/FYP/config/config.yaml \
  --robot.camera=top \
  --task="pick and place the block" \
  --policy_type=smolvla \
  --pretrained_name_or_path=outputs/train/<run>/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --fps=20 \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average
```

- **two unrelated `127.0.0.1` values live here, and confusing them costs an afternoon.**
  `robot.host` in `config.yaml` is the ARM's address and becomes the real UR5e IP.
  `--host` on the policy server is the interface that server binds to and stays loopback.
  Pointing it at the arm's IP fails to start, because the laptop does not own that address
- to run the policy on another machine, the server takes `--host=0.0.0.0` and the client's
  `--server_address` becomes that machine's LAN address. Those two always change together,
  and the ports have to match
- `--robot.camera` must match the `--camera` used at export, because the checkpoint expects
  `observation.images.<name>` and anything else is a `KeyError` on the first observation
- `--fps` on both processes and `robot.servo.time` in `config.yaml` all describe the same
  control period. Disagreeing values make the arm track a target it thinks is further away
  in time than it is
- `--task` is the instruction string, so drive it from the paraphrase sets in
  `evaluation/suite.py` rather than typing a new sentence each run
- the arm stops when the client stops, because `disconnect` calls `servo_stop`. Killing the
  terminal without a clean exit leaves the servo stream open and the arm creeping

---

## Scoring

```bash
uv run python scripts/run_evaluation.py --architecture A --tier 0 --trials 20
uv run python scripts/run_evaluation.py --tier 0 --layouts-only   # no robot needed
uv run python scripts/run_evaluation.py --summarise               # rebuild the tables and exit
```

---

## Troubleshooting

### `VIRTUAL_ENV does not match the project environment path`

Expected on every `cd ~/lerobot` command while the FYP venv is active in your shell. It means
`uv` is correctly using the lerobot environment instead of the active one, which is what you
want. Run `deactivate` first if you would rather not see it.

### `Library libcublas.so.12 is not found or cannot be loaded`

`faster-whisper` runs on CTranslate2, which needs **cuBLAS for CUDA 12 and cuDNN 9** and
dlopens them by bare name, so it searches only the loader path. The symptom is misleading:
the model loads, reports `large-v3 on cuda/float16`, and fails on the first utterance.

Both libraries are project dependencies now, so `uv sync` fixes it, at about 1.5 GB.
`shared/speech.py` preloads them by absolute path before CTranslate2 starts, so **no
`LD_LIBRARY_PATH` export is needed**. Two fallbacks if the download is inconvenient:

```bash
# borrow the lerobot venv's copies, since torch ships them
export LD_LIBRARY_PATH=$HOME/lerobot/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$HOME/lerobot/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib

# or run on CPU, which works everywhere and is far slower
uv run python scripts/listen.py --cpu
```

### A clip comes back silent

`sounddevice` needs PortAudio present, `sudo apt install libportaudio2` on Ubuntu. A USB
headset does not become the default input just by being plugged in, so `--devices` is the
first thing to try.

### The hardware tests error instead of skipping

`tests/conftest.py` is missing, or a file in `tests/hardware/` lacks its `pytestmark`.

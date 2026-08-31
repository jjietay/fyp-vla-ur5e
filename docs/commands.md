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

Each step fails in a way that looks like the next one's problem.

### 0. Before power

These eight values were set against the dead simulator, or never set at all:

```
config/config.yaml
   8:  host: "127.0.0.1"       # MEASURE. Still the old simulator's address
  11:    power_pin: 1          # CONFIRM against the real wiring before powering it
  12:    control_pin: 2        # CONFIRM
  13:  # home_q: [ , , , , , ] # MEASURE. Six joint angles. skills.home() refuses without it
  
  ################# WORKPLACE SAFETY BOX ##########################
  23:  x: [-0.700, 0.700]      # MEASURE, and err tight 
  24:  y: [-0.700, 0.700]      # MEASURE
  25:  z: [ 0.005, 0.700]      # MEASURE. z min sits just above the table

  ##################### ORIENTATION OF GRIPPER POINTING DOWN ########################
  80:    down_rotvec: [3.142, 0.0, 0.0]  # MEASURE. See scripts/check_orientation.py
```

- [ ] `host`, the arm's IP
- [ ] the two gripper pins. **Ask** which outputs it is wired to. `gripper_start()` writes to
      them the moment Architecture A runs, so do not find out by toggling
- [ ] USB 3 cable in a USB 3 port. USB 2 connects, streams, and silently caps resolution
- [ ] camera opposite the arm, centred, 0.7 to 0.8 m above the **table surface**
- [ ] tape the tripod feet. Shared lab, this comes down between sessions

`down_rotvec` comes from step 2. The workspace box and `home_q` come from jogging the arm once
it is powered.

### 1. Camera

```bash
uv run python scripts/check_camera.py
```

Good: USB says **3.2**, format matches what you asked for, coverage above 50 percent, centre
pixel back-projects to **positive** z.

Negative z means the OpenGL convention is back in `pixel_to_3d.py`. Stop and fix it. Low
coverage is the scene, not the camera: use matte objects, and nothing under 0.3 m returns
anything.

Writes `data/cache/frame_top.png`, which the detector check needs later.

### 2. Down pose, which gives you two things

1. Freedrive on, hand-guide the gripper to point straight down in the pose you would pick from
2. Release freedrive so the arm holds. No hand on it
3. Check RX, RY, RZ on the Move tab. Their **length** should be near 3.14, not any one of them

```bash
uv run python scripts/check_orientation.py --seconds 10
```

Read-only, cannot move the arm.

**Verdict.** *Stable* means Euler is safe and nothing changes. *Unstable* means the rotation
format has to change before recording anything.

**And your real `down_rotvec`.** Take the middle of each printed range and paste the three
numbers into `architecture_a.skills.down_rotvec`.

### 3. Measure the mount

- [ ] **jog the TCP in +x and watch which way it goes** before measuring anything. Guessing the
      base axes is a 90 degree error that looks like a calibration fault
- [ ] measure to the **middle lens**, the RGB one. The imager sits a few mm behind the glass and
      that offset is unpublished, so measure to the glass
- [ ] square the tripod to the table edge. One degree is about 17 mm at 1 m

### 4. Write it

```bash
uv run python scripts/write_calibration.py --xyz X Y Z --rpy ROLL PITCH YAW
```

Metres and degrees. Refuses to overwrite without `--force`.

### 5. Prove it

```bash
uv run python scripts/check_calibration.py --dry-run   # compute targets, do not move
uv run python scripts/check_calibration.py --points 5
```

Run `--dry-run` first. If the printed positions are not somewhere on your table, step 3 is
wrong and there is no point moving anything.

Then the real run: click an object, it hovers the TCP above where it thinks the object is, and
**you measure the miss with a rule and type it in**. Manual on purpose, because anything the
software measured would come from the transform being tested.

Expect 1 to 3 cm. Large mean with small scatter means the six numbers are slightly off, so
re-measure the angles. Large scatter means depth alignment or the datum, not the numbers.

If it is too large to grasp with, the fallback is a matte ball in the gripper and a fitted
transform, reusing code that already exists. Worth having a ping-pong ball in your bag.

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

- always pass `--model` and `--threshold` explicitly, because the script defaults to OWL-ViT
  at 0.1, which is the combination that does not work
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
the arm through our adapter. Reasoning in `docs/plan_vault/Decisions.md`. Both run out of the
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

# CLAUDE.md

NTU EEE FYP A1005-261. User speaks a command, a UR5e does it. The deliverable is a
**comparison of two architectures** on the same cell, not a working robot.

**`docs/fyp_plan.md` is authoritative** for scope, deliverables and schedule. This file
is only what you need to not break things. If the two disagree, the plan wins.

- Target: all work **complete** by 10 Nov 2026. Submissions stay on official NTU dates in 2027.
- Current tasks: `docs/plan_vault/To Do (Non-Technical).md`
- Vault: `docs/plan_vault/` (Obsidian, 12 flat notes) mirrors the plan for tracking.
  Start Here, Weekly Focus, Flowchart of Architecture and Dependencies, FYP Submissions,
  To Do (Non-Technical), Workstreams, Tasks, Decisions to Make, Hypotheses for Comparison,
  Evaluation of Architecture, Different Risk in Project, Research Papers for Reading.
  No subfolders, and a new decision is a `##` section rather than a new file.

## Repo map

```
src/fyp/
├── shared/           CONTROLLED VARIABLE. Both architectures, at runtime.
│   ├── helpers/      rotations, transforms, config, rate_limiter
│   ├── hardware/     ur5e_controller, camera, safety, intrinsics
│   └── speech.py     push-to-talk ASR. Outside both architectures on purpose
├── architecture_a/   modular. Nothing trained.
│   ├── perception/   detector, filters, localiser, pixel_to_3d
│   ├── calibration/  hand_eye  <- A only; B never forms a 3D point
│   ├── skills.py     pick/place/pour/open_drawer, the only things A can do
│   ├── tools.py      schemas the LLM sees; must match skills.py signatures
│   ├── planner.py    LLM + plan validation + ask_user + query grounding
│   ├── pipeline.py   the orchestrator
│   └── trace.py      per-stage failure log <- the H6 evidence
├── architecture_b/   end-to-end SmolVLA. Nothing hand-authored.
│   ├── demos/        recorder, hdf5_store, lerobot_export
│   └── policy_action.py  clamp, delta maths, envelope check. ALL of B's inference
│                     logic, kept here rather than in the adapter so it is testable
└── evaluation/       the harness. Imports NEITHER architecture, on purpose.
    ├── suite.py      tiers, utterances, layouts, success criteria. FROZEN.
    └── harness.py    trial runner + results tables

scripts/              thin CLIs: check_detector, export_lerobot, replay_episode,
                      run_architecture_a, run_evaluation
config/               config.yaml + calibration/ (tracked on purpose)
docs/                 fyp_plan.md, plan_vault/, guidelines/ (NTU specs)
```

**`shared/` admission rule:** a module goes there only if **both** architectures use it at
runtime, not merely because it is generically useful. Pure maths used by one side stays with
that side. Test: delete `architecture_b/` and `architecture_a/` should still run.

Violating this quietly destroys the fairness argument the whole FYP rests on.

## Commands

```bash
uv sync
uv run pytest tests/unit                             # no device needed, always green
uv run pytest tests                                  # adds tests/hardware, which skip without a robot

uv run python scripts/run_architecture_a.py --instruction "put the red cube in the tray"
uv run python scripts/run_evaluation.py --tier 0 --layouts-only   # no robot needed
uv run python scripts/run_evaluation.py --architecture A --tier 0
```

**`ANTHROPIC_API_KEY` must be set** or Architecture A cannot run a single instruction:
the planner and the grounding step both need it.

`uv run` for everything. Python 3.12+. There is a separate lerobot venv at `~/lerobot` for
torch work. `tests/hardware/` needs the real UR5e answering on RTDE at `robot.host`, and
running those tests MOVES THE ARM. There is no simulator any more.

## Landmines

Each of these has already cost time. Verified as of 11 Aug 2026.

- **A stale `.git/index.lock` blocks every git operation.** If git says "Another git process
  seems to be running", check the file first: `ls -la .git/index.lock`. **Zero bytes and more
  than a few minutes old means it is stale**, because a live process writes to it. Delete it
  and carry on. Any agent running git through a sandboxed file bridge should use
  `GIT_OPTIONAL_LOCKS=0 git status`, since a bridge that cannot unlink will leave the lock
  behind and lock the user out of their own repo.

- **`pytest tests` should be GREEN.** Tests split by whether a device is needed:
  `tests/unit/` needs nothing, `tests/hardware/` needs a UR controller. The hardware ones
  carry the `integration` marker and auto-skip when nothing answers on `robot.host`, port 30004.
  Force them with `--integration`. If you see 5 errors instead of 5 skips, `tests/conftest.py`
  is missing. A new file in `tests/hardware/` without `pytestmark` turns the suite red on a
  laptop, which is the one mistake this layout does not prevent by itself.
- **Everything stays in base frame.** `get_state` reports the TCP in base frame and `moveL`
  expects targets in it. Do not invent a world frame or mix in coordinates measured against
  the table without converting first. See the `hand_eye.py` header.
- **Never hardcode `parents[n]` to find the repo root.** `shared/helpers/config.py` walks up
  to `pyproject.toml` instead, because a hardcoded depth silently broke during the
  architecture split and every path in `config.yaml` resolves through it.
- **`data/` is gitignored and has no backup.** Only `ep_001.h5` survives, recovered from
  history after a reformat wiped it. Deleting anything there is permanent. Never `rm` under
  `data/` without saying so first.
- **faster-whisper on the RTX 50-series** must use `compute_type="float16"`; the default int8
  crashes with `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120.
- **`detect()` reloads the model on every call.** Fine for `check_detector.py`, fatal in a
  loop. `ArchitectureA` is built once and reused for exactly this reason.
- **Two config values are still placeholders and WILL misbehave on hardware:**
  `workspace.x` and `workspace.y` (wait for the table layout to be agreed with Koji), and
  `open_drawer(pull_axis=...)`. Measured on 23 Sep 2026 and in `config.yaml`: `robot.host`,
  `robot.home_q`, `workspace.z`, `workspace.max_radius` and `skills.down_rotvec`.
- **Read poses from RTDE, never from the pendant.** The pendant shows TCP z in another frame
  (-0.020 where RTDE reported 0.380 on 23 Sep).
- **The RG2 gripper is driven by URScript through `sendCustomScriptFunction`.** The pendant
  must be in Remote Control mode. Each call stops the ur_rtde control script, so it ends any
  open servo stream, and it returns only after the fingers stop: about 0.2 s with no motion,
  about 1.5 s for a full stroke. The grip-detected tool input is unreliable, so nothing may
  use it. The protocol numbers in `ur5e_controller.py` are protocol facts, kept out of config
  on purpose.
- **Everything in `skills.py` goes through `WorkspaceEnvelope` first**, including intermediate
  approach and retreat waypoints. Never add a motion path that skips it.
- **Every joint move goes through `WorkspaceEnvelope.check_joint_move` first.** UR joints turn
  +/-360 degrees, so a `moveJ` to the other joint value of the same wrist angle spins the wrist
  a full turn. `skills.home()` and `go_home.py` both use it. Rotations are built with rotation
  matrices, never by adding angles to rotation-vector or Euler numbers.
- **OWLv2 takes short noun phrases, never sentences.** It encodes each query into ONE CLIP
  embedding matched against image patches, so a sentence has no corresponding region and the
  match is meaningless. This is why `planner.py::extract_queries` exists at all. The 77-token
  CLIP limit is real but is not the binding constraint: a command fits and still fails.
  Reasoning in `architecture_a/perception/detector.py` and `docs/plan_vault/Decisions to Make.md`.
- **Never pass `queries=` to `ArchitectureA` for a recorded trial.** It pins the detector
  vocabulary and skips grounding, which hands A a list of what is on the table that B never
  receives. Debugging aid only. `run_evaluation.py` deliberately passes nothing.

## What exists right now (23 Sep 2026)

Week 7 of 13. The arm became available about 22 Sep, after the lab move, and the first
session at the real UR5e was 23 Sep.

**Working on real hardware:** the speech front end; `URController` motion (`go_home.py`
reached `home_q` within 0.012 degrees, and `tests/hardware -k "not gripper"` passed 7 of 7
twice); the RG2 gripper protocol (`scripts/check_gripper.py`); `check_orientation.py`.

**Written, not yet run on hardware:** the rewritten gripper control with its controller-side
check and one resend (23 Sep); Architecture A's skills, perception and planner on the real
cell; `shared/hardware/camera.py` on the cell (benched without the arm on 1 Sep); the
evaluation harness; Architecture B's robot side. `pytest tests/unit` collects 233 as of
23 Sep 2026, all green. `tests/hardware/test_orientation.py` (not yet run) checks that
rotations at the down pose take the short way.

**Settled on 23 Sep:** Architecture B's state stores rotation as the first two columns of the
rotation matrix (10 numbers), because Euler roll flipped on the real arm. The action stays
7 numbers with an Euler rotation delta. `WorkspaceEnvelope` also limits the horizontal
distance from the base axis to `workspace.max_radius`.

**Architecture B, 31 Aug:** inference runs on LeRobot's own `policy_server` and `robot_client`,
driving the UR5e adapter at `integrations/lerobot_robot_ur5e/`, which is installed into the
lerobot venv rather than this one. `URController` has a non-blocking `servo_to_pose`,
because `moveL` blocks and a 20 Hz policy loop built on it waits every cycle. There is no
Architecture B agent for the evaluation harness yet: `run_evaluation.py --architecture B`
prints a message and exits.

**Not done:** the camera on the cell and `T_base_cam`, the detector on real frames, the
workspace x and y bounds, the e-stop recovery test, and all demonstration capture. There are
zero episodes. Demonstrations will be driven with a 3D jog device, not freedrive.

**No fiducial markers, anywhere.** `T_base_cam` is written by hand into
`config/calibration/T_base_cam.json` from the measured camera position, not fitted. The lab
is shared, so the camera is taped in place and re-measured each session.
`solve_rigid_transform` in `hand_eye.py` stays in the repo as the documented fallback (matte
ball in the gripper, still tag-free) if the measured transform proves too coarse to grasp
with. Expect a systematic 1-3 cm bias: 1 degree of mount orientation error is ~17 mm at a 1 m
standoff. The reach test is the only check on stage 4, because a hand-measured transform
produces no residual of its own. Reasoning in `docs/plan_vault/Decisions to Make.md`.

## Evaluation harness

`src/fyp/evaluation/suite.py` is the experimental design and it is **frozen**. It was written
before either pipeline was tuned. Changing a tier, an utterance or a success criterion after
tuning begins means the benchmark has been fitted to whichever architecture was built first.
If a change is genuinely unavoidable, append to `CHANGELOG` at the bottom of that file with a
date and a reason, and say so in the report.

The harness must never import `architecture_a` or `architecture_b`. It talks to an agent
through one method, `run(instruction) -> outcome`. A test enforces this.

## simulation/

A local archive folder that git ignores completely. **It is not part of this project.** JJ is
finished with it. Never read it, never raise it, never debug it, never spend a tool call on
it. Treat anything inside as if it does not exist.

## Conventions

**Docstrings open with what the thing does in one plain line**, in the form
*"it takes X and gives you Y"*, then detail underneath. Never lead with mechanism.

```python
def solve_rigid_transform(p_cam, p_base):
    """
    It takes matched 3D points seen in the camera and the same points measured
    in the robot base frame, and gives you the 4x4 transform T_base_cam.
    """
```

**Comments explain why, not what.** The existing code documents traps and reasoning, not
line-by-line behaviour. Match that.

**Vault notes** (`docs/plan_vault/`) follow a strict house style: no em dashes or en dashes,
bullets never end in a full stop, one concept per note, sparse wikilinks. Verify with:

```bash
find docs/plan_vault -name '*.md' -print0 | xargs -0 python3 <skill>/scripts/check_notes.py --vault docs/plan_vault
```

## Working with JJ

- Be concise and direct. He notices padding and dislikes it.
- **Ask before restructuring or any multi-file change.** He has reformatted the repo himself
  several times and has his own view of the layout.
- Don't act on the backlog unasked. Report findings, let him choose.
- He is doing this solo to learn it, so explain reasoning rather than only handing over code.
- When a shell command fails on the mounted folder with "Operation not permitted", it is a
  sandbox delete restriction, not a real error.

## Verify before claiming done

There is no CI. After any move or rename, run this and expect `none`:

```bash
python3 - <<'PY'
import ast, pathlib
root = pathlib.Path('.'); mods = set()
for p in (root/'src').rglob('*.py'):
    if '__pycache__' in p.parts: continue
    m = '.'.join(p.relative_to(root/'src').with_suffix('').parts)
    mods |= {m, m.removesuffix('.__init__')}
bad = [(str(p), n.lineno, m) for p in list((root/'src').rglob('*.py')) + list((root/'scripts').rglob('*.py')) + list((root/'tests').rglob('*.py'))
       if '__pycache__' not in p.parts
       for n in ast.walk(ast.parse(p.read_text()))
       for m in ([n.module] if isinstance(n, ast.ImportFrom) and n.module else
                 [a.name for a in n.names] if isinstance(n, ast.Import) else [])
       if m.split('.')[0] == 'fyp' and m not in mods]
print("BROKEN:", bad or "none")
PY
```

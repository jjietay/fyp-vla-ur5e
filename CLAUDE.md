# CLAUDE.md

NTU EEE FYP A1005-261. User speaks a command, a UR5e does it. The deliverable is a
**comparison of two architectures** on the same cell, not a working robot.

**`docs/fyp_plan.md` is authoritative** for scope, deliverables and schedule. This file
is only what you need to not break things. If the two disagree, the plan wins.

- Target: all work **complete** by 10 Nov 2026. Submissions stay on official NTU dates in 2027.
- Current tasks: `docs/fyp_plan.md` §12, Open actions
- Decisions and their reasoning: `docs/adr/`, one file per decision. The Obsidian vault
  (`docs/plan_vault/`) was removed on 16 Oct 2026; its last state is in git at `7bb85a6`.

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
docs/                 fyp_plan.md, adr/, guidelines/ (NTU specs)
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

uv run python scripts/run_evaluation.py --tier 0 --layouts-only   # no robot needed

# Architecture A needs the lerobot venv (OWLv2 needs PyTorch). Speech is the ONLY
# input (13 Oct 2026): instructions, clarification answers and evaluation trials
# are all spoken; there is no --text or --instruction any more.
PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/run_architecture_a.py
PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/run_evaluation.py --architecture A --tier 0
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
- **`data/` is gitignored and has no backup.** It holds camera captures in `data/cache/` and
  no episodes yet; the old `ep_001.h5` was simulation output and was deleted on purpose.
  Deleting anything there is permanent. Never `rm` under `data/` without saying so first.
- **faster-whisper on the RTX 50-series** must use `compute_type="float16"`; the default int8
  crashes with `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120.
- **`detect()` loads OWLv2 once per process** and reuses it (`_load` in `detector.py`, 13 Oct
  2026). Before that it reloaded on every call: 10.7 s median per detection on 1 Oct, inside
  A's re-perception loop, all of it counted as A's latency. Never move the loading back into
  `detect()`. The first detection in a process still pays the load.
- **One value is still a placeholder and WILL misbehave on hardware:**
  `open_drawer(pull_axis=...)`. Measured and in `config.yaml`: `robot.host`, `robot.home_q`,
  `workspace.z`, `workspace.max_radius` and `skills.down_rotvec` (23 Sep 2026), and
  `workspace.x` and `workspace.y` (1 Oct 2026, the bounding box of the table corners, so part
  of the box lies off the table, which sits about 41 degrees to the base axes).
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
  Reasoning in `architecture_a/perception/detector.py` and `docs/adr/0004-owlv2-takes-noun-phrases.md`.
- **Never pass `queries=` to `ArchitectureA` for a recorded trial.** It pins the detector
  vocabulary and skips grounding, which hands A a list of what is on the table that B never
  receives. Debugging aid only. `run_evaluation.py` deliberately passes nothing.

## What exists right now (13 Oct 2026)

Week 10 of 13. The arm became available about 22 Sep, after the lab move. The first
session at the real UR5e was 23 Sep, and Architecture A ran end to end on the real cell on
1 Oct 2026 ("pick up the orange").

**Working on real hardware:** the speech front end; `URController` motion (`go_home.py`
reached `home_q` within 0.012 degrees, and `tests/hardware -k "not gripper"` passed 7 of 7
twice); the RG2 gripper protocol (`scripts/check_gripper.py`); `check_orientation.py`; the
fixed camera on the cell and OWLv2 on real frames; Architecture A's grounding, perception,
planner and `pick`, end to end on 1 Oct.

**Written, not yet run on hardware:** the finger-width grasp check in `skills.py` (a close
below `min_grasp_width_mm` returns "grasp missed", opens and backs off) and
`scripts/calibrate_ball.py`, both 13 Oct; the detector model cache, 13 Oct; the evaluation
harness; Architecture B's robot side. `pytest tests/unit` collects 254 as of 13 Oct 2026.
`tests/hardware/test_orientation.py` (not yet run) checks that rotations at the down pose
take the short way.

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

**Not done:** the first ball fit of `T_base_cam` (the current file is the 1 Oct hand-adjusted
session fix, `validated: false`), the drawer `pull_axis`, the e-stop recovery test, and all
demonstration capture. There are zero episodes. Demonstrations will be driven with a 3D jog
device, not freedrive.

**No fiducial markers, anywhere.** Since 13 Oct 2026 `T_base_cam` is FITTED at the start of
each session by `scripts/calibrate_ball.py`: a matte ball held in the gripper is carried to 16
poses, the camera finds it, and the pairs go to `solve_rigid_transform`. Tag-free, so the rule
holds. The lab is shared, so the camera is taped in place and the fit is redone each session.
The fit reports its own RMS, but it CANNOT see a wrong `calibration.ball_offset_tool_m`: every
pose has the same tool orientation, so that error is the same shift at every point. The reach
test (`check_calibration.py`) checks it once, and again whenever the ball or the way it is
placed changes. Measuring the mount by hand (`write_calibration.py`) is now the fallback, with
the 1-3 cm bias that implies: 1 degree of mount orientation error is ~17 mm at a 1 m standoff.
Reasoning in `docs/adr/0007-no-marker-ball-fit-each-session.md`.

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

**ADRs** (`docs/adr/`) are numbered `NNNN-slug.md`, one decision each, and open with the
decision as the title. A revised decision is updated in place with a dated note, not
duplicated.

## Agent skills

### Issue tracker

Issues and specs are local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles, written as `Status:` lines in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

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

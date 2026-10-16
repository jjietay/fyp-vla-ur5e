# FYP: Modular vs. End-to-End Vision-Language-Action Models for Embodied Robotic Manipulation

Modular LLM pipeline against a fine-tuned VLA, on a real UR5e.

NTU EEE Final Year Project (A1005-261). The user speaks a command and a UR5e
carries it out. The project compares two ways of making that happen: a modular
LLM pipeline (open-vocab perception + scripted primitives + LLM planner) against
an end-to-end VLA model (SmolVLA), across a tiered task suite covering
pick-and-place, pouring, ambiguous instructions that need clarifying, and
dynamic tasks whose world state changes mid-execution.

**Supervisor:** A/P Cheah Chien Chern
**Period:** Aug 2026 – May 2027
**Plan:** `docs/fyp_plan.md` is authoritative. Design decisions and their
reasoning are in `docs/adr/`.

## Setup

Requires Python 3.12+, `uv`, and a real UR5e. Tests under `tests/unit/` need none of it;
tests under `tests/hardware/` move the arm and skip themselves when it is not reachable.

```bash
uv sync
uv run pytest tests/unit   # no device needed
uv run pytest tests        # adds the hardware tests, which skip without a controller
```

## Layout

Organised around the real UR5e. Every result comes from hardware.

```
src/fyp/
├── shared/           the CONTROLLED VARIABLE - both architectures use these
│   ├── helpers/      rotations, transforms, config, rate_limiter
│   ├── hardware/     ur5e_controller, camera, safety, intrinsics
│   └── speech.py     push-to-talk ASR, outside both architectures
├── architecture_a/   modular pipeline, nothing trained
│   ├── perception/   detector, filters, localiser, pixel_to_3d
│   ├── calibration/  hand_eye - A only, B never forms a 3D point
│   ├── skills.py     pick / place / pour / open_drawer
│   ├── tools.py      the schemas the LLM sees
│   ├── planner.py    LLM planning, plan validation, ask_user, query grounding
│   ├── pipeline.py   the orchestrator
│   └── trace.py      per-stage failure log
├── architecture_b/   end-to-end SmolVLA, nothing hand-authored
│   ├── demos/        recorder, hdf5_store, lerobot_export
│   └── policy_action.py  the clamp, the delta maths and the envelope check
└── evaluation/       the harness. Imports neither architecture, deliberately
    ├── suite.py      tiers, utterances, layouts, success criteria. Frozen
    └── harness.py    trial runner and results tables

```

The three-way split is deliberate: the tree mirrors the experiment. `shared/`
admits a module only if BOTH architectures use it at runtime, not merely because
it is generically useful. The test is that you could delete `architecture_b/`
entirely and `architecture_a/` would still run.

- `scripts/` : thin CLIs - `check_*` verify one stage each, `export_*` does work
- `integrations/` : the LeRobot UR5e adapter, installed into the lerobot venv, not this one
- `tests/` : split by whether a device is needed - `unit/` runs anywhere,
  `hardware/` needs a UR controller and skips itself when none answers
- `config/` : `config.yaml`, plus `calibration/` which is **tracked**, because you
  must know which calibration produced which result
- `data/` : demonstration recordings (`raw/`), rebuildable datasets (`processed/`),
  disposable intermediates (`cache/`)
- `docs/` : the plan, the NTU deliverable specs, the decision records (`adr/`)

Only `architecture_a/perception/detector.py` needs torch; everything else runs in the FYP
venv. That split is visible in the tree rather than something you remember.

`shared/helpers/` is genuinely pure: numpy and the standard library only.

## Status

Week 10 of 13, 13 Oct 2026.

Both architectures are written. Architecture A is complete end to end: grounding,
perception, skills, planning, the clarification loop and per-stage tracing.
Architecture B's robot side runs on LeRobot's own `policy_server` and
`robot_client` through the UR5e adapter in `integrations/`. The safety envelope,
the camera driver and the evaluation harness all exist.

**The arm moves, and Architecture A runs on it.** The first session on the real
UR5e was 23 Sep 2026: the arm moves from Python, the hardware tests pass, the
OnRobot RG2 gripper is driven over its URCap protocol, and the down pose, table
height and reach limit are measured. On 1 Oct 2026 Architecture A ran end to end
on the real cell, picking up an orange from "pick up the orange". Speech
transcribes on the GPU and drives Architecture A.

Still to do on the cell: the first per-session ball fit of `T_base_cam`
(`scripts/calibrate_ball.py`, 13 Oct), the drawer pull axis, and the e-stop
recovery test. There are zero recorded episodes, so Architecture B has no
checkpoint yet. Current tasks live in
`docs/fyp_plan.md` §12, Open actions.

# FYP: LLM-Based Interface for Robotics Systems using UR5e

NTU EEE Final Year Project. Comparing a modular LLM-driven pipeline (open-vocab perception + scripted primitives + LLM planner) against an end-to-end VLA model (SmolVLA), evaluated on generalisation to unseen objects with a UR5e arm.

**Supervisor:** Prof Cheah
**Period:** Aug 2026 – May 2027 (with early prep period Jun–Aug 2026)

## Setup

Requires Python 3.10+, `uv`, and URSim 5.25.2 running locally (see `docs/ursim_setup_reference.md`).

```bash
uv sync
uv run python scripts/explore_control.py
```

## Layout

Organised around the real FYP (real UR5e, Aug 2026 – May 2027), not around the
simulation. MuJoCo is a development substrate and is quarantined in one folder
that gets deleted when hardware arrives.

```
src/fyp/
├── helpers/          pure, stateless maths — rotations, transforms,
│                     pixel_to_depth, ik, config, rate_limiter
├── hardware/         the physical cell
│   ├── ur5e_controller.py    real arm (ur_rtde)
│   └── sim/                  MuJoCo substrate — DELETABLE
├── policy/modular/   Architecture A — detector, filters, localiser
└── demos/            episode capture + LeRobot export (Architecture B data)
```

- `scripts/` : thin CLIs — `check_*` verify one stage each, `record_*`/`export_*` do work
- `tests/` : unit tests
- `data/` : logs and demonstration recordings
- `docs/` : setup references, plan PDFs
- `assets/` : MuJoCo scene/robot/gripper XML (dies with the sim)

Only `policy/modular/detector.py` needs torch; everything else runs in the FYP
venv. That split is visible in the tree rather than something you remember.

## Status

Pre-FYP prep period : see `docs/fyp_pre_start_plan_revised.pdf`.
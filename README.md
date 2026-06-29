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

- `src/fyp/` : package code (controller, transforms)
- `scripts/` : exploratory one-off scripts
- `notebooks/` : Jupyter explorations
- `tests/` : unit tests
- `data/` : logs and demonstration recordings
- `docs/` : setup references, plan PDFs

## Status

Pre-FYP prep period : see `docs/fyp_pre_start_plan_revised.pdf`.
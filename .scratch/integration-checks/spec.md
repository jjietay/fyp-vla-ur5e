Status: ready-for-human

# Spec: integration checks that run without the arm

## Problem Statement

The project has no CI. The unit suite (279 tests, about 2 s) is green, but it proves that each
module works alone. Nothing proves that the modules work together. When JJ adds a new file to a
pipeline, a wrong signature, a wrong return shape or a skipped safety step shows only at the
bench, on the real arm, where lab time is expensive and a mistake can move metal.

Three more gaps make this worse:

- The import check after a move or rename is a script in `CLAUDE.md`. It runs only when someone
  remembers it.
- Of the architecture rules that the fairness argument rests on, only one has a test (the harness
  does not import either architecture). The `shared/` admission rule and the rule that A and B
  stay separate are text only.
- Nothing runs the suite automatically. A commit can go in with a red suite.

## Solution

Add four automatic checks and one gate. All of them run on a laptop with no robot, no camera, no
GPU model and no network:

1. **Import test.** The import check from `CLAUDE.md` becomes a unit test.
2. **Boundary tests.** The fairness rules become unit tests that parse imports.
3. **End-to-end smoke test plus contract test.** One spoken instruction goes through the real
   harness, the real Architecture A, the real planner, skills and workspace envelope, to a fake
   controller. A contract test makes sure that every agent satisfies `run(instruction) -> outcome`
   in the same way.
4. **Static type check (pyright).** Lowest priority, done last.
5. **Pre-commit hook.** It runs the unit suite before each commit and stops the commit if the
   suite is red.

After this, "is `pytest tests/unit` green" answers "does my new file work with the rest", not only
"does my new file work alone".

## User Stories

1. As JJ, I want a broken `fyp.*` import to fail the unit suite, so that I find a bad move or
   rename before the bench.
2. As JJ, I want the import check to cover `src/`, `scripts/` and `tests/`, so that a script I run
   at the bench does not fail on its first line.
3. As JJ, I want the import check to run without me remembering it, so that the manual step in
   `CLAUDE.md` goes away.
4. As JJ, I want a test that fails if any module in `shared/` imports `architecture_a`,
   `architecture_b` or `evaluation`, so that the `shared/` admission rule is enforced, not only
   written.
5. As JJ, I want a test that fails if Architecture A imports Architecture B, so that A cannot use
   anything that B does not have.
6. As JJ, I want a test that fails if Architecture B imports Architecture A, so that B cannot use
   anything that A does not have.
7. As JJ, I want each boundary failure to name the file, the line and the import, so that I can
   fix it in one step.
8. As JJ, I want the boundary tests to parse imports, not search raw text, so that a docstring that
   names an architecture does not cause a false failure.
9. As JJ, I want one test that sends a transcript through the real harness and the real
   Architecture A, so that I know the harness, the agent, the planner, the skills and the
   envelope connect.
10. As JJ, I want the smoke test to use fakes only for the camera, the controller, the microphone,
    the `ask` callback, the LLM and the detector, so that every other line in the chain is real
    code.
11. As JJ, I want the smoke test to fail if a new pipeline file changes a signature that another
    file calls, so that the failure shows on my laptop and not on the arm.
12. As JJ, I want the smoke test to check that every pose the fake controller received is inside
    the workspace envelope, so that a new file cannot skip the envelope without a red test.
13. As JJ, I want the smoke test to check that the harness wrote a trial record with the outcome
    fields filled in, so that I know the harness reads the agent's outcome correctly.
14. As JJ, I want a second smoke case where the planner asks a clarifying question, so that the
    `ask` path and the `asked_question` column are covered.
15. As JJ, I want a smoke case where the planner proposes a pose outside the envelope, so that I
    know the rejection reaches the outcome as a failed stage and the fake controller received no
    motion for it.
16. As JJ, I want the smoke test to write nothing under `data/` or `logs/`, so that a test run
    cannot pollute or delete real trial data, which has no backup.
17. As JJ, I want the smoke test to run without `ANTHROPIC_API_KEY`, so that it costs nothing and
    works offline.
18. As JJ, I want the smoke test to run in the `uv` environment without PyTorch, so that the same
    single command runs the whole unit suite.
19. As JJ, I want a contract test that every agent returns an object with `ok`, `failed_stage`,
    `asked` and `run_id`, so that the harness stays architecture-agnostic.
20. As JJ, I want the contract test to be parametrised over agents, so that Architecture B joins
    it with one line when its `run()` wrapper exists.
21. As JJ, I want Architecture B's absence from the contract test to be visible (a skip with a
    reason, not a silent omission), so that I do not forget to add it.
22. As JJ, I want `pytest tests/unit` to stay fast (a few seconds), so that I run it often.
23. As JJ, I want a pre-commit hook that runs `pytest tests/unit`, so that a red suite cannot be
    committed by accident.
24. As JJ, I want the hook to be in the repo and enabled with one command, so that I can set it up
    again on the lab machine.
25. As JJ, I want the hook never to run `tests/hardware`, so that a commit can never move the arm.
26. As JJ, I want to skip the hook in an emergency with the standard git flag, so that a broken
    hook cannot block me at the bench.
27. As JJ, I want pyright to find a wrong argument name or a wrong return type between modules
    without running the code, so that I find "two files disagree" bugs earlier.
28. As JJ, I want pyright to start from a clean baseline before it joins the hook, so that the
    hook is not red on the first day.
29. As JJ, I want pyright not to fail on modules that need PyTorch or the lerobot venv, so that the
    `uv` environment can check everything else.
30. As the FYP examiner, I want the fairness rules to be enforced by tests, so that the claim that
    A and B got the same shared code is checkable, not only stated.
31. As an AFK agent, I want the boundary and smoke tests to fail with a clear message, so that I
    can see a rule I broke without reading `CLAUDE.md` first.

## Implementation Decisions

- **No production code changes for the tests.** The smoke test uses seams that already exist:
  `Harness(agent, architecture, ears, cfg, out_dir)`, `ArchitectureA(camera, controller, cfg=...,
  ask=...)`, and `run(instruction) -> outcome`. If a test seems to need a production change, stop
  and ask JJ first.
- **The highest seam is the harness.** The smoke test calls `Harness.run_tier` with
  `interactive=False`. JJ chose this over starting at `ArchitectureA.run`, because it also proves
  that the harness reads the outcome correctly.
- **Fakes only at the edges:**
  - `ears`: a fake with `hear(prompt)` that gives a fixed transcript.
  - Camera: a fake with `capture(...)` that gives an RGB frame and a depth frame that agree with
    the faked detections.
  - Controller: a fake that records every command and its pose. It never connects to anything.
  - `ask`: a fake that gives a fixed answer and records the question.
  - LLM: replace the Anthropic client at its constructor, because `run()` builds its `Planner`
    inside itself and has no client argument. The fake gives scripted responses: first the
    grounding tool call, then the planner's tool calls, then "done". Prior art: `_FakeClient` in
    the Architecture A unit tests.
  - Detector: replace `detect` in the pipeline module. OWLv2 needs PyTorch, which the `uv` venv
    does not have.
- **Calibration in the smoke test.** `ArchitectureA` refuses to start without `T_base_cam`. The
  test writes a known transform to a temporary calibration directory and points the config at
  it. It never reads the real calibration file.
- **Config.** The test takes the real config and overrides only the paths (calibration, trace
  log, evaluation output) to temporary directories. The envelope, skills and planner settings stay
  real, so the test checks the real limits.
- **Repo root.** New tests find the repo root with the helper in `shared/helpers/config.py`, never
  with `parents[n]`. The existing harness-import test uses `parents[1]`; move it to the helper in
  the same change.
- **Boundary rules, as one table in one test module:**
  - `shared` must not import `architecture_a`, `architecture_b` or `evaluation`.
  - `architecture_a` must not import `architecture_b`.
  - `architecture_b` must not import `architecture_a`.
  - `evaluation` must not import `architecture_a` or `architecture_b` (this is the rule that has
    a test now; move it into the same table so that all rules have one home).

  All rules hold today (checked 16 Oct 2026), so the tests start green.
- **Import test.** Same logic as the script in `CLAUDE.md`. After the test exists, the "Verify
  before claiming done" section in `CLAUDE.md` points to the test and the script is removed, so
  the fact has one home.
- **Contract test.** Parametrised over agent factories. A's factory builds A with the same fakes
  as the smoke test. B's entry skips with the reason "no Architecture B agent yet" until B has a
  `run()` wrapper.
- **Hook.** A versioned hooks directory in the repo, enabled once with `git config
  core.hooksPath`. The hook runs `uv run pytest tests/unit -q`. It never runs `tests/hardware`
  and never passes `--integration`. The one-time enable command goes into an existing doc
  (`README.md` or `docs/commands.md`, whichever already holds setup commands), not into a new doc.
- **Pyright, last.** Add it as a dev dependency. Check `src/` only. Exclude or allow unresolved
  imports for modules that need PyTorch or the lerobot venv. Keep the existing
  `reportAttributeAccessIssue = false`. Fix errors to a clean baseline first; add pyright to the
  hook only when the baseline is clean. If the baseline needs more than small fixes, stop and
  report to JJ instead of changing many files.
- **Suggested order:** import test → boundary tests → smoke and contract tests → hook → pyright.
  Each step leaves the suite green.

## Testing Decisions

- **A good test checks external behaviour only.** The smoke test checks what leaves the system
  (commands at the fake controller, the trial record, the outcome), never internal state or the
  order of private calls. A refactor inside the pipeline that keeps the behaviour must not break
  it.
- **Assertions for the smoke test:**
  - The harness gives one record, and its fields agree with the outcome (`success`,
    `failed_stage`, `asked_question`, `run_id`, `transcript`).
  - The fake controller received at least one motion, and every received pose is inside the
    workspace envelope.
  - The out-of-envelope case gives a failed outcome with a failed stage, and the fake controller
    received no motion to that pose.
  - Nothing was written outside the temporary directories.
- **Modules covered:** `evaluation/harness`, `architecture_a/pipeline`, `architecture_a/planner`,
  `architecture_a/skills`, `shared/hardware/safety`, through the one seam. The boundary and import
  tests cover the whole `src/`, `scripts/` and `tests/` tree.
- **Prior art:**
  - `test_harness_does_not_import_either_architecture`: the AST import scan.
  - `_FakeClient` and `_FakeResponse`: scripted LLM responses.
  - `test_layouts_are_identical_across_architectures`: a test that states a fairness rule.
  - `tests/conftest.py`: the unit/hardware split and the `integration` marker. All new tests go in
    `tests/unit/` and need no marker.
- **Speed budget:** the whole unit suite must stay under about 5 s.

## Out of Scope

- GitHub Actions or any cloud CI. The dependencies include about 1.5 GB of CUDA libraries,
  `pyrealsense2` and `ur-rtde`, and Architecture A needs the lerobot venv for PyTorch. A local
  hook gives most of the value at no cost.
- Tests that load OWLv2, faster-whisper or a real LLM.
- Any test that moves the arm. `tests/hardware/` and the bench checks in `docs/commands.md`
  already cover the real cell.
- An Architecture B agent or its `run()` wrapper. The contract test only makes room for it.
- Any change to `evaluation/suite.py`. It is frozen.
- Linting and formatting (ruff). This was not discussed.

## Further Notes

- Run every step with `uv run pytest tests` at the end and expect green, with the hardware tests
  skipped.
- After the import test replaces the `CLAUDE.md` script, run it once before the script is
  removed, and confirm that both give the same result.
- `CLAUDE.md` asks for approval before any multi-file change. This spec is that approval for the
  files that the steps above name. Anything outside them needs a new question to JJ.

## Comments

**16 Oct 2026.** Steps 1 to 4 are done and on `main` (`e50dd3a` to `67d42a7`). Additions not
in the spec: the hook also runs `tests/unit` in the lerobot venv, and `FakeController` moved
to `tests/unit/fakes.py`. The old harness-import test used `parents[1]` and checked nothing.

Step 5 (pyright) is stopped by its own stop rule: 56 errors on `src/`, of which 37 are h5py
typing in `architecture_b/demos/`, 8 are imports that only the lerobot venv has, and 8 are
spread over 6 production files (`camera.py:168` is probably a real bug when
`depth_frames < 1`). It needs JJ to decide how to configure it.

Also open: no test sends a request through the real Anthropic SDK, and the uv venv has
anthropic 0.121 while the lerobot venv has 1.11.

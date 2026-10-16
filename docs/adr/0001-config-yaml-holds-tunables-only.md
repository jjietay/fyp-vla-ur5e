# config.yaml holds every tunable value, and six things stay in code on purpose

`config/config.yaml` is the single source of truth for every **tunable** value, so that Architecture A and Architecture B cannot be constrained differently. A number living in a module is invisible to whoever is reviewing the workspace bounds, and invisible to every test.

Six things stay in code on purpose. Each would be made worse by living in an editable file.

- **`STATE_NAMES`, `ACTION_NAMES`, `REQUIRED_DATASETS`** in `lerobot_export.py`. The format contract with a trained checkpoint. Editing one line in config would silently break every checkpoint trained before the edit, with no error, and the symptom would be a policy sending roll to z.
- **`TASKS` and `CHANGELOG`** in `evaluation/suite.py`. The experimental design, frozen before either pipeline was tuned. That freeze is what the evaluation chapter claims, and a file anyone can edit weakens the claim.
- **`SYSTEM_PROMPT` and `GROUNDING_PROMPT`** in `planner.py`. Multi-paragraph text that YAML handles badly, and "behaviour changes by editing prompts" is a stated property of Architecture A. They are the architecture rather than settings for it.
- **`TOOLS`, `TOOL_NAMES`, `MOTION_TOOLS`** in `tools.py`. Derived from the skill signatures with a test enforcing the match. In config they could disagree with the code.
- **`GRIPPER_CLOSED` and `GRIPPER_OPEN`** in `skills.py`. An API contract, not a tunable, so these never change with the cell.
- **The RG2 protocol numbers** in `ur5e_controller.py`, agreed 23 Sep 2026. The word formula (the 9.2 offset, the 110 mm clamp, the 3 to 40 N force range), the width formula and the sync cycle waits. They are the gripper's protocol copied from `docs/rg2.script`, not tuning, and a changed number sends a wrong command or reads a wrong width. The tunables (open and close width, force, tolerance, resends) are in `robot.gripper`.

`PROJECT_ROOT`, `DEFAULT_CONFIG_PATH` and `_CACHE` in `helpers/config.py` are the machinery that finds and reads the config, so they cannot be read from it.

The file itself stays lean: values, with at most one line of comment each. Anything that needs a paragraph belongs in an ADR or in `docs/commands.md`.

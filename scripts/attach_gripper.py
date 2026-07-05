"""One-time: attach the Robotiq 2F-85 to the UR5e at the wrist site using mjSpec,
then save the composed model as a single scene_gripper.xml for normal use.

Run from repo root:
    uv run python scripts/attach_gripper.py

Produces: assets/mujoco/ur5e/scene_gripper.xml
"""

from pathlib import Path
import mujoco

REPO = Path.cwd()
ARM_SCENE = "assets/mujoco/ur5e/scene.xml"
GRIPPER   = "assets/mujoco/robotiq_2f85/2f85.xml"
OUT       = "assets/mujoco/ur5e/scene_gripper.xml"

# Absolute mesh dirs so the composed file finds meshes regardless of its location.
ARM_MESHDIR  = str((REPO / "assets/mujoco/ur5e/assets").resolve())
GRIP_MESHDIR = str((REPO / "assets/mujoco/robotiq_2f85/assets").resolve())


def main() -> None:
    arm = mujoco.MjSpec.from_file(ARM_SCENE)
    grip = mujoco.MjSpec.from_file(GRIPPER)

    # Force absolute mesh directories before composing.
    arm.meshdir = ARM_MESHDIR
    grip.meshdir = GRIP_MESHDIR

    site = arm.site("attachment_site")
    site.attach_body(grip.body("base_mount"), "2f85_", "")

    model = arm.compile()
    print(f"Composed OK: nu={model.nu}  njnt={model.njnt}  nq={model.nq}")

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}")

    xml = arm.to_xml()
    Path(OUT).write_text(xml)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
# FYP A1005-261

A spoken instruction makes a UR5e act, by two architectures on the same cell. The
project is the comparison of the two, so both sides use one vocabulary.

## Experiment

**Tier**:
One group of tasks in the frozen suite: Pick-place, Pour, Ambiguity, Dynamic or Generalisation.
_Avoid_: level, difficulty

**Layout**:
One reproducible starting arrangement of objects on the table for a tier.
_Avoid_: scene setup, configuration

**Utterance**:
One phrasing in the frozen suite that the operator is told to say. It is a training utterance or a held-out utterance.
_Avoid_: command, prompt

**Held-out utterance**:
An utterance never used to record demonstrations or shown to Architecture A as an example, kept only to test phrasing drift.
_Avoid_: test phrase, unseen command

**Instruction**:
The transcript an architecture receives: the utterance after speech-to-text, with its errors.
_Avoid_: command, query

**Trial**:
One scored attempt by one architecture at one instruction on one layout.
_Avoid_: run, episode, attempt

**Failure stage**:
The stage a failed trial is blamed on: capture, grounding, detection, depth, transform, planning, clarify or execution. This list is the failure taxonomy of the report.
_Avoid_: error type, failure mode

**Clarification**:
A question the architecture asks the user before it acts, and the user's answer.
_Avoid_: follow-up, disambiguation

**Shared module**:
A module that both architectures use at runtime: the controlled variable of the experiment.
_Avoid_: common code, utility

## Cell

**Base frame**:
The UR5e base coordinate frame. Every position in the project is in it.
_Avoid_: world frame, table frame, robot frame

**Envelope**:
The base-frame region the arm may go to: a ring sector around the base axis (radius and bearing) and a height range. Every motion of either architecture is checked against it.
_Avoid_: workspace, safety box, limits

**Out of reach**:
A located object outside the envelope's sector or below its floor. The planner gets its name and the reason, never its position. The height ceiling is not applied, because a held object reads above it.
_Avoid_: out of workspace, unreachable

## Architecture A

**Query**:
One short noun phrase the detector looks for, such as "orange" or "box".
_Avoid_: noun phrase, detector vocabulary, prompt

**Grounding**:
Turning an instruction into queries. Only Architecture A does it, so only A can fail there.
_Avoid_: query extraction, parsing

**Located object**:
A detected object with a base-frame point.
_Avoid_: detection, target

**Raw point**:
A located object's point as the camera and the calibration give it, before any runtime correction. Calibration steps measure this point.
_Avoid_: unclamped point

**Runtime point**:
The point Architecture A acts on: the raw point after the floor lift, kept only if it is not out of reach.
_Avoid_: final point, corrected point

**Floor lift**:
Moving a located object whose centre reads a little below the table up to the table height.
_Avoid_: clamp, z correction

**Skill**:
A scripted motion Architecture A can perform: pick, place, pour or open_drawer.
_Avoid_: primitive, action

**Tool**:
Something the planner can call: each skill, plus ask_user, which is a tool but not a skill.
_Avoid_: function, command

**Calibration**:
The current camera-to-base transform, T_base_cam. Only Architecture A forms a 3D point, so only A has one.
_Avoid_: extrinsics, hand-eye result

**Calibration step**:
One of the operations that writes a new calibration: ball fit, levelling, place-and-look, or a measured mount as the fallback.
_Avoid_: calibration pass

**Reach test**:
The check where the arm goes to points the camera located and the operator measures the miss.
_Avoid_: validation run, accuracy check

**Validated calibration**:
A calibration that passed the reach test after it was last written. Any calibration step makes it unvalidated again.
_Avoid_: verified, checked

## Architecture B

**Demonstration**:
One teleoperated recording of a task, used to train Architecture B.
_Avoid_: episode (except in LeRobot and HDF5 code, whose formats use it), demo

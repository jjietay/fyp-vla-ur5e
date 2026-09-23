---
tags: [moc]
---

# Start Here

Navigation hub for the FYP plan. The authoritative document is `docs/fyp_plan.md` in the repo, and this vault is the working surface for tracking it.

Project A1005-261, *Modular vs. End-to-End Vision-Language-Action Models for Embodied Robotic Manipulation*. Supervisor A/P Cheah Chien Chern, Robotics I lab.

Title updated by Prof Cheah on 31 Aug 2026, from *Development of a Large Language Model-Based Interface for Robotic Systems*. The new one names the comparison directly, which is what the project argues.

## Summary

The user speaks a command, the UR5e does it, and the project compares two ways of making that happen: a modular LLM pipeline with scripted primitives, against a fine-tuned vision-language-action model.

## Where to look

#### 1. Planning

[[Weekly Focus]] for the week by week route to Nov 2026.
[[FYP Submissions]] for what NTU wants, when, and where the marks are.
[[To Do (Non-Technical)]] for the things blocking everything else.
[[Flowchart of Architecture and Dependencies]] for the dependency and runtime diagrams.

#### 2. Doing the Work

[[Workstreams]] for the build order and what state each part is in.
[[Tasks]] for what the robot has to accomplish.
[[Decisions to Make]] for the choices made and why.

#### 3. Thinking
[[Hypotheses for Comparison]] for what the project claims.
[[Evaluation of Architecture]] for how those claims get tested.
[[Different Risk in Project]] for what goes wrong and what to do about it.
[[Research Papers for Reading]] for the papers.

## Status, 23 Sep 2026

Week 7 of 13. **The arm moves.** The first session at the real UR5e was on 23 Sep, after the lab move.

Proved on hardware so far:

- speech transcribes on the GPU and drives Architecture A
- the arm moves from Python. `go_home.py` reached `home_q` to within 0.012 degrees, and the hardware tests passed 7 of 7, twice
- the RG2 gripper opens and closes from Python, by sending the URCap's own protocol as URScript
- the down pose, the table height and the reach limit are measured and in `config.yaml`

The rotation format is settled. Euler roll flipped on the real arm, so the state for Architecture B now uses two columns of the rotation matrix, see [[Decisions to Make#Action space]].

Not yet done: the camera on the cell, `T_base_cam`, the detector on real frames, the workspace x and y bounds, and the e stop recovery test. There are still **zero recorded episodes**, so nothing can be trained yet.

The gripper code was rewritten on 23 Sep to check every command on the controller and send it again once if the fingers did not move. That version has not run on the arm yet.

## How this vault is organised

Twelve files, flat, no folders. Each one is a subject rather than a note, so the D-notes live inside [[FYP Submissions]], the tiers inside [[Tasks]], and so on. It was 42 files across five folders until 31 Aug, and most of them were shorter than their own frontmatter.

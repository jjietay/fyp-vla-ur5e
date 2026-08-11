---
tags: [overview]
---

# Workstreams Overview

Build order, arranged by dependency rather than by date. [[Schedule]] places these in time.

The split that matters is what needs the lab and what does not:

* [[W1 Repo Reset]], [[W2 Architecture A Software]] and [[W2b Speech Front End]] need no hardware, so they should be finished before the arm is available
* [[W3 Camera and Calibration]] onward are all hardware gated

Critical path to a result: lab access, then [[W3 Camera and Calibration]], then [[W5 Demonstration Capture]], then [[W6 Architecture B Training]], then [[W7 Evaluation and Write Up]]. Everything in Architecture A sits off that path, which is exactly why it goes first.

| Workstream | Needs lab | Blocks |
|---|---|---|
| [[W1 Repo Reset]] | no | everything |
| [[W2 Architecture A Software]] | no | W4 |
| [[W2b Speech Front End]] | no | W5 recording |
| [[W3 Camera and Calibration]] | yes | W4, W5 |
| [[W4 Architecture A End to End]] | yes | W7 |
| [[W5 Demonstration Capture]] | yes | W6 |
| [[W6 Architecture B Training]] | partly | W7 |
| [[W7 Evaluation and Write Up]] | yes | D4 |

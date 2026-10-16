# A fixed RGB-D plus a wrist RGB camera, the fixed one square on to the table and 0.7 to 0.8 m above it

Decided by 13 Sep 2026, when both cameras were in hand. The lab unit is a **RealSense D435i**, confirmed 31 Aug. Its SDK is `pyrealsense2`, and `shared/hardware/camera.py` drives it.

## Does the wrist camera need depth

No. SmolVLA consumes RGB images, a state vector and a language string, and has no depth input at all. Architecture A needs depth only from the **fixed** camera, because that is where detection happens and where pixels are back projected into 3D.

| Camera | Type | Architecture A | Architecture B |
|---|---|---|---|
| fixed, side view | RGB-D, the lab D435i | detection, depth to 3D, base frame | first RGB view |
| wrist mounted | RGB is sufficient | optional close range re-detect | second RGB view |

**Hard rule: nothing may depend on wrist depth.** Design as if it is RGB. If depth arrives it is a branch that can be deleted without touching anything else. If the lab pays, specify a short minimum range model, because a D435 class sensor cannot focus closer than about 0.3 m and is useless at grasp distance.

Two cameras is evidence rather than preference. Reference SmolVLA fine tuning setups use two views, and overhead plus wrist produces noticeably smoother trajectories than wrist only. A single camera chosen against the reference configuration would compromise the evaluation.

## Where the fixed camera stands

Square on to the flat side of the table, arm at the far end, workspace between the two.

- **azimuth: square on**, optical axis along the base x axis rather than diagonally across a corner
- **lateral: centred**, camera at `y = 0` in base frame, position `(x_c, 0, z_c)` with `x_c` beyond the 0.75 m reach limit measured on 23 Sep, looking back along minus x
- **elevation: 0.7 to 0.8 m above the table**, about 35 to 39 degrees at a 1 m standoff

Two D435i limits that constrain this: min-Z is about **0.3 m**, so nothing may sit closer than that, and it needs a **USB 3** cable, because a USB 2 cable enumerates and then silently caps resolution.

## Why square on and centred

Range error lands along the camera ray. Pointing that ray down the base x axis keeps almost all of it in one axis, so the `T_base_cam` bias and the surface-to-centroid offset collapse into a single constant the reach test can measure. From a diagonal the same error smears across x and y and needs two.

Centring is not cosmetic: a lateral offset tilts the optical axis and range error starts splitting into x and y, the same cross talk as an azimuth mounting error but caused by position. With both, the error decomposes cleanly. y carries no range error at all, since it comes purely from pixel column. x and z carry all of it, roughly 80 percent x and 20 percent z at 39 degrees, and z is absorbed by `pick` descending from its approach height.

Occlusion also drops. Against the frozen layouts in `evaluation/suite.py`, square on puts 15 of 100 at risk against 25 of 100 at a diagonal, because the layouts spread objects over 40 cm in y and only 25 cm in x.

One consequence to expect rather than fix: the arm column becomes the backdrop behind the workspace, so OWLv2 sees objects against metal. If detection scores look soft on the real cell, rule this out before blaming the threshold.

## Why 0.7 to 0.8 m

A near object hides a region behind it of length `s = d * H / (h - H)`, for standoff `d`, object height `H` and camera height `h`. It diverges as `h` approaches `H`, which is why a camera near table level is unusable.

At `d` of 1 m, counting a hit when one object falls in another's shadow within 5 cm laterally:

| camera height | elevation | 5 cm objects | 20 cm objects |
|---|---|---|---|
| 0.30 m | 17 deg | 12/100 | 15/100 |
| 0.60 m | 31 deg | 0/100 | 15/100 |
| 0.80 m | 39 deg | 0/100 | 15/100 |
| 1.50 m | 56 deg | 0/100 | 6/100 |

Short object occlusion disappears past a knee at 0.6 m, and 0.7 to 0.8 m is that knee plus margin. Measure `d` on the cell and re-run the formula before settling the tripod. Going higher is not free: more elevation foreshortens the vertical faces OWLv2 needs to find a drawer handle.

**What elevation cannot fix** is tall objects. A 20 cm bottle still occludes at 1.0 m. Tier 1 and Tier 2 layouts must place tall objects so nothing sits behind them, and Tier 2 (Ambiguity) is the exposed one: a hidden second candidate turns a clarification trial into Architecture A confidently picking the visible object, which scores a mount choice as an architecture finding.

## Considered options

Whichever is used, **tape the tripod position**, because the lab is shared and the camera comes down between sessions.

- **Far end, looking back at the arm.** The choice. Nothing stands between the camera and the objects, and range error collapses into one axis.
- **Off to one side, looking across.** The acceptable fallback. Occlusion is moderate, but range error splits between x and y, so the bias is two constants instead of one.
- **Behind the arm, looking forward over it.** Avoid. The arm stands between the camera and everything it is trying to see, and **it moves**. The detector can lose the target mid-task, and `depth_at` can sample the arm's depth rather than the object's, which puts a grasp point on the gripper with a perfectly valid number. If it is the only option: mount as high as the tripod allows, re-perceive only from the home pose, and state it in the report.

## Consequences

Square on foreshortens the reach axis, and foreshortening is the weakest cue in an RGB image. That cost falls on Architecture B alone, since B never forms a 3D point and cannot benefit from the geometry square on buys.

Accepted for two reasons. The wrist camera reads approach distance directly, and the reference configuration is overhead plus wrist, where the descent runs straight at the overhead camera and is foreshortened the same way. Occlusion, by contrast, costs both architectures equally, so a diagonal is not a neutral compromise: it pays a penalty on both to buy back what the wrist camera already covers.

State the placement and this residual in the D2 Interim Report. Documenting the direction of a known bias is what makes the comparison defensible.

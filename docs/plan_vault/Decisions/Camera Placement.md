---
status: decided
---

# Camera Placement

**Square on to the flat side of the table, 0.7 to 0.8 m above the table surface.**

Arm at the far end, workspace between the arm and the camera. This fixes where the tripod stands, not what the camera is, which is [[Camera Setup]].

## The three numbers

**Azimuth: square on.** Optical axis along the robot base x axis, not diagonally across a corner.

**Lateral: centred.** Camera at `y = 0` in base frame, so the arm sits directly in front of it and the camera position is `(x_c, 0, z_c)` with `x_c` beyond the 0.55 m reach limit, looking back along minus x.

**Elevation: 0.7 to 0.8 m above the table**, which is about 35 to 39 degrees at a 1 m standoff.

## Why centred

Not cosmetic. The single constant correction below is only valid when the optical axis is parallel to base x. A lateral offset tilts that axis and range error starts splitting `cos` into x and `sin` into y, which is the same cross talk as an azimuth mounting error but caused by position rather than rotation. Centring drives that term to zero.

With centring and elevation together the error decomposes cleanly:

* y carries no range error at all, since the ray has no y component and y comes purely from pixel column
* x and z carry all of it, split by elevation angle, roughly 80 percent x and 20 percent z at 39 degrees
* z is absorbed by `pick` descending from its approach height, so what is left to correct is effectively one constant in x

Centring also makes the residual off axis leak symmetric about zero, maxing at the `y = +/- 0.20` edges at about 4.6 mm. An offset camera would make one side worse than the other for no gain.

One consequence to be aware of rather than fix: the arm column becomes the backdrop directly behind the workspace, so OWLv2 sees objects against metal rather than table. If detection scores look soft on the real cell, rule this out before blaming the detector threshold.

## Why square on

Range error lands along the camera ray. Pointing that ray down the base x axis keeps almost all of it in one axis, so the hand measured `T_base_cam` bias and the surface to centroid offset collapse into a single constant that the W3 reach test can measure. From a diagonal the same error smears across x and y and needs two.

Lateral position comes from pixel column rather than depth, so y stays accurate. Two second order leaks, both millimetres at these distances:

* depth error leaks into y in proportion to how far off axis the object sits, since `y = (u - cx) z / fx` scales with depth
* an azimuth mounting error of angle phi splits range error as `cos phi` into x and `sin phi` into y, so square the tripod to the table edge rather than eyeballing it

Occlusion also drops. Checking the frozen layouts from `build_layouts` in `evaluation/suite.py`, square on puts 15 of 100 at risk against 25 of 100 at a diagonal, because `build_layouts` spreads objects over 40 cm in y and only 25 cm in x. Viewing along x puts the wide axis across the image where separation reads most easily.

## Why 0.7 to 0.8 m

A near object hides a region behind it of length

```
s = d * H / (h - H)
```

for camera to workspace distance `d`, object height `H` and camera height above the table `h`. It diverges as `h` approaches `H`, which is why a camera near table level is unusable.

At `d` of 1 m, counting a hit when one object falls in another's shadow and within 5 cm laterally:

| camera height | elevation | 5 cm objects | 20 cm objects |
|---|---|---|---|
| 0.30 m | 17 deg | 12/100 | 15/100 |
| 0.40 m | 22 deg | 5/100 | 15/100 |
| 0.60 m | 31 deg | 0/100 | 15/100 |
| 0.80 m | 39 deg | 0/100 | 15/100 |
| 1.50 m | 56 deg | 0/100 | 6/100 |

Short object occlusion disappears entirely past a knee at 0.6 m. 0.7 to 0.8 m is that knee plus margin, since `s` scales linearly with `d` and the real standoff may exceed 1 m. Measure `d` on the cell and re run the formula before settling the tripod.

Going higher is not free. More elevation pushes range error out of x and into z, which `pick` absorbs through its approach height, but it also foreshortens the vertical faces that OWLv2 needs to find a drawer handle. At 39 degrees roughly 80 percent of range error is still in x and the side faces are still readable, which is the reason to stop there rather than at 56 degrees.

## What elevation cannot fix

Tall objects. A 20 cm bottle still occludes at 1.0 m and only partly clears at 1.5 m, which is an impractical tripod. The shadow of a tall object is long at any height that can actually be mounted.

Tier 1 and tier 2 setups therefore have to place tall objects so that nothing sits behind them. This is a real constraint on the layout sheet, not a nicety. Tier 2 is the exposed one, because a hidden second candidate turns a clarification trial into Architecture A confidently picking the visible object, which scores a mount choice as an architecture finding.

## The residual, and why it is accepted

Square on foreshortens the reach axis, and foreshortening is the weakest cue in an RGB image. That is a cost to Architecture B alone, since Architecture B never forms a 3D point and cannot benefit from the geometry that square on buys.

It is accepted for two reasons. The wrist camera reads approach distance directly. And the reference configuration is overhead plus wrist, where the descent runs straight at the overhead camera and is foreshortened in the same way, so square on plus wrist sits no further from the reference than a diagonal plus wrist.

Occlusion, by contrast, costs both architectures equally, so a diagonal is not a neutral compromise. It pays a penalty on both to buy back something the wrist camera already covers.

State the placement and this residual in D2. Documenting the direction of a known bias is what makes the comparison defensible.

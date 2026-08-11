---
status: decided
---

# Camera Setup

**Fixed RGB-D plus a wrist RGB camera.**

## The question that was asked

Does the wrist camera need depth?

**No.** SmolVLA consumes multi view RGB images, a robot state vector and a language string. It has no depth input at all. Adding depth to the wrist contributes nothing to Architecture B.

Architecture A does need depth, but only from the **fixed** camera, since that is where detection happens and where pixels get back projected into 3D and transformed into the base frame. The existing lab RGB-D covers that.

| Camera | Type | Architecture A | Architecture B |
|---|---|---|---|
| fixed overhead or side | RGB-D, existing lab unit | detection, depth to 3D, base frame | first RGB view |
| wrist mounted | RGB is sufficient | optional close range re detect | second RGB view |

## When to claim RGB-D anyway

If the lab pays rather than you, an RGB-D wrist camera buys optionality for close range grasp verification. Specify a **short minimum range model**, since a standard D435 class sensor cannot focus closer than about 0.3 m and is useless at grasp distance.

**Hard rule either way: nothing may depend on wrist depth.** Design as if it is RGB. If depth arrives it is a branch that can be deleted without touching anything else.

## Justifying two cameras

Not preference, evidence. Reference SmolVLA fine tuning setups use two views, and policies trained with overhead plus wrist produce noticeably smoother trajectories than wrist only. If Architecture B underperforms because of a single camera chosen against the reference configuration, the comparison in [[Evaluation Protocol]] is compromised.

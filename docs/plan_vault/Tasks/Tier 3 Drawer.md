---
tier: 3
status: not started
---

# Tier 3 Drawer

Spoken: *"Put the snack in the drawer."* The drawer starts closed, the robot opens it, and **the human closes it again mid task**. The robot has to notice and re open before placing.

## Why it matters

This is the tier where the prediction flips, which is what makes it worth doing.

Architecture A must detect the state change and re plan. Plan then execute is bad at exactly this, which is why re detection before each grasp is required work in [[W4 Architecture A End to End]].

Architecture B is a closed loop visuomotor policy running at 30 Hz. If the demonstrations included re closures, reactive recovery is the thing it is naturally good at. Expect B to win here.

## Being even handed

A comparison where one architecture wins everything is a comparison nobody believes. This tier is where Architecture B should look better, and reporting that honestly is what makes the Tier 2 result credible.

## Recording note

The demonstrations must **include** the perturbation. Record episodes where the drawer gets closed part way through and the operator re opens it. If B only ever saw clean episodes it cannot recover, and the result measures the dataset instead of the architecture.

Hardware: a small drawer unit with a handle the gripper can actually grasp, see [[Shopping List]].

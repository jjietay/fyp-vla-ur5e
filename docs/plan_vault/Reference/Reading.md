---
tags: [reference]
---

# Reading

Three papers, two of them still outstanding, and both outstanding ones are needed *before* the work they inform rather than after.

## RTC, Black et al. 2025

Asynchronous inference and action chunking. Directly about the problem the inference loop has to solve.

**Read before designing the async loop in [[W6 Architecture B Training]].** Reading it afterwards means rewriting.

## Open X-Embodiment, Padalkar et al. 2023

Action and observation formats across a large collection of robot datasets, including UR5.

**Read before settling [[Action Space]].** It is the closest thing to a survey of what conventions other people converged on, which is exactly the question being decided.

## SmolVLA

Already read and summarised. Key facts carried into the plan:

* multi view RGB, robot state and a language string as input, with **no depth channel**
* action chunk horizon of 50, executing in joint space
* asynchronous inference reported at roughly 30 percent off completion time
* around 50 episodes as a fine tuning starting point, with workspace density mattering more than raw count

Citing all three properly is worth marks under references, see [[Assessment Mapping]].

---
tags: [reference]
---

# Reading

Three papers, two of them still outstanding, and both outstanding ones are needed *before* the work they inform rather than after.

## RTC, Black et al. 2025

Asynchronous inference and action chunking. Directly about the problem the inference loop has to solve.

**Read before designing the async loop in [[Workstreams#W6 Architecture B training\|W6 Architecture B Training]].** Reading it afterwards means rewriting.

## Open X-Embodiment, Padalkar et al. 2023

Action and observation formats across a large collection of robot datasets, including UR5.

Relevant to [[Decisions to Make#Action space\|Action Space]], which was settled on 31 Aug and 23 Sep. It is the closest thing to a survey of what conventions other people converged on, so it is the reference for justifying that choice in the report.

## SmolVLA

Already read and summarised. Key facts carried into the plan:

* multi view RGB, robot state and a language string as input, with **no depth channel**
* action chunk horizon of 50. The reference setups act in joint space, while this project uses TCP pose deltas to match Architecture A
* asynchronous inference reported at roughly 30 percent off completion time
* around 50 episodes as a fine tuning starting point, with workspace density mattering more than raw count

Citing all three properly is worth marks under references, see [[FYP Submissions#Where the marks actually are\|Assessment Mapping]].

"""FYP: an LLM-based interface for a UR5e.

The tree is the experiment. Three packages, one per role:

    shared/           the CONTROLLED VARIABLE. Anything both architectures
                      touch at runtime: the arm, the cameras, speech, frame
                      maths. If A and B differ here, the comparison measures
                      plumbing instead of architectures.
    architecture_a/   modular pipeline. Open-vocab detection, depth to 3D,
                      hand-eye transform, LLM planner over scripted skills.
                      Nothing in it is trained.
    architecture_b/   end-to-end VLA. Real demonstrations, LeRobot dataset,
                      fine-tuned SmolVLA, action chunks straight to the arm.
                      Nothing in it is hand-authored.

ADMISSION RULE for `shared/`: a module belongs there only if BOTH architectures
use it at runtime, not merely because it is generically useful. Pure maths used
by only one side stays with that side. The test is that you could delete
`architecture_b/` entirely and `architecture_a/` would still run, and vice versa.

Without that rule `shared/` slowly becomes `utils/` and stops being evidence of
anything. With it, the fairness argument in the write-up is just a directory
listing.
"""

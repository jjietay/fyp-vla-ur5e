---
status: decided
---

# Detector Query Format

**OWLv2 takes short noun phrases, never sentences. This is why Architecture A needs a language model in front of its detector.**

## The constraint

OWLv2 encodes text with CLIP's text encoder. Every query becomes a single embedding, which is then matched against the per patch embeddings of the image.

What follows from that:

* a sentence produces one embedding for the whole sentence, and no region of the image corresponds to it
* there is no mechanism inside the model for splitting a sentence into the several objects it mentions
* there is no mechanism for ignoring the verbs, articles and destinations a sentence also contains
* queries are embedded independently, so the model cannot represent a relationship like "the cube that is inside the tray"

CLIP's 77 token context is a real limit but it is not the binding one here. A spoken command fits comfortably inside 77 tokens and still fails, because the problem is what the embedding means rather than how long it is.

## Why this matters to the project

It is the architectural reason Architecture A cannot be built from a detector alone. Something has to convert

    "place that red cube into that metal tray"

into

    ["red cube", "metal tray"]

before OWLv2 can be asked anything at all. That step is the grounding call in `planner.py::extract_queries`, and it is a genuine requirement rather than a convenience.

## The comparison angle

Worth stating plainly in the report, because it cuts both ways.

Architecture A pays a real cost here. It needs a second language model call purely to bridge from human language into detector language, and that bridge is a failure point, recorded as H9 in [[Hypotheses]]. A wrong noun phrase kills the run before perception even begins.

Architecture B pays nothing, because SmolVLA takes the raw instruction string straight into the policy and never converts it into anything. The language never has to become a list of object names.

So the modular architecture is not simply more interpretable at a small cost. It requires an extra translation step that the end to end model does not, and that step is one of the places it breaks.

## Practical rules

* keep distinguishing words the user actually said, so "red cube" not "cube"
* do not invent distinguishing words, so "the cube" stays "cube"
* include the destination, since the robot has to find that too
* cap the number of queries, because more queries slow OWLv2 and add false positives

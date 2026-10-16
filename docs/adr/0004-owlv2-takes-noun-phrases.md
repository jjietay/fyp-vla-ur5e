# OWLv2 takes short noun phrases, so Architecture A needs a language model in front of its detector

Forced by how OWLv2 works rather than chosen.

OWLv2 encodes text with CLIP's text encoder, so every query becomes a single embedding matched against per patch image embeddings. What follows:

- a sentence produces one embedding for the whole sentence, and no region of the image corresponds to it
- there is no mechanism for splitting a sentence into the several objects it mentions
- there is no mechanism for ignoring the verbs, articles and destinations a sentence also contains
- queries are embedded independently, so a relationship like "the cube that is inside the tray" cannot be represented

CLIP's 77 token context is a real limit but not the binding one. A spoken command fits inside it and still fails, because the problem is what the embedding means rather than how long it is.

## Consequences

Something has to convert `"place that red cube into that metal tray"` into `["red cube", "metal tray"]` before OWLv2 can be asked anything. That is the grounding call in `planner.py::extract_queries`, and it is a genuine architectural requirement.

Worth stating plainly in the report, because it cuts both ways. Architecture A needs a second language model call purely to bridge from human language into detector language, and that bridge is a failure point, recorded as H9. A wrong noun phrase kills the run before perception begins. Architecture B pays nothing, because SmolVLA takes the raw string into the policy and never converts it.

So the modular architecture is not simply more interpretable at a small cost. It requires a translation step the end to end model does not, and that step is one of the places it breaks.

## Practical rules for the grounding prompt

- keep distinguishing words the user actually said, so "red cube" not "cube"
- do not invent distinguishing words, so "the cube" stays "cube"
- include the destination, since the robot has to find that too
- cap the number of queries, because more queries slow OWLv2 and add false positives

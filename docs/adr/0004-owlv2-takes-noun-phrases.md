# OWLv2 takes short noun phrases, so Architecture A needs a language model in front of its detector

OWLv2 turns each query into one CLIP embedding matched against image patches, so a whole sentence matches no region, and nothing splits it into objects or drops its verbs. A language model therefore converts the command into noun phrases ("red cube", "metal tray") first. This is a cost of the modular design and a failure mode B does not have (H9): a wrong noun phrase ends the run before perception starts. Keep the user's distinguishing words, add none, include the destination, and cap the query count.

Updated 26 Oct 2026: the conversion is now the planner's own `perceive` call, not a separate `extract_queries` call to the same model, and it can run again mid-task with new words (ADR 0012). The rules above moved into the `perceive` tool description. A query that finds nothing now goes back to the planner, so it no longer ends the run on its own.

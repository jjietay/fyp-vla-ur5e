"""suite.py

It takes nothing and gives you the frozen definition of what both architectures
are scored on: the tiers, the utterances, and the layouts.

This file is the experimental design. It is written before either pipeline is
tuned, and changing it after tuning starts means the benchmark has been fitted to
whichever architecture happened to be built first. If a change is unavoidable,
record the date and the reason in `CHANGELOG` at the bottom and say so in the
report. That is the difference between a protocol and a rationalisation.

Lives in `fyp.evaluation` rather than under either architecture on purpose. The
harness must not be able to see inside A or B, or it will grow an affordance for
one of them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    """The task tiers. Ordered by what they test, not by difficulty."""
    PICK_PLACE = 0      # verification that both pipelines are wired correctly
    POUR = 1            # motion that is hard to script
    AMBIGUITY = 2       # instruction that does not determine the target
    DYNAMIC = 3         # world state changes mid-execution
    GENERALISATION = 4  # unseen objects and unseen phrasings


@dataclass(frozen=True)
class Task:
    """
    One tier's definition: what is on the table, what gets said, and what counts
    as success.

    `success_criterion` is prose, and deliberately so. It is read aloud to the
    person scoring the trial, and it belongs in the report verbatim. A criterion
    that lives only in someone's head is how a marginal trial becomes a success
    on a good day and a failure on a bad one.
    """
    tier: Tier
    name: str
    objects: tuple[str, ...]
    train_utterances: tuple[str, ...]
    heldout_utterances: tuple[str, ...]
    success_criterion: str
    perturbation: str | None = None
    notes: str = ""

    @property
    def queries(self) -> list[str]:
        """
        It takes nothing and gives you this scene's object names, for setting the
        cell up and for reading the layout sheet.

        NOT for feeding to a detector during a scored trial. Architecture A
        derives its own vocabulary from the instruction, because handing it this
        list would be information Architecture B never receives.
        """
        return list(self.objects)


# Utterances are split into two sets. `train_utterances` are the phrasings used
# when recording Architecture B's demonstrations and shown as examples to
# Architecture A. `heldout_utterances` are never used for either, and exist only
# to test robustness to phrasing drift, which is what a spoken interface actually
# produces. Scoring on training phrasings alone would flatter Architecture B.

TASKS: dict[Tier, Task] = {
    Tier.PICK_PLACE: Task(
        tier=Tier.PICK_PLACE,
        name="pick and place",
        objects=("red cube", "metal tray"),
        train_utterances=(
            "place that red cube into that metal tray",
            "put the red cube in the metal tray",
            "pick up the red cube and drop it in the tray",
            "move the red cube to the tray",
            "the red cube goes in the metal tray",
        ),
        heldout_utterances=(
            "could you pop that red block into the metal tray",
            "I need the red cube moved into the tray please",
            "stick the red one in the tray",
        ),
        success_criterion=(
            "The red cube is fully inside the metal tray, the gripper is clear of "
            "the tray, and the arm has returned above the workspace. Cube resting "
            "on the tray rim counts as a failure."
        ),
        notes="Verification only. Not a headline result. Deictic phrasing is deliberate.",
    ),
    Tier.POUR: Task(
        tier=Tier.POUR,
        name="pour, unambiguous",
        objects=("orange juice bottle", "water bottle", "glass"),
        train_utterances=(
            "pour the orange juice into the glass",
            "fill the glass with orange juice",
            "put some orange juice in the glass",
            "give me orange juice in the glass",
            "pour me some orange juice",
        ),
        heldout_utterances=(
            "can you fill that glass up with the orange one",
            "I would like the OJ poured out please",
            "serve the orange juice into the glass",
        ),
        success_criterion=(
            "Liquid is transferred from the orange juice bottle into the glass, "
            "the glass remains upright, the bottle is returned to the table upright, "
            "and no liquid lands outside the tray."
        ),
        notes="Both bottles present but the instruction names one, so nothing to clarify.",
    ),
    Tier.AMBIGUITY: Task(
        tier=Tier.AMBIGUITY,
        name="ambiguous instruction",
        objects=("orange juice bottle", "water bottle", "glass"),
        train_utterances=(
            "pour me a drink",
            "I am thirsty",
            "get me something to drink",
            "fill the glass",
            "pour something into the glass",
        ),
        heldout_utterances=(
            "could I have a drink",
            "sort me out with a drink would you",
            "put a drink in that glass",
        ),
        success_criterion=(
            "The system asks which drink is wanted BEFORE any bottle is grasped, "
            "and then pours the drink the user named. Pouring without asking is a "
            "failure even when the guess happens to match."
        ),
        notes=(
            "The headline experiment. Architecture B cannot ask, so record WHICH "
            "bottle it chooses on every trial. The distribution of that choice is "
            "the result, not the success rate."
        ),
    ),
    Tier.DYNAMIC: Task(
        tier=Tier.DYNAMIC,
        name="drawer, closed mid-task",
        objects=("snack packet", "drawer handle"),
        train_utterances=(
            "put the snack in the drawer",
            "place the snack packet into the drawer",
            "store the snack in the drawer",
            "the snack goes in the drawer",
            "put that snack away in the drawer",
        ),
        heldout_utterances=(
            "tidy the snack into the drawer",
            "can the snack go in the drawer please",
            "stow that snack",
        ),
        success_criterion=(
            "The snack ends up inside the open drawer. The operator closes the "
            "drawer once, after it has been opened and before the snack is "
            "released; the system must re-open it. Releasing the snack onto a "
            "closed drawer is a failure."
        ),
        perturbation=(
            "Close the drawer by hand exactly once, at the moment the arm begins "
            "moving toward the snack. Same moment every trial."
        ),
        notes="Expect Architecture B to win here. Reporting that is what makes Tier 2 credible.",
    ),
    Tier.GENERALISATION: Task(
        tier=Tier.GENERALISATION,
        name="unseen objects",
        objects=("blue mug", "wooden bowl"),
        train_utterances=(),
        heldout_utterances=(
            "put the blue mug in the wooden bowl",
            "place that blue mug into the bowl",
            "the mug goes in the bowl",
        ),
        success_criterion=(
            "The blue mug is fully inside the wooden bowl. Neither object appeared "
            "in any demonstration or prompt example."
        ),
        notes=(
            "Vary objects and phrasing INDEPENDENTLY across the trial block. If both "
            "change at once and B fails, you cannot say which caused it."
        ),
    ),
}


@dataclass(frozen=True)
class Layout:
    """One starting arrangement, reproducible from its seed."""
    index: int
    tier: Tier
    positions: dict[str, tuple[float, float]]   # object -> (x, y) on the table, metres
    utterance: str
    heldout: bool

    def describe(self) -> str:
        """It takes nothing and gives you the setup instruction for the person resetting the cell."""
        rows = ", ".join(f"{k} at ({x:+.3f}, {y:+.3f})" for k, (x, y) in self.positions.items())
        tag = "held-out phrasing" if self.heldout else "training phrasing"
        return f"L{self.index:02d}  {rows}\n      say: \"{self.utterance}\"  [{tag}]"


def build_layouts(tier: Tier, n: int = 20, seed: int = 20260811,
                  x_range: tuple[float, float] = (0.30, 0.55),
                  y_range: tuple[float, float] = (-0.20, 0.20),
                  min_separation: float = 0.12) -> list[Layout]:
    """
    It takes a tier and gives you n reproducible starting layouts, half using
    training phrasings and half held-out ones.

    Seeded so the same 20 layouts are used for both architectures, in the same
    order. Randomising per architecture would mean the two were scored on
    different problems, and any difference in success rate could be the layouts.

    Objects are kept `min_separation` apart because two items closer than the
    gripper is wide make a failed grasp a property of the layout rather than of
    the architecture.
    """
    task = TASKS[tier]
    rng = random.Random(seed + int(tier) * 1000)
    train = list(task.train_utterances)
    held = list(task.heldout_utterances)

    layouts: list[Layout] = []
    for i in range(n):
        positions: dict[str, tuple[float, float]] = {}
        for obj in task.objects:
            for _ in range(200):
                p = (round(rng.uniform(*x_range), 3), round(rng.uniform(*y_range), 3))
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_separation ** 2
                       for q in positions.values()):
                    positions[obj] = p
                    break
            else:
                raise RuntimeError(
                    f"could not place {obj} with {min_separation} m separation in "
                    f"x{x_range} y{y_range}. Widen the ranges or reduce the separation.")

        use_held = (i % 2 == 1) and held
        pool = held if use_held else (train or held)
        layouts.append(Layout(index=i, tier=tier, positions=positions,
                              utterance=pool[i % len(pool)], heldout=bool(use_held)))
    return layouts


CHANGELOG: list[str] = [
    "2026-08-11  Suite created before either pipeline was tuned.",
]

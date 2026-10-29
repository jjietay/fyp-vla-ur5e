"""Unit tests for the evaluation harness, with a fake agent and a scripted operator.

The harness knows an agent only through `run(instruction) -> outcome`, so a fake
agent is the honest way to test it: anything it does with a real architecture it
does with this one. The operator is the person typing the scores, scripted here.
"""
from __future__ import annotations

import csv
import time
from types import SimpleNamespace

from fyp.evaluation.harness import Harness
from fyp.evaluation.suite import Tier


class FakeEars:
    def hear(self, prompt):
        return SimpleNamespace(text="place that red cube into that metal tray")


class FakeAgent:
    """It takes an outcome and gives you an agent that returns it, after taking
    `busy_s` seconds of wall-clock time."""

    def __init__(self, busy_s=0.0, **outcome):
        self.busy_s = busy_s
        self.outcome = SimpleNamespace(ok=True, failed_stage=None, asked=[], run_id="r1",
                                       summary="I did it.", **outcome)

    def run(self, instruction):
        end = time.perf_counter() + self.busy_s
        while time.perf_counter() < end:
            pass
        return self.outcome


class Operator:
    """It takes the answers to type and gives you an `operator` callable that
    records every prompt it was shown."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        # Enter (empty) for everything not scripted: skip nothing, 0 interventions.
        return self.answers.pop(0) if self.answers else ""


def _one_trial(tmp_path, agent, operator):
    h = Harness(agent, "A", FakeEars(), cfg={}, out_dir=tmp_path, n_trials=1,
                operator=operator)
    return h.run_tier(Tier(0))


def test_the_operators_typing_time_is_not_the_architectures_time(tmp_path):
    """
    A per-step verdict makes the arm wait for a typed answer. That wait is
    inside agent.run, so the harness takes the agent's `operator_s` off the
    duration, or A's latency includes how fast the operator types.
    """
    agent = FakeAgent(busy_s=0.30, operator_s=0.25)
    rec = _one_trial(tmp_path, agent, Operator("", "y"))[0]
    assert 0.03 <= rec.duration_s <= 0.10, rec.duration_s


def test_nothing_the_agent_concluded_is_shown_before_the_success_verdict(tmp_path, capsys):
    """
    13 Oct 2026: A's own report said "picked up" while the orange stayed on the
    table. Printing that report before the question pushes the operator towards
    yes, and the verdict stops being independent of the system it judges.
    """
    shown_before: list[str] = []

    def operator(prompt):
        if "SUCCESS" in prompt:
            shown_before.append(capsys.readouterr().out + prompt)
            return "y"
        return ""

    _one_trial(tmp_path, FakeAgent(), operator)
    assert shown_before, "the success question was never asked"
    assert "I did it." not in shown_before[0]
    assert "reported" not in shown_before[0]


def test_success_comes_from_the_operator_even_when_the_agent_says_ok(tmp_path):
    rec = _one_trial(tmp_path, FakeAgent(), Operator("", "n"))[0]
    assert rec.success is False


def test_the_operators_step_verdicts_are_written_beside_the_trials(tmp_path):
    """
    Each step has two verdicts, the agent's and the operator's. Written per
    step, with the trial's run_id, they give the step success rates and how
    often the agent's own check agreed with what happened.
    """
    judged = [{"step": "pick {'xyz': [0.4, -0.3, 0.2]}", "tool": "pick",
               "a_ok": True, "operator_ok": False}]
    _one_trial(tmp_path, FakeAgent(judged=judged), Operator("", "n"))

    with (tmp_path / "A_tier0_steps.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["run_id"], r["tool"], r["a_ok"], r["operator_ok"]) for r in rows] == \
        [("r1", "pick", "True", "False")]


def test_an_agent_without_step_verdicts_writes_no_step_file(tmp_path):
    """Architecture B has no steps. The harness must not need it to pretend."""
    _one_trial(tmp_path, FakeAgent(), Operator("", "y"))
    assert not (tmp_path / "A_tier0_steps.csv").exists()


def test_the_step_file_is_not_read_back_as_trials(tmp_path):
    from fyp.evaluation.harness import load_records, summarise

    judged = [{"step": "pick", "tool": "pick", "a_ok": True, "operator_ok": True}]
    _one_trial(tmp_path, FakeAgent(judged=judged), Operator("", "y"))

    rows = load_records(tmp_path)
    assert len(rows) == 1 and rows[0]["architecture"] == "A"
    assert "SUCCESS RATE BY TIER" in summarise(rows)


def test_the_step_prompt_asks_again_until_it_gets_y_or_n():
    """A stray key at the bench must not be recorded as a verdict."""
    from fyp.evaluation.harness import judge_step

    op = Operator("", "maybe", "Y")
    assert judge_step("pick {'xyz': [0.4, -0.3, 0.2]}", operator=op) is True
    assert len(op.prompts) == 3 and "pick" in op.prompts[0]


def test_a_trial_records_its_api_cost_and_the_peoples_time_apart(tmp_path):
    """
    End-to-end latency runs from the end of speech recognition to when the agent
    stops. The user's spoken answers and the operator's typing happen inside it,
    so they are recorded beside it and taken off the architecture's own time.
    """
    usage = {"calls": 3, "input_tokens": 1200, "output_tokens": 300,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 800,
             "cost_usd": 0.00556}
    agent = FakeAgent(busy_s=0.40, operator_s=0.10, clarify_s=0.20, usage=usage)
    rec = _one_trial(tmp_path, agent, Operator("", "y"))[0]

    assert 0.38 <= rec.latency_s <= 0.50, rec.latency_s
    assert (rec.operator_s, rec.clarify_s) == (0.1, 0.2)
    assert abs(rec.duration_s - (rec.latency_s - 0.30)) < 0.02
    assert (rec.api_calls, rec.input_tokens, rec.output_tokens) == (3, 1200, 300)
    assert rec.cache_read_tokens == 800 and rec.api_cost_usd == 0.00556
    assert rec.agent_said == "I did it."

    row = next(csv.DictReader(open(tmp_path / "A_tier0.csv")))
    assert float(row["api_cost_usd"]) == 0.00556


def test_an_agent_that_reports_chunk_times_has_them_recorded(tmp_path):
    """Architecture B's wrapper reports how long each action chunk took to infer."""
    agent = FakeAgent(timing={"chunk_ms": [80.0, 100.0, 120.0]})
    rec = _one_trial(tmp_path, agent, Operator("", "y"))[0]
    assert (rec.chunks, rec.chunk_ms_mean, rec.chunk_ms_max) == (3, 100.0, 120.0)
    assert rec.api_cost_usd == 0.0

"""Tests for the confidence gate.

This is the largest mechanism in the system after clarification and state tracking, and it
is the one with a genuine failure mode: it works by *withholding* recommendations, and a
gate that never releases scores 0.0000 rather than merely scoring badly. The turn cap is
what stands between the two, so it is asserted here rather than trusted.

These are behavioural tests on the agent, not measurements. The measured value of the gate
(+0.0579, CI [+0.0457, +0.0703]) lives in `results/ablation_ci.json` and is asserted against
the README by `tests/test_documentation.py`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from copilot.agent import ShoppingCopilot
from copilot.config import DEFAULT_CONFIG

_OPENER = "I'm looking for Accessories Scarves, but I'm still exploring."


def _agent(catalog, index, **overrides):
    config = replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG
    agent = ShoppingCopilot(catalog, config=config, index=index)
    agent.reset("gate", {})
    return agent


# ------------------------------------------------------------------ default behaviour
def test_an_uninformed_opening_turn_shows_one_recommendation(synthetic_catalog, synthetic_index):
    """Browsing opens with no constraints, so the gate should be active immediately."""
    agent = _agent(synthetic_catalog, synthetic_index)
    response = agent.respond("gate", _OPENER, 1, 10)
    assert len(response["recommendations"]) == 1


def test_the_gate_never_returns_an_empty_list(synthetic_catalog, synthetic_index):
    """One, not zero. A withheld list cannot convert; a single correct guess converts at 1.

    This is also the protocol-conformance argument: the agent recommends on every turn, it
    simply recommends narrowly while under-informed.
    """
    agent = _agent(synthetic_catalog, synthetic_index)
    for turn in range(1, 5):
        response = agent.respond("gate", _OPENER, turn, 10)
        assert response["recommendations"], f"turn {turn} returned nothing"


def test_it_still_asks_a_question_while_gated(synthetic_catalog, synthetic_index):
    """Withholding the list is only worth anything if the turn still buys information."""
    agent = _agent(synthetic_catalog, synthetic_index)
    response = agent.respond("gate", _OPENER, 1, 10)
    assert response["ask_attribute"] is not None


# ------------------------------------------------------------------ the release valve
def test_the_gate_releases_after_the_turn_cap(synthetic_catalog, synthetic_index):
    """Past `gate_max_turn` the full list must come back even with no new evidence.

    Without this the mechanism has a catastrophic mode: an uninformative session is gated
    forever and can never hit. The sweep measured that at 0.0000.
    """
    agent = _agent(synthetic_catalog, synthetic_index, gate_max_turn=2)
    gated = agent.respond("gate", _OPENER, 1, 10)
    released = agent.respond("gate", "Those options are not quite right yet.", 3, 10)
    assert len(gated["recommendations"]) == 1
    assert len(released["recommendations"]) > 1, "the gate never released"


def test_enough_constraints_release_the_gate_before_the_cap(synthetic_catalog, synthetic_index):
    """Evidence, not just elapsed turns, should open it."""
    agent = _agent(synthetic_catalog, synthetic_index, gate_min_constraints=1)
    response = agent.respond(
        "gate", "I'm looking for Shirts T-Shirts. A key requirement is: moisture wicking polyester.",
        1, 10,
    )
    assert len(response["recommendations"]) > 1


# ----------------------------------------------------------------------- the switch
def test_disabling_the_gate_restores_the_previous_behaviour(synthetic_catalog, synthetic_index):
    """`use_confidence_gate=False` must be an exact restoration, so the ablation reproduces."""
    gated = _agent(synthetic_catalog, synthetic_index)
    plain = _agent(synthetic_catalog, synthetic_index, use_confidence_gate=False)
    assert len(gated.respond("gate", _OPENER, 1, 10)["recommendations"]) == 1
    assert len(plain.respond("gate", _OPENER, 1, 10)["recommendations"]) == min(
        10, synthetic_index.count
    )


@pytest.mark.parametrize("size", [1, 2, 3])
def test_the_gated_list_length_follows_its_setting(synthetic_catalog, synthetic_index, size):
    agent = _agent(synthetic_catalog, synthetic_index, gate_list_size=size)
    assert len(agent.respond("gate", _OPENER, 1, 10)["recommendations"]) == size


def test_the_gated_list_is_the_head_of_the_ungated_one(synthetic_catalog, synthetic_index):
    """Gating must truncate the ranking, never reorder it."""
    gated = _agent(synthetic_catalog, synthetic_index, gate_list_size=2)
    plain = _agent(synthetic_catalog, synthetic_index, use_confidence_gate=False)
    short = [r["parent_asin"] for r in gated.respond("gate", _OPENER, 1, 10)["recommendations"]]
    full = [r["parent_asin"] for r in plain.respond("gate", _OPENER, 1, 10)["recommendations"]]
    assert short == full[: len(short)]

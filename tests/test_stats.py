"""Tests for the resampling statistics.

A confidence interval is a claim about uncertainty, and a wrong one is worse than none: it
launders noise into apparent precision. These tests pin the two properties the README leans
on -- that the interval brackets the point estimate and narrows with sample size -- and
that the composite is computed the way the official evaluator computes it, since a CI
around a subtly different statistic would be quoting an interval for a number nobody
reports.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from evaluator.local_evaluator import metric_summary
from tools.stats import METRIC_NAMES, _metrics, bootstrap_metrics

_ROOT = Path(__file__).resolve().parent.parent


def _sessions(hits: int, total: int, turn: int = 2) -> list[dict]:
    return [
        {
            "hit": index < hits,
            "first_hit_turn": turn if index < hits else None,
            "reciprocal_rank": 1.0 if index < hits else 0.0,
        }
        for index in range(total)
    ]


# ------------------------------------------------------- agreement with the evaluator
def test_metrics_match_the_evaluators_own_aggregation():
    """`_metrics` re-implements `metric_summary`; drift between them is silent."""
    sessions = _sessions(hits=37, total=50, turn=3)
    official = metric_summary(sessions)
    mine = _metrics(sessions)
    assert mine["hit_rate_at_10"] == pytest.approx(official["hit_rate_at_10"])
    assert mine["mrr"] == pytest.approx(official["mrr"])
    assert mine["mttc"] == pytest.approx(official["mttc"])


def test_a_miss_costs_eleven_turns_not_zero():
    """Treating a miss as turn 0 would make failure look efficient."""
    perfect = _metrics(_sessions(hits=1, total=1, turn=2))
    missed = _metrics(_sessions(hits=0, total=1))
    assert missed["mttc"] == 11.0
    assert missed["technical_score"] < perfect["technical_score"]


def test_the_composite_reproduces_the_committed_official_score():
    """The end-to-end check: same records in, same headline number out."""
    path = _ROOT / "results" / "official_evaluation.json"
    if not path.exists():  # pragma: no cover - results are committed
        pytest.skip("results/official_evaluation.json not present")
    report = json.loads(path.read_text(encoding="utf-8"))
    recomputed = _metrics(report["sessions"])["technical_score"]
    assert recomputed == pytest.approx(report["recommended_technical_score"], abs=5e-7)


# --------------------------------------------------------------------- the bootstrap
def test_every_headline_metric_gets_an_interval():
    summary = bootstrap_metrics(_sessions(90, 100), random.Random(3), rounds=400)
    assert set(summary["metrics"]) == set(METRIC_NAMES)
    for stats in summary["metrics"].values():
        assert stats["ci95_low"] <= stats["point"] <= stats["ci95_high"]


def test_intervals_narrow_as_the_sample_grows():
    small = bootstrap_metrics(_sessions(45, 50), random.Random(3), rounds=400)
    large = bootstrap_metrics(_sessions(900, 1000), random.Random(3), rounds=400)
    for name in METRIC_NAMES:
        narrow = large["metrics"][name]
        wide = small["metrics"][name]
        assert (narrow["ci95_high"] - narrow["ci95_low"]) <= (
            wide["ci95_high"] - wide["ci95_low"]
        ), name


def test_a_run_with_no_variation_has_a_degenerate_interval():
    summary = bootstrap_metrics(_sessions(50, 50), random.Random(3), rounds=200)
    composite = summary["metrics"]["technical_score"]
    assert composite["ci95_low"] == composite["ci95_high"]
    assert composite["std_error"] == 0.0


def test_an_empty_session_list_yields_no_interval():
    assert bootstrap_metrics([], random.Random(3)) == {}


def test_the_bootstrap_is_reproducible_from_its_seed():
    first = bootstrap_metrics(_sessions(70, 100), random.Random(7), rounds=300)
    second = bootstrap_metrics(_sessions(70, 100), random.Random(7), rounds=300)
    assert first == second


def test_components_and_composite_come_from_the_same_draws():
    """Resampling each metric independently would let the reported rows contradict."""
    summary = bootstrap_metrics(_sessions(80, 100), random.Random(5), rounds=500)
    metrics = summary["metrics"]
    # Every draw satisfies the 50/30/20 identity, so the point estimates must too.
    efficiency = max(0.0, min(1.0, (11.0 - metrics["mttc"]["point"]) / 10.0))
    implied = (
        0.50 * metrics["hit_rate_at_10"]["point"]
        + 0.30 * metrics["mrr"]["point"]
        + 0.20 * efficiency
    )
    assert metrics["technical_score"]["point"] == pytest.approx(implied, abs=1e-6)

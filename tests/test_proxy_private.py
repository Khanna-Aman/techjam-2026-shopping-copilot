"""Tests for the held-out generalisation harness.

The value of `tools/proxy_private.py` rests entirely on the synthetic sessions being
structurally indistinguishable from real ones and on the popularity match being real. Both
are asserted here: a sample the evaluator would reject, or a match that quietly drifts,
would turn the reported generalisation gap into a fabrication.
"""

from __future__ import annotations

import random

import pytest

from evaluator.local_evaluator import materialize_hidden_fields
from tests.conftest import SYNTHETIC_PRODUCTS
from tools.proxy_private import (
    SCENARIO_MIX,
    _rating_number,
    _scenarios_for,
    bootstrap_interval,
    build_samples,
    eligible_targets,
    popularity_summary,
    sample_matched,
    sample_uniform,
)

PRODUCTS = {str(item["parent_asin"]): item for item in SYNTHETIC_PRODUCTS}


# --------------------------------------------------------------------- scenario mix
def test_scenario_apportionment_is_exact_and_matches_the_published_mix():
    for count in (10, 100, 110, 137, 1000):
        scenarios = _scenarios_for(count)
        assert len(scenarios) == count
        for name, share in SCENARIO_MIX:
            # Largest-remainder apportionment: never more than one off the ideal share.
            assert abs(scenarios.count(name) - count * share) <= 1


def test_scenario_apportionment_covers_every_published_scenario():
    scenarios = set(_scenarios_for(200))
    assert scenarios == {name for name, _ in SCENARIO_MIX}


# ------------------------------------------------------------------ target selection
def test_eligible_targets_excludes_the_public_targets():
    pool = eligible_targets(PRODUCTS, exclude={"B000000001"})
    assert "B000000001" not in pool


def test_eligible_targets_requires_features_and_details():
    # B000000004 has empty details {}, so it cannot produce a well-formed intent card.
    pool = eligible_targets(PRODUCTS, exclude=set())
    assert "B000000004" not in pool
    assert "B000000001" in pool


def test_matched_sampling_never_returns_an_excluded_target():
    exclude = {"B000000001", "B000000002"}
    pool = eligible_targets(PRODUCTS, exclude=exclude)
    reference = [_rating_number(PRODUCTS[a]) for a in exclude]
    chosen, _ = sample_matched(PRODUCTS, pool, reference, random.Random(1), 10)
    assert not (set(chosen) & exclude)


def test_matched_sampling_reports_the_feasible_ceiling():
    """The catalog, not the request, is what caps this regime -- and it must say so."""
    pool = eligible_targets(PRODUCTS, exclude=set())
    reference = [_rating_number(PRODUCTS[a]) for a in pool]
    chosen, feasible = sample_matched(PRODUCTS, pool, reference, random.Random(1), 10_000)
    assert feasible <= len(pool) * 10
    assert len(chosen) <= feasible


def test_uniform_sampling_is_bounded_by_the_pool():
    pool = eligible_targets(PRODUCTS, exclude=set())
    assert len(sample_uniform(pool, random.Random(1), 10_000)) == len(pool)


# ----------------------------------------------------------------- session synthesis
def _samples(count: int = 20) -> list[dict]:
    pool = eligible_targets(PRODUCTS, exclude=set())
    rng = random.Random(7)
    targets = [rng.choice(pool) for _ in range(count)]
    profile = {"preference_tags": ["fit", "comfort"], "average_prior_rating": 4.0}
    return build_samples(targets, [profile], rng, prefix="test")


def test_synthetic_samples_carry_every_field_the_evaluator_reads():
    for sample in _samples():
        assert isinstance(sample["sample_id"], str) and sample["sample_id"]
        assert sample["scenario_type"] in {name for name, _ in SCENARIO_MIX}
        assert isinstance(sample["user_profile"], dict)
        assert sample["ground_truth"]["parent_asin"] in PRODUCTS


def test_synthetic_sample_ids_are_unique():
    samples = _samples(50)
    assert len({sample["sample_id"] for sample in samples}) == len(samples)


def test_synthetic_samples_carry_no_hidden_fields():
    """Real public samples ship without them; the simulator materialises them itself.

    Supplying our own would let the harness choose what the customer knows, which is
    exactly the bias this tool exists to avoid.
    """
    for sample in _samples():
        assert "intent_card" not in sample
        assert "behavior" not in sample


def test_the_official_simulator_accepts_a_synthetic_sample():
    """The load-bearing assertion: the organiser's own code must drive these sessions."""
    for sample in _samples():
        card, behavior = materialize_hidden_fields(sample, PRODUCTS)
        assert card["hard_constraints"] or card["soft_preferences"]
        assert behavior["scenario_type"] == sample["scenario_type"]
        if sample["scenario_type"] == "intent_override":
            assert behavior["override"]["turn"] in (3, 4)


def test_synthesis_is_deterministic_for_a_fixed_seed():
    assert [s["sample_id"] for s in _samples()] == [s["sample_id"] for s in _samples()]
    assert [s["ground_truth"] for s in _samples()] == [s["ground_truth"] for s in _samples()]


# ------------------------------------------------------------------------ statistics
def _sessions(hits: int, total: int) -> list[dict]:
    return [
        {
            "hit": index < hits,
            "first_hit_turn": 2 if index < hits else None,
            "reciprocal_rank": 1.0 if index < hits else 0.0,
        }
        for index in range(total)
    ]


def test_bootstrap_interval_brackets_the_point_estimate():
    sessions = _sessions(hits=90, total=100)
    interval = bootstrap_interval(sessions, random.Random(3), rounds=500)
    point = 0.50 * 0.90 + 0.30 * 0.90 + 0.20 * max(0.0, (11.0 - 2.9) / 10.0)
    assert interval["ci95_low"] <= point <= interval["ci95_high"]


def test_bootstrap_interval_narrows_as_the_sample_grows():
    small = bootstrap_interval(_sessions(45, 50), random.Random(3), rounds=500)
    large = bootstrap_interval(_sessions(900, 1000), random.Random(3), rounds=500)
    assert (large["ci95_high"] - large["ci95_low"]) < (small["ci95_high"] - small["ci95_low"])


def test_bootstrap_interval_handles_an_empty_session_list():
    assert bootstrap_interval([], random.Random(3)) == {}


def test_a_perfect_run_has_a_degenerate_interval():
    interval = bootstrap_interval(_sessions(50, 50), random.Random(3), rounds=200)
    assert interval["ci95_low"] == interval["ci95_high"]


# --------------------------------------------------------------------- match quality
def test_popularity_summary_is_ordered():
    summary = popularity_summary(PRODUCTS, sorted(PRODUCTS))
    assert summary["p25_rating_number"] <= summary["median_rating_number"]
    assert summary["median_rating_number"] <= summary["p75_rating_number"]
    assert 0.0 <= summary["priced_fraction"] <= 1.0


@pytest.mark.parametrize("count", [40, 200])
def test_matched_sampling_tracks_the_reference_distribution(real_index, count):
    """On the real catalog, the matched regime must actually match -- not merely claim to.

    A drifting match would silently turn the generalisation number into a comparison
    between two different tasks.
    """
    from evaluator.local_evaluator import catalog_index, load_jsonl

    _, _, products = catalog_index(real_index.path)
    public = load_jsonl("data/public_set.jsonl")
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    reference = [_rating_number(products[a]) for a in public_targets if a in products]

    pool = eligible_targets(products, public_targets)
    chosen, _ = sample_matched(products, pool, reference, random.Random(11), count)

    reference_median = sorted(reference)[len(reference) // 2]
    chosen_median = popularity_summary(products, chosen)["median_rating_number"]
    # Popularity spans five orders of magnitude here, so compare on ratio, not difference.
    assert 0.5 <= chosen_median / reference_median <= 2.0


def test_matched_sampling_ignores_the_order_of_the_reference_list():
    """Same popularities in a different order must draw the same targets.

    This is the bug that made the harness non-reproducible, and it is worth stating exactly
    because the first guard written for it did not catch it. `main()` built the reference
    list by iterating a *set* of product ids, whose order changes with per-process hash
    randomisation. `sample_matched` draws by indexing into that list, so two runs at the same
    seed drew different targets and scored 0.9444 and 0.9474.

    A test that passes an already-sorted reference cannot detect this -- it asserts a
    property that held before the fix too. The defect is *order sensitivity*, so the test
    shuffles the reference and demands the same output, which fails against the unsorted
    implementation and needs no subprocess or catalog to do it.
    """
    products = {
        f"P{i:04d}": {"rating_number": i * 7 % 991, "features": ["f"], "details": {"d": 1}}
        for i in range(1, 400)
    }
    pool = eligible_targets(products, set())
    reference = [products[a]["rating_number"] for a in pool[:120]]

    shuffled = list(reference)
    random.Random(3).shuffle(shuffled)
    assert shuffled != reference, "the shuffle did not change the order; test is vacuous"

    first, _ = sample_matched(products, pool, reference, random.Random(11), 40)
    second, _ = sample_matched(products, pool, shuffled, random.Random(11), 40)

    assert first == second, (
        "sample_matched drew different targets from the same popularities in a different "
        "order; the reference list has become order-dependent again and the harness is no "
        "longer reproducible across processes"
    )

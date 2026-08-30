"""Tests for the latency and memory harness.

The feasibility table is the submission's strongest claim, so the harness behind it has to
be trustworthy in the two ways it could quietly lie: reporting a percentile that is not the
percentile it says, and reporting a memory figure it did not actually obtain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.latency import _distribution, _median_of_runs, _percentile, resident_bytes

_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------ percentiles
def test_percentile_is_nearest_rank_not_interpolated():
    """An interpolated p95 would report a latency no turn actually took."""
    ordered = [float(n) for n in range(1, 101)]
    assert _percentile(ordered, 0.95) in ordered
    assert _percentile(ordered, 0.95) == 95.0
    assert _percentile(ordered, 1.0) == 100.0


def test_percentile_handles_the_degenerate_cases():
    assert _percentile([], 0.95) == 0.0
    assert _percentile([7.0], 0.95) == 7.0
    # A fraction below the first rank must still return a real observation.
    assert _percentile([3.0, 4.0], 0.0) == 3.0


def test_distribution_is_ordered_and_self_consistent():
    timings = [5.0, 1.0, 100.0, 2.0, 3.0, 4.0, 50.0, 20.0, 10.0, 7.0]
    summary = _distribution(timings)
    assert summary["turns"] == 10
    assert summary["max"] == 100.0
    assert summary["median"] <= summary["p95"] <= summary["max"]
    assert summary["p95"] <= summary["p99"] <= summary["max"]


def test_distribution_does_not_mutate_its_input():
    """`_time_turns` hands over a live list; sorting it in place would be a silent bug."""
    timings = [3.0, 1.0, 2.0]
    original = list(timings)
    _distribution(timings)
    assert timings == original


# ---------------------------------------------------------------- aggregation of runs
def test_median_of_runs_takes_the_middle_pass_per_statistic():
    runs = [
        {"turns": 100, "median": 10.0, "mean": 20.0, "p95": 90.0, "p99": 95.0, "max": 99.0},
        {"turns": 100, "median": 12.0, "mean": 22.0, "p95": 70.0, "p99": 96.0, "max": 98.0},
        {"turns": 100, "median": 11.0, "mean": 21.0, "p95": 80.0, "p99": 97.0, "max": 97.0},
    ]
    summary = _median_of_runs(runs)
    assert summary["median"] == 11.0
    assert summary["p95"] == 80.0
    assert summary["turns"] == 100


def test_median_of_runs_survives_a_single_pass():
    run = {"turns": 5, "median": 1.0, "mean": 2.0, "p95": 3.0, "p99": 4.0, "max": 5.0}
    assert _median_of_runs([run])["p95"] == 3.0


# ------------------------------------------------------------------------- memory
def test_resident_bytes_is_a_plausible_size_or_an_honest_none():
    """Never a zero or a negative: a wrong-looking number is worse than a missing one."""
    value = resident_bytes()
    if value is None:
        pytest.skip("resident set size is not readable on this platform")
    assert value > 1_000_000, "a live interpreter cannot be resident in under a megabyte"


# ------------------------------------------------------------- the committed artifact
def test_the_committed_latency_report_is_internally_consistent():
    path = _ROOT / "results" / "latency.json"
    if not path.exists():  # pragma: no cover - results are committed
        pytest.skip("results/latency.json not present")
    report = json.loads(path.read_text(encoding="utf-8"))

    scored = report["per_turn_ms"]["scored_loop"]
    exhaustive = report["per_turn_ms"]["all_ten_turns"]

    # Driving every session to ten turns must produce strictly more turns than a loop that
    # stops on a hit; if it does not, `early_stop` is not doing anything.
    assert exhaustive["turns"] > scored["turns"]
    assert exhaustive["turns"] == report["sessions"] * 10

    for summary in (scored, exhaustive):
        assert summary["median"] <= summary["p95"] <= summary["p99"] <= summary["max"]

    build = report["index_build_seconds"]
    assert build["cold"] > build["warm_from_cache"], "a cache load should beat a full build"

    memory = report["memory_mb"]
    assert memory["agent_only_resident"] < memory["harness_after_run"], (
        "the agent alone must be smaller than the harness that also holds the simulator"
    )
    assert len(report["per_turn_ms_runs"]["scored_loop"]) == report["repeats"]

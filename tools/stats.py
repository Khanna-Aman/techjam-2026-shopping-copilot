"""Resampling statistics over evaluated sessions.

Every headline number in this repository is a mean over a finite set of sessions -- 200 for
the public set, fewer for some diagnostics -- and a mean over 200 draws is not a constant.
Reporting `0.906151` to six figures implies a precision the sample size does not support.
These helpers put an interval around it.

The composite is recomputed here exactly as `evaluator.local_evaluator` computes it, from
the same per-session fields, so a resampled score is comparable to the official one rather
than merely correlated with it. `_metrics` is the single definition of that arithmetic and
both callers go through it.

**What the interval does and does not cover.** A percentile bootstrap over sessions
estimates how much the score would move if you drew a *different sample of sessions from
the same distribution*. That is the right uncertainty for "is this difference real?" on the
public set. It says nothing about the private set being distributed differently -- targets
drawn from a different popularity regime, or turns paraphrased. That is distribution shift,
not sampling noise, and it is measured separately by `tools/proxy_private.py` and
`tools/robustness.py`. Quoting a bootstrap CI as though it bounded private-set performance
would be exactly the kind of overclaim this file exists to prevent.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import MAX_TURNS  # noqa: E402

#: The metrics a bootstrap draw produces, in the order they are reported.
METRIC_NAMES: tuple[str, ...] = ("technical_score", "hit_rate_at_10", "mrr", "mttc")


def _metrics(sessions: list[dict]) -> dict[str, float]:
    """The evaluator's aggregation, over an arbitrary list of session records.

    A miss contributes `MAX_TURNS + 1` to MTTC, which is what `metric_summary` does; the
    efficiency clamp and the 50/30/20 composite are `evaluate`'s.
    """
    size = len(sessions)
    hit_rate = sum(int(item["hit"]) for item in sessions) / size
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1
        for item in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "technical_score": 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency,
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
    }


def bootstrap_metrics(
    sessions: list[dict], rng: random.Random, rounds: int = 2000
) -> dict:
    """Percentile bootstrap over sessions, for every headline metric at once.

    Returns the point estimate on the observed sample alongside the 2.5th and 97.5th
    percentiles of the resampled distribution, plus its standard deviation as a standard
    error. Resampling all metrics from the *same* draw keeps them mutually consistent --
    the composite in a given round is the composite of that round's components.
    """
    if not sessions:
        return {}
    size = len(sessions)
    point = _metrics(sessions)
    draws: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(rounds):
        sample = [sessions[rng.randrange(size)] for _ in range(size)]
        for name, value in _metrics(sample).items():
            draws[name].append(value)

    summary: dict[str, dict] = {}
    low_index = int(0.025 * rounds)
    high_index = min(int(0.975 * rounds), rounds - 1)
    for name in METRIC_NAMES:
        ordered = sorted(draws[name])
        summary[name] = {
            "point": round(point[name], 6),
            "ci95_low": round(ordered[low_index], 6),
            "ci95_high": round(ordered[high_index], 6),
            "std_error": round(statistics.pstdev(ordered), 6),
        }
    return {"sample_count": size, "rounds": rounds, "metrics": summary}


def paired_delta_interval(
    baseline: list[dict],
    variant: list[dict],
    rng: random.Random,
    rounds: int = 20000,
) -> dict:
    """Bootstrap the *difference* between two runs over the same sessions.

    An ablation is a paired comparison: both configurations answer the identical 200
    sessions, so most of the variance in either score is shared -- a session with an
    obscure target is hard for both. Comparing two marginal intervals throws that pairing
    away and will call a real effect insignificant simply because each run's own interval
    is wide. Resampling session *indices* once per round and applying them to both runs
    keeps the pairing, which is why a delta of 0.001 can be resolvable here while the
    marginal intervals on 0.906 overlap almost completely.

    `significant` reports whether the interval excludes zero. That is a statement about
    sampling noise on the public set and nothing more: an effect can be real, tiny, and
    still not worth shipping.
    """
    if not baseline or len(baseline) != len(variant):
        return {}
    size = len(baseline)
    point = _metrics(variant)["technical_score"] - _metrics(baseline)["technical_score"]
    deltas: list[float] = []
    for _ in range(rounds):
        picks = [rng.randrange(size) for _ in range(size)]
        base_draw = [baseline[index] for index in picks]
        variant_draw = [variant[index] for index in picks]
        deltas.append(
            _metrics(variant_draw)["technical_score"]
            - _metrics(base_draw)["technical_score"]
        )
    deltas.sort()
    low = deltas[int(0.025 * rounds)]
    high = deltas[min(int(0.975 * rounds), rounds - 1)]
    return {
        "delta": round(point, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "significant": bool(low > 0.0 or high < 0.0),
    }

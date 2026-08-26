"""Held-out generalisation harness: how far does the public score actually travel?

The agent is tuned on 200 public sessions, but 800 private sessions decide the result. That
gap is the largest unquantified risk in the submission, and stating it as a caveat is weaker
than measuring it.

This harness measures it. The public samples carry **no** hidden fields -- only a target
`parent_asin`, a scenario label and an anonymised profile (see `data/public_set.jsonl`). The
intent card and the customer's behaviour are materialised from the catalog product at
evaluation time by the organiser's own `materialize_hidden_fields`. So a synthetic sample
built from a *different* target product is not an approximation of a real session: it is
structurally identical, and it is driven through the unmodified official `evaluate()` loop.
Only the choice of target product differs.

Two sampling regimes, because they answer different questions.

``matched``
    Target popularity is decile-matched to the public set. Public targets are drawn from a
    5-core split and are overwhelmingly popular: their median `rating_number` is ~7,000
    against a catalog median of 12, and 89% carry a price against 21% catalog-wide. A
    uniform sample would therefore be a much harder task and would understate the score for
    reasons that have nothing to do with generalisation. The catalog caps this regime at
    roughly 110 sessions -- the public 200 have already consumed most of the high-review
    tail -- so it is small, and the reported interval is correspondingly wide.

    The match is on popularity, and it is close: median rating_number 6,845 against the
    public set's 6,614. It is *not* exact on everything. Matched targets carry a price 55%
    of the time against the public set's 89%, because popularity and price coverage are not
    the same axis and the tail is too thin to constrain both. Budget is disclosed in roughly
    0.5% of turns, so the residual should be immaterial -- but it is a real difference, it is
    reported in the output under `target_popularity`, and it is not being hidden.

``uniform``
    Uniform over every eligible product. Deliberately harsher than the private set is
    likely to be, since it abandons the popularity regularity entirely. This is the
    pessimistic bound, and the gap between the two regimes is a direct measurement of how
    much the popularity prior depends on how sessions are sampled -- a dependence the README
    flags as a limitation but could not previously quantify.

Scores produced here are DIAGNOSTIC ONLY and are not the official metric.

Usage:
    python -m tools.proxy_private                    # both regimes
    python -m tools.proxy_private --mode matched
    python -m tools.proxy_private --n-uniform 2000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
)

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.catalog import CatalogIndex  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402

#: The published scenario mix, as it appears in the 200 public sessions.
SCENARIO_MIX: tuple[tuple[str, float], ...] = (
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
)

#: Difficulty label is a function of scenario in the public set (buying=easy,
#: browsing/boundary=medium, intent_override=hard). Reproduced for structural fidelity;
#: nothing in the agent or the evaluator reads it.
_DIFFICULTY = {
    "buying": "easy",
    "browsing": "medium",
    "boundary": "medium",
    "intent_override": "hard",
}

DECILES = 10


# ------------------------------------------------------------------ target selection
def eligible_targets(products: dict[str, dict], exclude: set[str]) -> list[str]:
    """Products that can produce a well-formed intent card.

    ``intent_card`` mines constraints from ``features`` and ``details``; every one of the
    200 public targets carries both. A product missing them yields a degenerate card that
    falls back to the title, which is not a session the private set would contain either.
    """
    return sorted(
        asin
        for asin, product in products.items()
        if asin not in exclude and product.get("features") and product.get("details")
    )


def _rating_number(product: dict) -> int:
    try:
        return int(product.get("rating_number") or 0)
    except (TypeError, ValueError):
        return 0


def _decile_edges(values: list[int]) -> list[int]:
    ordered = sorted(values)
    last = len(ordered) - 1
    return [ordered[round(q / DECILES * last)] for q in range(DECILES + 1)]


def sample_matched(
    products: dict[str, dict],
    pool: list[str],
    reference: list[int],
    rng: random.Random,
    requested: int,
) -> tuple[list[str], int]:
    """Draw targets whose popularity distribution matches ``reference`` by decile.

    Returns the sample and the largest size the catalog could actually support, which is
    almost always the binding constraint rather than ``requested``.
    """
    edges = _decile_edges(reference)
    buckets: list[list[str]] = [[] for _ in range(DECILES)]
    for asin in pool:
        count = _rating_number(products[asin])
        for index in range(DECILES):
            low, high = edges[index], edges[index + 1]
            # The final decile is open-ended; the rest are half-open so the edges do not
            # double-count a product sitting exactly on a boundary.
            if (low <= count <= high) if index == DECILES - 1 else (low <= count < high):
                buckets[index].append(asin)
                break

    per_decile = min(len(bucket) for bucket in buckets)
    feasible = per_decile * DECILES
    take = min(requested, feasible) // DECILES

    chosen: list[str] = []
    for bucket in buckets:
        chosen.extend(rng.sample(bucket, take))
    rng.shuffle(chosen)
    return chosen, feasible


def sample_uniform(pool: list[str], rng: random.Random, count: int) -> list[str]:
    return rng.sample(pool, min(count, len(pool)))


# ------------------------------------------------------------------ session synthesis
def _scenarios_for(count: int) -> list[str]:
    """Apportion scenarios by the published mix, largest-remainder so the total is exact."""
    raw = [(name, count * share) for name, share in SCENARIO_MIX]
    counts = {name: int(value) for name, value in raw}
    shortfall = count - sum(counts.values())
    for name, value in sorted(raw, key=lambda item: -(item[1] - int(item[1]))):
        if shortfall <= 0:
            break
        counts[name] += 1
        shortfall -= 1
    out: list[str] = []
    for name, _ in SCENARIO_MIX:
        out.extend([name] * counts[name])
    return out


def build_samples(
    targets: list[str], profiles: list[dict], rng: random.Random, prefix: str
) -> list[dict]:
    """Assemble samples in exactly the shape the official evaluator consumes.

    Profiles are resampled from the public set's own profile pool rather than invented, so
    the profile distribution is identical and the target product is the only thing that
    varies. The agent reads only ``preference_tags`` from a profile in any case.
    """
    scenarios = _scenarios_for(len(targets))
    rng.shuffle(scenarios)
    samples: list[dict] = []
    for position, (asin, scenario) in enumerate(zip(targets, scenarios), start=1):
        samples.append(
            {
                "sample_id": f"{prefix}_{position:05d}",
                "scenario_type": scenario,
                "category_bucket": "clothing",
                "difficulty_bucket": _DIFFICULTY[scenario],
                "user_profile": rng.choice(profiles),
                "ground_truth": {"parent_asin": asin},
            }
        )
    return samples


# ------------------------------------------------------------------------ statistics
def bootstrap_interval(
    sessions: list[dict], rng: random.Random, rounds: int = 2000
) -> dict:
    """Percentile bootstrap over sessions for the composite score.

    At the sample sizes the matched regime allows, a point estimate on its own would imply
    a precision the data does not support.
    """
    if not sessions:
        return {}
    scores: list[float] = []
    size = len(sessions)
    for _ in range(rounds):
        draw = [sessions[rng.randrange(size)] for _ in range(size)]
        hit_rate = sum(int(item["hit"]) for item in draw) / size
        mrr = statistics.fmean(item["reciprocal_rank"] for item in draw)
        mttc = statistics.fmean(
            item["first_hit_turn"] if item["first_hit_turn"] is not None else 11
            for item in draw
        )
        efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
        scores.append(0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency)
    scores.sort()
    return {
        "ci95_low": round(scores[int(0.025 * rounds)], 6),
        "ci95_high": round(scores[int(0.975 * rounds)], 6),
    }


def popularity_summary(products: dict[str, dict], targets: list[str]) -> dict:
    counts = sorted(_rating_number(products[asin]) for asin in targets)
    last = len(counts) - 1
    return {
        "median_rating_number": counts[last // 2],
        "p25_rating_number": counts[round(0.25 * last)],
        "p75_rating_number": counts[round(0.75 * last)],
        "priced_fraction": round(
            sum(1 for a in targets if products[a].get("price") not in (None, "")) / len(targets), 4
        ),
    }


# ------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out generalisation harness")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--mode", default="both", choices=("matched", "uniform", "both"))
    parser.add_argument("--n-matched", type=int, default=200)
    parser.add_argument("--n-uniform", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", default="results/proxy_private.json")
    parser.add_argument(
        "--base", default=None, help="JSON object of AgentConfig overrides"
    )
    args = parser.parse_args()

    public = load_jsonl(args.dataset)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]

    catalog_ids, categories, products = catalog_index(args.catalog)
    reference = [_rating_number(products[a]) for a in public_targets if a in products]

    config: AgentConfig = DEFAULT_CONFIG
    if args.base:
        overrides = json.loads(args.base)
        allowed = set(AgentConfig.__dataclass_fields__)
        config = replace(config, **{k: v for k, v in overrides.items() if k in allowed})

    index = CatalogIndex(args.catalog)
    pool = eligible_targets(products, public_targets)

    regimes = ("matched", "uniform") if args.mode == "both" else (args.mode,)
    results: dict[str, dict] = {}

    print(f"eligible non-public targets: {len(pool)} of {len(products)}")
    print(f"{'regime':<10}{'n':>7}{'score':>9}{'hit@10':>9}{'MRR':>9}{'MTTC':>8}  95% CI")
    print("-" * 72)

    for regime in regimes:
        rng = random.Random(args.seed)
        if regime == "matched":
            targets, feasible = sample_matched(products, pool, reference, rng, args.n_matched)
            note = f"decile-matched; catalog supports at most {feasible}"
        else:
            targets = sample_uniform(pool, rng, args.n_uniform)
            note = "uniform over eligible targets"

        samples = build_samples(targets, profiles, rng, prefix=f"proxy_{regime}")
        # Share the prebuilt index across regimes rather than paying the build cost twice.
        agent = ShoppingCopilot(args.catalog, config=config, index=index)
        outcome = evaluate(agent, samples, catalog_ids, categories, products)
        interval = bootstrap_interval(outcome["sessions"], random.Random(args.seed))

        results[regime] = {
            "note": note,
            "sample_count": outcome["sample_count"],
            "hit_rate_at_10": outcome["hit_rate_at_10"],
            "mrr": outcome["mrr"],
            "mttc": outcome["mttc"],
            "technical_score": outcome["recommended_technical_score"],
            **interval,
            "target_popularity": popularity_summary(products, targets),
            "scenario_metrics": outcome["scenario_metrics"],
        }
        row = results[regime]
        print(
            f"{regime:<10}{row['sample_count']:>7}{row['technical_score']:>9.4f}"
            f"{row['hit_rate_at_10']:>9.3f}{row['mrr']:>9.4f}{row['mttc']:>8.2f}"
            f"  [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )

    results["public_reference"] = {
        "note": "the official public-set score, for comparison",
        "target_popularity": popularity_summary(products, sorted(public_targets)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

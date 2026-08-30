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
    Target popularity follows the public set's. Public targets come from the organiser's
    Clothing 5-core leave-last-out split (docs/PARTICIPANT_KIT_README.md) and are therefore
    overwhelmingly popular -- median `rating_number` 6,846, the 99.4th percentile of the
    catalog, against a catalog median of 12 -- because a leave-last-out target is somebody's
    real last purchase, and real purchases concentrate on popular products. `docs` also says
    both splits use the same fixed scenario mix and the same deterministic sampling, so the
    private 800 carry this same skew. This is the regime to read for generalisation.

    **This regime was rebuilt.** It previously took an equal count from each of ten
    popularity deciles, which capped it at ten times the *smallest* decile. The smallest
    decile is the most popular one, holding 11 eligible products, so the sample was 110 of
    43,149 available targets and its 95% interval was 0.046 wide. An effect of a few
    thousandths could not have been resolved, and `w_popularity=1.2` was rejected partly on
    a reading of this regime that the sample size did not support. It now draws a reference
    popularity per target and takes the nearest unused product, following the whole shape of
    the distribution at any sample size. The residual bias is stated in `sample_matched`:
    the catalog is thin at the top, so a large sample drifts *less* popular than the truth,
    which makes this harder than reality rather than easier.

``uniform``
    Uniform over every eligible product. **This is an out-of-distribution stress test and
    must not be read as an estimate of the private-set score.** It draws targets with fewer
    than 100 ratings 81% of the time; the real split does so 5% of the time. It answers "how
    much does the ranking lean on the popularity regularity", which is a fair question about
    fragility, and it is reported for that reason. It is not a generalisation estimate, and
    treating it as one is the specific mistake that cost this submission a correct
    calibration of `w_popularity` for several days.

Scores produced here are DIAGNOSTIC ONLY and are not the official metric.

Usage:
    python -m tools.proxy_private                    # both regimes
    python -m tools.proxy_private --mode matched
    python -m tools.proxy_private --n-uniform 2000
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    catalog_index,
    evaluate,
    load_jsonl,
)

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.catalog import CatalogIndex  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402

from tools.stats import bootstrap_metrics  # noqa: E402

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
    """Draw targets whose popularity distribution follows ``reference``.

    This replaces an equal-count-per-decile scheme that capped the sample at 110 of the
    43,149 eligible targets -- 10 x the *smallest* decile, and the smallest decile is the
    most popular one, which held 11 products. Matching popularity was the goal, and the
    popularity skew was what starved the sample. At n=110 the 95% interval was 0.046 wide,
    so an effect of a few thousandths was invisible; that is how `w_popularity=1.2` came
    to be rejected on a test that could not have detected it.

    Instead we draw one reference popularity per target and take the nearest unused
    product to it, which follows the whole shape of the reference rather than forcing
    equal mass into ten buckets, and scales to any requested size.

    Note the residual bias, because it matters when reading the result: the catalog holds
    far fewer very popular products than the reference wants, so a large sample must drift
    downward in popularity (at n=800, median ~3.2k ratings against the public set's 6.8k).
    That drift makes this test *harder* than the real distribution, so a gain measured
    here is a lower bound rather than an optimistic one.
    """
    if not reference or not pool:
        return [], 0
    # Sorted here as well as at the call site, deliberately. We draw by *indexing* into
    # `reference`, so its order decides which targets come out; a caller that builds it by
    # iterating a set hands us an order that changes with hash randomisation, and the whole
    # harness stops being reproducible across processes. That happened -- two runs at one
    # seed scored 0.9444 and 0.9474 -- so the invariant is enforced where it is relied on
    # rather than trusted to every caller.
    reference = sorted(reference)
    by_count = sorted((_rating_number(products[asin]), asin) for asin in pool)
    counts = [count for count, _ in by_count]
    feasible = min(requested, len(pool))

    used: set[str] = set()
    chosen: list[str] = []
    while len(chosen) < feasible:
        target_count = reference[rng.randrange(len(reference))]
        start = bisect.bisect_left(counts, target_count)
        # Walk outward from the reference popularity until an unused product turns up.
        for offset in range(len(by_count)):
            hit = None
            for index in (start + offset, start - offset):
                if 0 <= index < len(by_count) and by_count[index][1] not in used:
                    hit = by_count[index][1]
                    break
            if hit is not None:
                used.add(hit)
                chosen.append(hit)
                break
    rng.shuffle(chosen)
    return chosen, feasible


def sample_uniform(pool: list[str], rng: random.Random, count: int) -> list[str]:
    return rng.sample(pool, min(count, len(pool)))


def paired_delta(
    base_sessions: list[dict], other_sessions: list[dict], rng: random.Random,
    rounds: int = 20000,
) -> dict:
    """Bootstrap the per-session score difference between two configs on the same sessions.

    Marginal intervals on two scores can overlap while the difference between them is
    unambiguous, because the two runs answer the *same* sessions and the variance they share
    cancels in the difference. Comparing the two `technical_score` intervals reported per
    regime is the weaker test, and the weight recalibration rests on this one, so it needs a
    command that produces it rather than a number quoted from a notebook.

    The composite is a mean of per-session terms, so it decomposes exactly:
    ``0.5*hit + 0.3*rr + 0.2*(11 - turn)/10``, with a miss counted at ``MAX_TURNS + 1``
    exactly as the official evaluator does.
    """
    def per_session(sessions: list[dict]) -> dict[str, float]:
        out: dict[str, float] = {}
        for item in sessions:
            turn = item["first_hit_turn"]
            turn = (MAX_TURNS + 1) if turn is None else turn
            out[item["sample_id"]] = (
                0.5 * int(item["hit"]) + 0.3 * item["reciprocal_rank"] + 0.02 * (11 - turn)
            )
        return out

    left, right = per_session(base_sessions), per_session(other_sessions)
    shared = sorted(set(left) & set(right))
    deltas = [left[key] - right[key] for key in shared]
    if not deltas:
        return {}
    means = []
    for _ in range(rounds):
        means.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    means.sort()
    return {
        "n": len(deltas),
        "delta": round(sum(deltas) / len(deltas), 6),
        "ci95_low": round(means[int(0.025 * rounds)], 6),
        "ci95_high": round(means[int(0.975 * rounds)], 6),
        "improved": sum(1 for value in deltas if value > 1e-9),
        "regressed": sum(1 for value in deltas if value < -1e-9),
    }


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

    The resampling itself lives in `tools.stats`, which reports every headline metric; this
    regime table only has room for the composite, so it takes that pair of columns. See
    `tools/bootstrap.py` for the full breakdown on the public set.
    """
    summary = bootstrap_metrics(sessions, rng, rounds=rounds)
    if not summary:
        return {}
    composite = summary["metrics"]["technical_score"]
    return {
        "ci95_low": composite["ci95_low"],
        "ci95_high": composite["ci95_high"],
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
    parser.add_argument("--n-matched", type=int, default=800)
    parser.add_argument("--n-uniform", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", default="results/proxy_private.json")
    parser.add_argument(
        "--base", default=None, help="JSON object of AgentConfig overrides"
    )
    parser.add_argument(
        "--against", default=None,
        help=(
            "JSON object of AgentConfig overrides for a second configuration. Both are run "
            "over the SAME synthesised sessions and the difference is bootstrapped pairwise, "
            "which is the test a weight change has to pass -- two overlapping marginal "
            "intervals say much less than one paired one."
        ),
    )
    args = parser.parse_args()

    public = load_jsonl(args.dataset)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]

    catalog_ids, categories, products = catalog_index(args.catalog)
    # sorted() is load-bearing, not tidiness: `public_targets` is a set of strings, so its
    # iteration order changes with per-process hash randomisation. The decile sampler this
    # replaced only ever passed `reference` through `_decile_edges`, which sorts internally,
    # so the order was invisible. `sample_matched` now indexes into it, which made the whole
    # harness non-reproducible across processes -- two runs at the same seed drew different
    # targets and scored 0.9444 and 0.9474.
    reference = sorted(
        _rating_number(products[a]) for a in sorted(public_targets) if a in products
    )

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
            note = (f"popularity-matched to the public targets, n={feasible}; the catalog "
                    f"is thin at the top so this drifts less popular, making it a lower bound")
        else:
            targets = sample_uniform(pool, rng, args.n_uniform)
            note = ("OUT-OF-DISTRIBUTION STRESS TEST, not an estimate of private-set score: "
                    "uniform catalog sampling draws sub-100-rating targets 81% of the time "
                    "against the real split's 5%. Read `matched` for generalisation.")

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

        if args.against:
            rival = replace(
                config,
                **{
                    key: value
                    for key, value in json.loads(args.against).items()
                    if key in set(AgentConfig.__dataclass_fields__)
                },
            )
            rival_outcome = evaluate(
                ShoppingCopilot(args.catalog, config=rival, index=index),
                samples, catalog_ids, categories, products,
            )
            comparison = paired_delta(
                outcome["sessions"], rival_outcome["sessions"], random.Random(args.seed)
            )
            comparison["against"] = json.loads(args.against)
            comparison["against_technical_score"] = rival_outcome[
                "recommended_technical_score"
            ]
            results[regime]["paired_comparison"] = comparison
            verdict = (
                "gain" if comparison["ci95_low"] > 0
                else "harm" if comparison["ci95_high"] < 0
                else "spans zero"
            )
            print(
                f"{'':<10}{'':>7}{'paired vs --against':>9} "
                f"{comparison['delta']:+.4f} "
                f"[{comparison['ci95_low']:+.4f}, {comparison['ci95_high']:+.4f}] {verdict} "
                f"(+{comparison['improved']}/-{comparison['regressed']})"
            )
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

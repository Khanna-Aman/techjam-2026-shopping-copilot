"""Confidence intervals for the public-set headline numbers.

`results/official_evaluation.json` reports the composite as `0.906151`. Six significant
figures is a fact about floating-point arithmetic, not about the agent: the score is a mean
over 200 sessions, and a different 200 would give a different answer. This harness says how
different.

It resamples the *already-evaluated* per-session records rather than re-running the
evaluator, because the bootstrap is a statement about which sessions were drawn, not about
non-determinism in the agent -- the agent is deterministic, so re-running would reproduce
the same 200 records every time and add nothing. Reading the committed records also makes
the output verifiable against the file it came from: `point` for the composite must equal
`recommended_technical_score` exactly, and `tests/test_documentation.py` asserts it does.
If the two ever disagree, one of the files is stale.

The interval covers sampling variation across sessions from the *public* distribution. It
is not a bound on private-set performance -- see the module docstring in `tools/stats.py`
for why, and `tools/proxy_private.py` for the measurement that does address it.

Usage:
    python -m tools.bootstrap                 # writes results/bootstrap.json
    python -m tools.bootstrap --rounds 20000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.stats import METRIC_NAMES, bootstrap_metrics  # noqa: E402

#: Below this many sessions a percentile interval is too wide to be worth acting on. The
#: boundary scenario has 10, so the caveat is not hypothetical.
_SMALL_SAMPLE = 30

_LABELS = {
    "technical_score": "TechnicalScore",
    "hit_rate_at_10": "Hit@10",
    "mrr": "MRR",
    "mttc": "MTTC",
}


def _format_row(name: str, stats: dict) -> str:
    # ASCII only: this prints to a Windows console under CI, where cp1252 mangles "+/-".
    return (
        f"{_LABELS[name]:<16}{stats['point']:>9.4f}"
        f"   [{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]"
        f"   +/-{stats['std_error']:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(_ROOT / "results" / "official_evaluation.json"),
        help="evaluation whose per-session records get resampled",
    )
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--out", default=str(_ROOT / "results" / "bootstrap.json"))
    args = parser.parse_args(argv)

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    sessions = source["sessions"]
    if not sessions:
        print("source carries no per-session records", file=sys.stderr)
        return 1

    overall = bootstrap_metrics(sessions, random.Random(args.seed), rounds=args.rounds)

    # The point estimate is computed from the records, not copied from the header, so an
    # equality check here is a real consistency test rather than a tautology.
    reported = source["recommended_technical_score"]
    recomputed = overall["metrics"]["technical_score"]["point"]
    if abs(recomputed - reported) > 5e-7:
        print(
            f"resampled point estimate {recomputed} disagrees with the source's "
            f"reported {reported} -- the source file is internally inconsistent",
            file=sys.stderr,
        )
        return 1

    by_scenario: dict[str, dict] = {}
    for scenario in sorted({item["scenario_type"] for item in sessions}):
        subset = [item for item in sessions if item["scenario_type"] == scenario]
        summary = bootstrap_metrics(subset, random.Random(args.seed), rounds=args.rounds)
        summary["underpowered"] = len(subset) < _SMALL_SAMPLE
        by_scenario[scenario] = summary

    payload = {
        "source": Path(args.source).name,
        "source_technical_score": reported,
        "rounds": args.rounds,
        "seed": args.seed,
        "note": (
            "Percentile bootstrap over public-set sessions. Covers sampling variation "
            "within the public distribution only; distribution shift to the private set "
            "is measured by tools/proxy_private.py, not here."
        ),
        "overall": overall,
        "by_scenario": by_scenario,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"public set, n={overall['sample_count']}, {args.rounds} rounds\n")
    print(f"{'metric':<16}{'point':>9}   {'95% CI':<20}  std err")
    for name in METRIC_NAMES:
        print(_format_row(name, overall["metrics"][name]))
    print("\nby scenario (TechnicalScore)\n")
    for scenario, summary in by_scenario.items():
        stats = summary["metrics"]["technical_score"]
        flag = "  (underpowered)" if summary["underpowered"] else ""
        print(
            f"{scenario:<18}n={summary['sample_count']:>4}  {stats['point']:.4f}"
            f"   [{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]{flag}"
        )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

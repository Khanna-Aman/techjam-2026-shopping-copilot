"""Which ablation rows are real, and which are noise?

The ablation table in the README reports deltas down to `0.0000`. Four figures look like
precision, but the composite is a mean over 200 sessions with a standard error of about
0.008 -- so a naive reading says nothing below ~0.016 is distinguishable, and half the
table is noise.

That naive reading is wrong, and wrong in the conservative direction. An ablation is a
**paired** comparison: both configurations answer the same 200 sessions, in the same order,
with the same simulator seed. The variance shared between the two runs cancels in the
difference. Bootstrapping the delta directly (`tools.stats.paired_delta_interval`) resolves
effects far smaller than either marginal interval would suggest.

So this harness answers the question the ablation table cannot: for each mechanism, is the
measured contribution distinguishable from sampling noise on the public set? Rows whose
interval spans zero are reported as such rather than quietly dropped -- "we removed this and
could not measure a difference" is a finding, and it is the honest label for the MMR and
top-10 padding rows.

A caveat that belongs next to every "significant" flag: this bounds *sampling noise on the
public set*. It is not evidence the effect transfers to the private set, and a significant
effect can still be far too small to justify its complexity.

Usage:
    python -m tools.ablation_ci                     # writes results/ablation_ci.json
    python -m tools.ablation_ci --rounds 5000       # faster, slightly coarser
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.catalog import CatalogIndex  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402
from tools.robustness import PERTURBATIONS, run  # noqa: E402
from tools.stats import paired_delta_interval  # noqa: E402
from tools.sweep import ABLATIONS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--perturbation",
        default="control",
        choices=sorted(PERTURBATIONS),
        help=(
            "wording applied to every customer message. A mechanism that measures zero on "
            "the clean set may still be load-bearing once the phrasing moves, so 'heavy' "
            "asks whether a guard earns its place rather than assuming it does."
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    perturb = PERTURBATIONS[args.perturbation]
    out_path = Path(args.out) if args.out else _ROOT / "results" / (
        "ablation_ci.json" if args.perturbation == "control"
        else f"ablation_ci_{args.perturbation}.json"
    )

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    index = CatalogIndex(args.catalog)
    base: AgentConfig = DEFAULT_CONFIG

    # The default perturbation is `control`, the no-op, so this reproduces the official
    # loop exactly while retaining the per-session records a paired test needs.
    def evaluate_config(overrides: dict) -> dict:
        agent = ShoppingCopilot(args.catalog, config=replace(base, **overrides), index=index)
        return run(
            agent, samples, catalog_ids, categories, products, perturb, keep_sessions=True
        )

    label_full, overrides_full = ABLATIONS[0]
    assert overrides_full == {}, "the first ablation row must be the unmodified system"
    full = evaluate_config(overrides_full)
    print(f"{label_full}: {full['technical_score']:.6f}  (n={full['sample_count']})\n")
    print(f"{'configuration':<26}{'score':>9}{'delta':>10}   {'95% CI on the delta':<24}")
    print("-" * 76)

    rows: dict[str, dict] = {}
    for label, overrides in ABLATIONS[1:]:
        variant = evaluate_config(overrides)
        interval = paired_delta_interval(
            full["sessions"], variant["sessions"], random.Random(args.seed), args.rounds
        )
        rows[label] = {
            "overrides": overrides,
            "technical_score": variant["technical_score"],
            **interval,
        }
        verdict = "" if interval["significant"] else "   spans zero"
        print(
            f"{label:<26}{variant['technical_score']:>9.4f}{interval['delta']:>+10.4f}"
            f"   [{interval['ci95_low']:+.4f}, {interval['ci95_high']:+.4f}]{verdict}"
        )

    payload = {
        "note": (
            "Paired percentile bootstrap of the ablation delta over the 200 public "
            "sessions. Bounds sampling noise on the public set only; it is not evidence "
            "that an effect transfers to the private set."
        ),
        "perturbation": args.perturbation,
        "rounds": args.rounds,
        "seed": args.seed,
        "full_system": {
            "technical_score": full["technical_score"],
            "sample_count": full["sample_count"],
        },
        "ablations": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")

    resolved = sum(1 for row in rows.values() if row["significant"])
    print(f"{resolved} of {len(rows)} mechanisms are distinguishable from noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

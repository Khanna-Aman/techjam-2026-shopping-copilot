"""Re-derive the P(yield) prior from the catalog, using the organiser's own rules.

`copilot/question.py` ships `YIELD_PROBABILITY` as seven hardcoded constants, with a comment
saying they were measured over the catalog. That comment was the only evidence. Hardcoded
priors in a submission scored on a hidden set invite an obvious and fair suspicion -- that
they were fitted to the 200 public sessions -- and "no, they came from the catalog" is a
claim, not a demonstration.

This harness demonstrates it. For every product in the 50,000-item catalog it builds the
intent card with the evaluator's own `intent_card`, classifies each constraint with the
evaluator's own `classify_constraint`, and counts the fraction of products whose card
carries at least one constraint of each type. That fraction *is* P(the customer can answer a
question about this attribute), because `customer_reply` answers exactly when such a
constraint exists and is still undisclosed.

Two properties worth stating plainly:

* **No session labels are used.** The public set is never opened. The input is the catalog,
  which every participant has, so the prior is reproducible by anyone and is not fitted to
  the sessions it is later scored on.
* **The evaluator is imported, not reimplemented.** If the organiser's classifier and mine
  disagreed, a reimplementation would hide it. Importing means the numbers are theirs.

`tests/test_documentation.py` asserts the shipped constants against the file this writes, so
the two cannot drift apart silently.

Usage:
    python -m tools.yield_prior              # writes results/yield_prior.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    classify_constraint,
    intent_card,
)

from copilot.question import YIELD_PROBABILITY  # noqa: E402


def derive(catalog: str) -> dict:
    _ids, _categories, products = catalog_index(catalog)

    carries: Counter[str] = Counter()
    for product in products.values():
        card = intent_card(product)
        constraints = [
            *[str(value) for value in card.get("hard_constraints", [])],
            *[str(value) for value in card.get("soft_preferences", [])],
        ]
        # "At least one", not "how many": the question is whether the customer *can* answer,
        # and a card holding two colour constraints still only answers the colour question
        # once before the attribute is exhausted.
        for attribute in {classify_constraint(value) for value in constraints}:
            carries[attribute] += 1

    total = len(products)
    measured = {
        attribute: round(carries.get(attribute, 0) / total, 6)
        for attribute in sorted(YIELD_PROBABILITY)
    }
    return {
        "products": total,
        "note": (
            "P(the intent card carries at least one constraint of this type), over the "
            "whole catalog, using the evaluator's own intent_card and classify_constraint. "
            "No session labels are involved."
        ),
        "measured": measured,
        "shipped": dict(YIELD_PROBABILITY),
        "max_abs_difference": round(
            max(abs(measured[a] - YIELD_PROBABILITY[a]) for a in measured), 6
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default=str(_ROOT / "results" / "yield_prior.json"))
    args = parser.parse_args(argv)

    report = derive(args.catalog)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{report['products']:,} products\n")
    print(f"{'attribute':<12}{'measured':>10}{'shipped':>10}{'diff':>10}")
    for attribute, value in sorted(report["measured"].items(), key=lambda kv: -kv[1]):
        shipped = YIELD_PROBABILITY[attribute]
        print(f"{attribute:<12}{value:>10.4f}{shipped:>10.4f}{value - shipped:>+10.4f}")
    print(f"\nlargest disagreement: {report['max_abs_difference']:.4f}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

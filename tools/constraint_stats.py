"""Why dense retrieval cannot help here: the constraints are already exact strings.

The dense sweeps (`tools/sweep.py --mode dense`) all land on the same shape -- a layer that
is harmless at low weight and harmful at high weight, and never resolvably useful. That
result needs a *structural* explanation or it is just a number, so this harness measures the
property of the task that produces it.

Dense retrieval exists to close a vocabulary gap: the shopper says one thing, the product
listing says another, and a lexical matcher misses the connection. This benchmark has almost
no such gap, and not by accident -- `evaluator.local_evaluator.intent_card` builds the
customer's hidden requirements *by copying substrings out of the target product's own
fields*. The customer is quoting the listing back at us. So the harness asks three questions
of every string the simulator is able to disclose:

``verbatim``
    Does it occur, character for character, in its own target's searchable text? If yes,
    there is no vocabulary gap for an embedding to bridge on that constraint.

``unique``
    Does it occur in exactly one product out of 50,000? If yes, that single constraint
    identifies the answer outright, and any smoothing of the match can only lose it.

``narrow``
    Does it occur in ten or fewer? Ten is the evaluator's TOP_K, so this is the fraction of
    constraints that on their own would put the target on the scored page.

Method, stated because the choices move the numbers:

* The universe is `hard_constraints + soft_preferences` from the materialised intent card of
  each session -- exactly the list `customer_reply` draws from, so every string counted here
  is one the customer can actually say. Four per session, 800 in all.
* Containment is tested against the evaluator's own `searchable_text`, not our index's
  fields, so the measurement is of the benchmark rather than of our preprocessing.
* Both sides are whitespace-collapsed and lowercased. Case matters more than it looks: the
  same measurement case-sensitively reports 84.4% verbatim rather than 94.5%, because the
  card lowercases the material it injects while the listing capitalises it. Lowercasing is
  the right choice -- our retrieval is case-insensitive, so a case-only difference is not a
  vocabulary gap anyone has to close -- but it is a choice, and it is worth 10 points.

The report also splits mined constraints (copied from `features`/`details`) from the three
the card synthesises -- a bare material word, `color: <x>`, `budget around $<n>` -- because
those behave oppositely and the aggregate hides it.

Diagnostic only. Produces no score.

Usage:
    python -m tools.constraint_stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    TOP_K,
    catalog_index,
    load_jsonl,
    materialize_hidden_fields,
    searchable_text,
)

_SYNTHETIC_RE = re.compile(r"^(color: |budget around \$)")
_MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _origin(value: str) -> str:
    """Did the card copy this string out of the product, or build it?

    `intent_card` injects three kinds of string that are not lifted verbatim from a field:
    a bare material word found by regex, a ``color: x`` label, and a price line. Everything
    else is a substring of `features` or `details`.
    """
    if _SYNTHETIC_RE.match(value) or _normalise(value) in _MATERIALS:
        return "synthesised"
    return "mined"


def analyse(catalog: str, dataset: str) -> dict:
    samples = load_jsonl(dataset)
    _, _, products = catalog_index(catalog)
    blobs = {asin: _normalise(searchable_text(product)) for asin, product in products.items()}

    records: list[dict] = []
    for sample in samples:
        card, _ = materialize_hidden_fields(sample, products)
        target = str(sample["ground_truth"]["parent_asin"])
        constraints = [
            *[str(value) for value in card.get("hard_constraints", [])],
            *[str(value) for value in card.get("soft_preferences", [])],
        ]
        for value in dict.fromkeys(constraints):
            needle = _normalise(value)
            matches = sum(1 for blob in blobs.values() if needle and needle in blob)
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "scenario_type": sample["scenario_type"],
                    "origin": _origin(value),
                    "chars": len(needle),
                    "verbatim_in_target": bool(needle) and needle in blobs[target],
                    "catalog_matches": matches,
                }
            )

    def _summarise(rows: list[dict]) -> dict:
        total = len(rows)
        if not total:
            return {"constraints": 0}
        verbatim = sum(1 for row in rows if row["verbatim_in_target"])
        unique = sum(1 for row in rows if row["catalog_matches"] == 1)
        narrow = sum(1 for row in rows if 1 <= row["catalog_matches"] <= TOP_K)
        return {
            "constraints": total,
            "verbatim_in_target": verbatim,
            "unique_in_catalog": unique,
            "narrows_to_top_k_or_fewer": narrow,
            "pct_verbatim_in_target": round(100.0 * verbatim / total, 1),
            "pct_unique_in_catalog": round(100.0 * unique / total, 1),
            "pct_narrows_to_top_k_or_fewer": round(100.0 * narrow / total, 1),
        }

    return {
        "catalog_size": len(products),
        "sessions": len(samples),
        "top_k": TOP_K,
        "summary": _summarise(records),
        "by_origin": {
            origin: _summarise([row for row in records if row["origin"] == origin])
            for origin in ("mined", "synthesised")
        },
        "constraints": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--out", default=str(_ROOT / "results" / "constraint_stats.json"))
    args = parser.parse_args(argv)

    report = analyse(args.catalog, args.dataset)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(
        f"{summary['constraints']} disclosable constraint strings "
        f"across {report['sessions']} sessions, against {report['catalog_size']:,} products\n"
    )
    print(
        f"  appear verbatim in their own target      "
        f"{summary['pct_verbatim_in_target']:>5.1f}%   "
        f"({summary['verbatim_in_target']}/{summary['constraints']})"
    )
    print(
        f"  unique to one product in the catalog     "
        f"{summary['pct_unique_in_catalog']:>5.1f}%   "
        f"({summary['unique_in_catalog']}/{summary['constraints']})"
    )
    print(
        f"  narrow the catalog to {report['top_k']} or fewer        "
        f"{summary['pct_narrows_to_top_k_or_fewer']:>5.1f}%   "
        f"({summary['narrows_to_top_k_or_fewer']}/{summary['constraints']})\n"
    )
    for origin in ("mined", "synthesised"):
        row = report["by_origin"][origin]
        print(
            f"  {origin:<12} n={row['constraints']:<4} "
            f"verbatim {row['pct_verbatim_in_target']:>5.1f}%  "
            f"unique {row['pct_unique_in_catalog']:>5.1f}%  "
            f"narrow {row['pct_narrows_to_top_k_or_fewer']:>5.1f}%"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

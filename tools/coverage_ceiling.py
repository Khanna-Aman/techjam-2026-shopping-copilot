"""Is the residual MRR gap reachable by a better ranker, or is it not there to take?

`tools/diagnose_rank.py` asks whether the target lost to an exact scoring tie. Almost none
do, which reads as good news -- if the target is separable, a better ranker could separate
it. That reading is wrong, and this harness is why.

Exact ties are a float-equality question. The question that decides whether ranking work can
help is a *semantic* one: do the products beating the target satisfy more of what the
customer actually said? Three cases, and they need opposite responses:

``beaten on evidence``
    A rival matches strictly more constraints. The ranking is correct and the target is
    genuinely a worse answer; nothing to fix, and "fixing" it would be fitting the label.

``mis-scored``
    A rival matches strictly fewer constraints and still outranks the target. This is a real
    defect -- popularity or lexical noise overpowering stated evidence -- and a reranker,
    different weights or a coverage-dominant ordering would repair it.

``saturated tie``
    The rival matches exactly as many constraints as the target. The customer has not said
    anything that distinguishes them. No reranker can fix this from the dialogue, because
    the information is not in the dialogue. The only way to separate them is a prior over
    which product is likelier a priori -- which is what `w_popularity` is -- or a guess.

The distinction matters because the three look identical in the metric and only one of them
is worth engineering against. If the losses are saturated ties, then the gap between the
current score and a perfect ranker is not headroom; it is the task refusing to be more
specific, and any weight that closes it on 200 sessions is fitting which particular parka
the dataset happened to label.

To show the measurement is not vacuous -- the candidate pool is filtered, so "everything
above matches all constraints" could be true by construction -- the report also carries the
coverage histogram of the whole pool at the deciding turn. When only 4.6% of a 681-product
pool matches every constraint and all eight products above the target are inside that 4.6%,
the ranker is demonstrably promoting the right ones and has simply run out of evidence.

Diagnostic only. Produces no score.

Usage:
    python -m tools.coverage_ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.catalog import CatalogIndex  # noqa: E402
from copilot.config import DEFAULT_CONFIG  # noqa: E402
from copilot.retrieval import _score_pool  # noqa: E402


def _coverage(products: dict, asin: str, constraints: list[str]) -> int:
    """How many disclosed constraint strings appear in the product's own text.

    Deliberately the crude surface test rather than the ranker's weighted
    `constraint_score`: the question is what a customer would say the product matches, not
    what our scoring function believes, and reusing the scorer here would make the finding
    circular.
    """
    product = products.get(asin, {})
    blob = " ".join(
        [
            product.get("title", ""),
            " ".join(product.get("features", []) or []),
            " ".join(product.get("description", []) or []),
        ]
    ).lower()
    return sum(1 for text in constraints if text in blob)


def analyse(catalog: str, dataset: str) -> dict:
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    index = CatalogIndex(catalog)
    agent = ShoppingCopilot(catalog, config=DEFAULT_CONFIG, index=index)

    records: list[dict] = []
    for sample in samples:
        session_id = f"cov_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            decided = (override_applied and target in ranked) or turn == MAX_TURNS
            if decided:
                state = agent._state(session_id)
                scores = _score_pool(index, state, agent.config, dense=agent.dense)
                target_doc = index.id_to_doc.get(target)
                if target_doc is None or target_doc not in scores:
                    break
                target_score = scores[target_doc]
                constraints = [item.text for item in state.active_constraints]
                target_cover = _coverage(products, target, constraints)
                above = [doc for doc, value in scores.items() if value > target_score]
                covers = [_coverage(products, index.ids[doc], constraints) for doc in above]
                histogram = Counter(
                    _coverage(products, index.ids[doc], constraints) for doc in scores
                )
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "scenario_type": sample["scenario_type"],
                        "turn": turn,
                        "rank": (ranked.index(target) + 1) if target in ranked else None,
                        "constraints_known": len(constraints),
                        "target_coverage": target_cover,
                        "pool_size": len(scores),
                        "pool_at_target_coverage": histogram[target_cover],
                        "above": len(above),
                        "above_higher_coverage": sum(1 for c in covers if c > target_cover),
                        "above_equal_coverage": sum(1 for c in covers if c == target_cover),
                        "above_lower_coverage": sum(1 for c in covers if c < target_cover),
                    }
                )
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get("message", "Actually, ignore my earlier preference.")
                )
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

    imperfect = [r for r in records if r["rank"] != 1]
    summary = {
        "sessions": len(records),
        "rank_1": sum(1 for r in records if r["rank"] == 1),
        "not_rank_1": len(imperfect),
        "rivals_above_targets": sum(r["above"] for r in imperfect),
        "rivals_beating_on_evidence": sum(r["above_higher_coverage"] for r in imperfect),
        "rivals_in_saturated_ties": sum(r["above_equal_coverage"] for r in imperfect),
        "rivals_mis_scored": sum(r["above_lower_coverage"] for r in imperfect),
    }
    return {"sessions": records, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--out", default=str(_ROOT / "results" / "coverage_ceiling.json"))
    args = parser.parse_args(argv)

    report = analyse(args.catalog, args.dataset)
    summary = report["summary"]
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    total = summary["rivals_above_targets"]
    print(
        f"sessions {summary['sessions']} | rank 1: {summary['rank_1']} | "
        f"not rank 1: {summary['not_rank_1']}\n"
    )
    print(f"products ranked above a target, across the imperfect sessions: {total}")
    print(
        f"  beating it on evidence (more constraints)  "
        f"{summary['rivals_beating_on_evidence']:>4}   correctly ranked"
    )
    print(
        f"  saturated ties        (same constraints)   "
        f"{summary['rivals_in_saturated_ties']:>4}   no ranker can fix these"
    )
    print(
        f"  mis-scored            (fewer constraints)  "
        f"{summary['rivals_mis_scored']:>4}   a better ranker could"
    )
    if total:
        share = 100.0 * summary["rivals_in_saturated_ties"] / total
        print(f"\n{share:.0f}% of the residual is saturated ties.")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

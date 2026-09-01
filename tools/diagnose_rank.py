"""Why is the target not rank 1 at the moment it first appears?

MRR is the only headroom left -- Hit@10 is saturated at 1.000 and MTTC is near its
structural floor -- and 13 of 200 sessions land somewhere other than rank 1 (77 before the
confidence gate; pass --base '{"use_confidence_gate": false}' for that figure). Before
building anything to fix that, it is worth knowing what "that" is, because two very
different failures produce the same symptom and they need opposite fixes:

``ties``
    Several products score identically or near-identically because the constraints
    disclosed so far genuinely do not separate them. No reranker can fix this; the
    information is not present. The fix is to ask another question, or to not show a
    ranking you cannot justify.

``mis-scoring``
    The target is separable on what is already known, and the scoring function simply puts
    something else first. This is the case a better ranker -- a cross-encoder, different
    weights, more features -- could actually repair.

The evaluator stops the session the moment the target enters the top 10, so the rank it
lands at is locked in permanently. That makes this the single decisive instant in each
session, and this harness reconstructs it: at the exact turn of first appearance it records
the target's score, the scores above it, how many constraints were known, and how big the
surviving candidate pool was.

Diagnostic only. Produces no score.

Usage:
    python -m tools.diagnose_rank
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from dataclasses import replace
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
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402
from copilot.retrieval import candidate_pool  # noqa: E402

#: Scores within this fraction of the target's are treated as ties rather than as genuine
#: separations. The combined score is a sum of weighted terms of order 1, so a gap this
#: small means the ranking between them is decided by the id tiebreak, not by evidence.
_TIE_EPSILON = 1e-9


def _scored_pool(agent: ShoppingCopilot, state) -> dict[int, float]:
    """Re-score the live pool with the agent's own configuration."""
    from copilot.retrieval import _score_pool

    return _score_pool(agent.index, state, agent.config, dense=agent.dense)


def analyse(catalog: str, dataset: str, overrides: dict | None = None) -> dict:
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    index = CatalogIndex(catalog)
    config = DEFAULT_CONFIG
    if overrides:
        allowed = set(AgentConfig.__dataclass_fields__)
        config = replace(config, **{k: v for k, v in overrides.items() if k in allowed})
    agent = ShoppingCopilot(catalog, config=config, index=index)

    records: list[dict] = []
    for sample in samples:
        session_id = f"diag_{uuid.uuid4().hex}"
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

            if override_applied and target in ranked:
                state = agent._state(session_id)
                scores = _scored_pool(agent, state)
                target_doc = index.id_to_doc.get(target)
                target_score = scores.get(target_doc, 0.0)
                above = [s for s in scores.values() if s > target_score + _TIE_EPSILON]
                tied = [s for s in scores.values() if abs(s - target_score) <= _TIE_EPSILON]
                records.append({
                    "sample_id": sample["sample_id"],
                    "scenario_type": sample["scenario_type"],
                    "turn": turn,
                    "rank": ranked.index(target) + 1,
                    "constraints_known": len(state.active_constraints),
                    "pool_size": len(candidate_pool(index, state, agent.config)),
                    "strictly_above": len(above),
                    "tied_with_target": len(tied) - 1,
                    "score_gap_to_top": round(
                        (max(scores.values()) - target_score) if scores else 0.0, 6
                    ),
                })
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

    return {"sessions": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--out", default=str(_ROOT / "results" / "rank_diagnosis.json"))
    parser.add_argument(
        "--base", default=None,
        help=(
            "JSON object of AgentConfig overrides. The finding that motivated the "
            "confidence gate is a statement about the agent *before* the gate existed, so "
            "reproducing it needs --base '{\"use_confidence_gate\": false}'. Without this "
            "flag the command reports the post-gate residual instead, which is a different "
            "and much smaller number."
        ),
    )
    args = parser.parse_args(argv)

    report = analyse(args.catalog, args.dataset, json.loads(args.base) if args.base else None)
    records = report["sessions"]
    missed = [r for r in records if r["rank"] > 1]

    tied_at_top = [r for r in missed if r["tied_with_target"] > 0 and r["strictly_above"] == 0]
    genuinely_beaten = [r for r in missed if r["strictly_above"] > 0]

    summary = {
        "hits": len(records),
        "rank_1": len(records) - len(missed),
        "not_rank_1": len(missed),
        "lost_to_ties": len(tied_at_top),
        "lost_to_scoring": len(genuinely_beaten),
        "constraints_known_when_not_rank_1": dict(
            Counter(r["constraints_known"] for r in missed)
        ),
        "constraints_known_when_rank_1": dict(
            Counter(r["constraints_known"] for r in records if r["rank"] == 1)
        ),
    }
    report["summary"] = summary
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"hits {summary['hits']} | rank 1: {summary['rank_1']} | not rank 1: {len(missed)}\n")
    print(f"  lost to exact ties      {len(tied_at_top):>3}   no ranker can fix these")
    print(f"  lost to scoring         {len(genuinely_beaten):>3}   a better ranker could")
    if genuinely_beaten:
        gaps = sorted(r["score_gap_to_top"] for r in genuinely_beaten)
        print(f"     median score gap to the top: {gaps[len(gaps)//2]:.4f}")
        print(f"     median products strictly above: "
              f"{sorted(r['strictly_above'] for r in genuinely_beaten)[len(genuinely_beaten)//2]}")
    print("\nconstraints known at the deciding turn:")
    print(f"  when rank 1     {summary['constraints_known_when_rank_1']}")
    print(f"  when not rank 1 {summary['constraints_known_when_not_rank_1']}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

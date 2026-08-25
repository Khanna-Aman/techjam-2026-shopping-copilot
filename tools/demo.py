"""Annotated single-session trace: watch the agent think.

Prints one full conversation turn by turn, showing the customer's message, what the agent
extracted from it, what it decided to ask next, and how the hidden target moved through
the ranking as evidence accumulated.

The target's identity and rank are printed by the *harness*, not known to the agent. The
agent receives only the anonymised profile and the message text, exactly as in scoring.

Usage:
    python -m tools.demo                              # a representative Buying session
    python -m tools.demo --scenario intent_override   # the hardest scenario
    python -m tools.demo --scenario browsing --index 3
    python -m tools.demo --list
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Product titles contain non-ASCII characters; Windows consoles default to cp1252.
try:  # pragma: no cover - console-dependent
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

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

BAR = "=" * 78
RULE = "-" * 78


def _fmt_constraints(state) -> str:
    if not state.active_constraints:
        return "(none yet)"
    parts = []
    for item in state.active_constraints:
        text = item.text if len(item.text) <= 46 else item.text[:43] + "..."
        marker = f" x{item.weight:g}" if item.weight != 1.0 else ""
        parts.append(f"[{item.kind}{marker}] {text}")
    return "\n                 ".join(parts)


def run_demo(sample: dict, agent: ShoppingCopilot, catalog_ids, categories, products) -> None:
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    profile = sample["user_profile"]
    print(BAR)
    print(f" SESSION {sample['sample_id']}   scenario: {sample['scenario_type']}")
    print(BAR)
    print(f" hidden target : {target}  {str(products[target].get('title'))[:52]}")
    print(f" profile       : {str(profile.get('summary', ''))[:70]}")
    print(" (the target is shown by the harness for illustration; the agent never sees it)")
    print()

    agent.reset(session_id, profile)
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f" TURN {turn}")
        print(f"   customer   > {message}")
        response = agent.respond(session_id, message, turn, TOP_K)
        state = agent._sessions[session_id]

        pool = len(agent.index.bucket(state.category)) if state.category else 0
        print(f"   parsed     > category={state.category!r}"
              f"  ({pool} in-category candidates)")
        print(f"   memory     > {_fmt_constraints(state)}")
        print(f"   agent      > \"{response['message']}\"")
        print(f"   asks       > ask_attribute={response['ask_attribute']!r}")

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        for position, asin in enumerate(ranked[:3], start=1):
            flag = "  <-- TARGET" if asin == target else ""
            title = str(products[asin].get("title") or "")[:44]
            print(f"     {position}. {asin}  {title}{flag}")

        if target in ranked:
            rank_of = ranked.index(target) + 1
            if override_applied:
                print(RULE)
                print(f" HIT at turn {turn}, rank {rank_of}   "
                      f"(reciprocal rank {1 / rank_of:.3f})")
                print(BAR)
                return
            print(f"   note       > target is at rank {rank_of}, but this is an Intent")
            print("                Override session and the customer has not yet")
            print("                revised their intent, so a hit here does not count.")
        else:
            print("   target     > not in top 10 yet")

        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
            print("   ** the customer is about to change their mind **")
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )
        print()

    print(RULE)
    print(" MISS: target never reached the top 10 within 10 turns.")
    print(BAR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotated single-session demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--scenario", default="buying",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--index", type=int, default=0, help="which session of that scenario")
    parser.add_argument("--list", action="store_true", help="list available sessions and exit")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.list:
        for scenario in ("buying", "browsing", "intent_override", "boundary"):
            count = sum(1 for s in samples if s["scenario_type"] == scenario)
            print(f"  {scenario:16s} {count} sessions")
        return

    pool = [s for s in samples if s["scenario_type"] == args.scenario]
    if not pool:
        raise SystemExit(f"no sessions with scenario {args.scenario!r}")
    random.Random(args.seed).shuffle(pool)
    sample = pool[args.index % len(pool)]

    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = ShoppingCopilot(args.catalog, index=CatalogIndex(args.catalog))
    run_demo(sample, agent, catalog_ids, categories, products)


if __name__ == "__main__":
    main()

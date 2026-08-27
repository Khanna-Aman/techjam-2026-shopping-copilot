"""Adversarial robustness harness: does the agent survive reworded customer messages?

The competition specification warns that the organiser may add natural-language
paraphrasing to the simulator, noting only that paraphrasing "cannot decide correctness".
Any agent that parses the message templates literally is therefore exposed: if the private
harness rewords its prompts, template matching silently stops firing and the agent falls
back to guessing.

This harness re-runs the official evaluation loop with a perturbation applied to every
customer message. It imports the evaluator's own simulator functions rather than
reimplementing them, so the customer policy, scoring and scenario mix stay identical --
only the surface wording changes. The official evaluator file is never modified.

Scores produced here are DIAGNOSTIC ONLY and are not the official metric.

Usage:
    python -m tools.robustness                 # all perturbations
    python -m tools.robustness --only heavy    # a single perturbation
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import uuid
from collections import defaultdict
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
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

# --------------------------------------------------------------------- perturbations
_OPEN_BUY = re.compile(r"^I'm looking for (.+?)\. A key requirement is: (.+)\.$")
_OPEN_BROWSE = re.compile(r"^I'm looking for (.+?), but I'm still exploring\.$")
_OPEN_OTHER = re.compile(r"^I'm looking for (.+?)\. (.+)$")
_DISCLOSE = re.compile(r"^For that, what matters is: (.+?)\.$")
_NO_ADD = re.compile(r"^I don't have an additional preference for (.+?)\.$")
_NO_PREF = re.compile(r"^I don't have a preference for (.+?); please use your judgment\.$")
_ASK_MORE = re.compile(r"^Those options are not quite right yet\. Ask me about one specific attribute\.$")
_OVERRIDE = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$")


def identity(message: str, rng: random.Random) -> str:
    return message


def casing(message: str, rng: random.Random) -> str:
    """Lowercase everything: mimics a casual typist, breaks case-sensitive matching."""
    return message.lower()


def punctuation(message: str, rng: random.Random) -> str:
    """Drop terminal punctuation and collapse separators."""
    out = message.replace(";", ",").replace(":", "")
    return out.rstrip(". ")


def _pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def paraphrase_light(message: str, rng: random.Random) -> str:
    """Reword the template chrome while preserving every content string verbatim."""
    match = _OPEN_BUY.match(message)
    if match:
        cat, con = match.groups()
        return _pick(rng, [
            f"I need {cat}. It has to be: {con}.",
            f"Hi, I want {cat}. One thing I really need: {con}.",
            f"Shopping for {cat}. Must-have: {con}.",
        ])
    match = _OPEN_BROWSE.match(message)
    if match:
        cat = match.group(1)
        return _pick(rng, [
            f"Just browsing {cat} for now.",
            f"I'm after {cat}, though I haven't decided yet.",
            f"Show me some {cat} - still making up my mind.",
        ])
    match = _DISCLOSE.match(message)
    if match:
        body = match.group(1)
        return _pick(rng, [
            f"What I care about there: {body}.",
            f"Mainly this: {body}.",
            f"Well, {body}.",
        ])
    match = _NO_ADD.match(message)
    if match:
        attr = match.group(1)
        return _pick(rng, [
            f"No strong feelings about {attr}.",
            f"Nothing particular on {attr}.",
        ])
    match = _NO_PREF.match(message)
    if match:
        attr = match.group(1)
        return _pick(rng, [
            f"Up to you on {attr}.",
            f"No preference on {attr} - your call.",
        ])
    if _ASK_MORE.match(message):
        return _pick(rng, [
            "Not quite there. Ask me something specific.",
            "Hmm, none of those. What do you want to know?",
        ])
    match = _OVERRIDE.match(message)
    if match:
        new = match.group(1)
        return _pick(rng, [
            f"Scratch that - what I actually need is {new}.",
            f"Forget what I said. I really need: {new}.",
        ])
    match = _OPEN_OTHER.match(message)
    if match:
        cat, rest = match.groups()
        return _pick(rng, [f"I want {cat}. {rest}", f"Looking at {cat}. {rest}"])
    return message


_FILLER = ["um", "so", "well", "honestly", "I guess", "you know"]


def paraphrase_heavy(message: str, rng: random.Random) -> str:
    """Light paraphrase plus filler, lowercase drift and dropped punctuation."""
    out = paraphrase_light(message, rng)
    if rng.random() < 0.6:
        out = f"{_pick(rng, _FILLER)}, {out[0].lower()}{out[1:]}"
    if rng.random() < 0.5:
        out = out.replace(";", ",")
    if rng.random() < 0.4:
        out = out.rstrip(".")
    if rng.random() < 0.3:
        out = out.lower()
    return out


PERTURBATIONS = {
    "control": identity,
    "casing": casing,
    "punctuation": punctuation,
    "light": paraphrase_light,
    "heavy": paraphrase_heavy,
}


# ------------------------------------------------------------------------- evaluation
def run(agent: Agent, samples: list[dict], catalog_ids, categories, products,
        perturb, seed: int = 20260825, keep_sessions: bool = False) -> dict:
    """Mirror of the official loop with a perturbation applied to customer messages.

    `keep_sessions` adds the per-session records to the result. It is off by default so the
    committed robustness and ablation JSON stay summaries; `tools/ablation_ci.py` turns it
    on because a paired comparison needs to line the two runs up session by session.
    """
    rng = random.Random(seed)
    sessions: list[dict] = []
    for sample in samples:
        session_id = f"robust_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        raw = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        user_message = perturb(raw, rng)
        hit_turn = None
        best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                raw = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                raw, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
            user_message = perturb(raw, rng)
        sessions.append({
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in sessions:
        grouped[item["scenario_type"]].append(item)
    result = {
        **overall,
        "efficiency": round(efficiency, 6),
        "technical_score": round(score, 6),
        "by_scenario": {
            name: metric_summary(grouped[name])["hit_rate_at_10"] for name in sorted(grouped)
        },
    }
    if keep_sessions:
        result["sessions"] = sessions
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Message-perturbation robustness harness")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--only", default=None, help="run a single perturbation by name")
    parser.add_argument("--output", default="runs/robustness.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    names = [args.only] if args.only else list(PERTURBATIONS)
    results: dict[str, dict] = {}
    print(f"{'perturbation':<14}{'score':>9}{'hit@10':>9}{'MRR':>9}{'MTTC':>8}   by-scenario hit rate")
    print("-" * 96)
    for name in names:
        perturb = PERTURBATIONS[name]
        outcome = run(agent, samples, catalog_ids, categories, products, perturb)
        results[name] = outcome
        by = "  ".join(f"{k[:4]}={v:.2f}" for k, v in outcome["by_scenario"].items())
        print(f"{name:<14}{outcome['technical_score']:>9.4f}{outcome['hit_rate_at_10']:>9.3f}"
              f"{outcome['mrr']:>9.4f}{outcome['mttc']:>8.2f}   {by}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

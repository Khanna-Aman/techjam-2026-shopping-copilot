"""Shop the real catalog yourself, in a terminal.

Every number in this repository comes from a simulator. That is the right way to *measure*
the agent, and it is a poor way to judge whether it behaves sensibly, because a simulated
customer only ever says the eight things the simulator knows how to say. This lets a person
type anything at all and watch what the agent does with it.

It is a diagnostic, not a product. UI development is out of scope for the challenge, and
this is a terminal loop over the same `ShoppingCopilot` the evaluator scores -- no special
casing, no demo mode, no rehearsed inputs. What it shows on every turn is the machinery the
README describes: which category got locked, what went into slot memory, what the agent
decided to ask next and why that question rather than another.

Usage:
    python -m tools.chat
    python -m tools.chat --top 5 --config '{"use_dense_rerank": true}'

Type /help inside the session for the commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Product titles carry non-ASCII characters; Windows consoles default to cp1252.
try:  # pragma: no cover - console-dependent
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402
from copilot.question import (  # noqa: E402
    OPEN_ATTRIBUTE,
    PRODUCTIVE_ATTRIBUTES,
    YIELD_PROBABILITY,
    discriminative_power,
    expected_gain,
    open_gain,
)
from copilot.retrieval import candidate_pool  # noqa: E402

BAR = "=" * 78
RULE = "-" * 78

BANNER = """\
Shopping Copilot -- interactive session

Say what you are looking for, the way you would say it to a person. The agent answers with
a question and a ranked list at the same time; answering the question is what moves it.

  /state     what the agent currently believes
  /why       why it asked what it asked
  /reset     start a new session
  /quit      leave
"""


def _constraints(state) -> str:
    if not state.active_constraints:
        return "   (nothing disclosed yet)"
    lines = []
    for item in state.active_constraints:
        text = item.text if len(item.text) <= 58 else item.text[:55] + "..."
        marker = f" x{item.weight:g}" if item.weight != 1.0 else ""
        lines.append(f"   [{item.kind}{marker}] {text}")
    return "\n".join(lines)


def _show_state(agent: ShoppingCopilot, state) -> None:
    pool = len(agent.index.bucket(state.category)) if state.category else agent.index.count
    print(RULE)
    print(f" category locked : {state.category!r}  ({pool:,} candidates of {agent.index.count:,})")
    print(f" scenario read as: {state.scenario}")
    print(" slot memory     :")
    print(_constraints(state))
    if state.asked:
        print(f" already asked   : {', '.join(state.asked)}")
    if state.exhausted:
        print(f" spent           : {', '.join(sorted(state.exhausted))}")
    print(RULE)


def _show_why(agent: ShoppingCopilot, state) -> None:
    """Score every question the policy could ask, on the one scale it compares them on.

    Worth watching rather than taking on trust: nothing hardcodes "ask the open question
    first". It wins on the numbers because it is answered whenever anything remains
    undisclosed and returns two constraints at once, and the agent moves to a specific
    attribute the moment that stops being true.
    """
    pool = candidate_pool(agent.index, state, agent.config)
    scored = [
        (attribute, expected_gain(agent.index, state, pool, attribute))
        for attribute in PRODUCTIVE_ATTRIBUTES
    ]
    scored.append((OPEN_ATTRIBUTE + " (open)", open_gain(state, pool)))

    print(RULE)
    print(f" pool of {len(pool):,} candidates")
    print(f" {'question':<16}{'value':>9}{'splits pool':>13}   P(answered)")
    for attribute, value in sorted(scored, key=lambda pair: -pair[1]):
        bare = attribute.split(" ")[0]
        power = "" if bare == OPEN_ATTRIBUTE else f"{discriminative_power(agent.index, pool, bare):>13.3f}"
        yielded = YIELD_PROBABILITY.get(bare)
        chance = "   (any type)" if yielded is None else f"   {yielded:.3f}"
        spent = "  spent" if bare in state.exhausted else ""
        print(f" {attribute:<16}{value:>9.4f}{power or '':>13}{chance}{spent}")
    print(RULE)
    print(" size is the illustration: it splits the pool better than colour and is worth")
    print(" far less, because colour gets answered and size does not. Value is a product,")
    print(" not entropy alone -- budget, at the bottom, is the same effect at its extreme.")
    print(RULE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive shopping session")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--config", default=None, help="JSON object of AgentConfig overrides")
    args = parser.parse_args()

    if not Path(args.catalog).exists():
        raise SystemExit(
            f"{args.catalog} not found. See the README for the one-command download."
        )

    config: AgentConfig = DEFAULT_CONFIG
    if args.config:
        overrides = json.loads(args.config)
        allowed = set(AgentConfig.__dataclass_fields__)
        config = replace(config, **{k: v for k, v in overrides.items() if k in allowed})

    # Printed before the load rather than conditionally on a cache miss, so the wording has
    # to be true either way: warm this returns in about 0.3 s. It said "about 25 s the first
    # time", a figure from an earlier build implementation that no document has agreed with
    # since -- README and DEVPOST both quote ~12 s cold against a committed latency artifact.
    # It is a small thing that happens to be on screen during the one segment the demo script
    # says never to cut, announcing a wait the viewer then does not see.
    print("loading the index (cached; about 12 s the very first time)...")
    agent = ShoppingCopilot(args.catalog, config=config)
    print(f"{agent.index.count:,} products across {len(agent.index.buckets):,} categories\n")
    print(BANNER)

    session_id = f"chat_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, {"preference_tags": ["fit", "comfort"]})
    turn = 0

    while True:
        try:
            message = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue

        lowered = message.lower()
        if lowered in ("/quit", "/exit", "/q"):
            return
        if lowered == "/help":
            print(BANNER)
            continue
        if lowered == "/reset":
            session_id = f"chat_{uuid.uuid4().hex[:8]}"
            agent.reset(session_id, {"preference_tags": ["fit", "comfort"]})
            turn = 0
            print("new session.\n")
            continue
        if lowered == "/state":
            _show_state(agent, agent._sessions[session_id])
            continue
        if lowered == "/why":
            _show_why(agent, agent._sessions[session_id])
            continue

        turn += 1
        response = agent.respond(session_id, message, turn, args.top)
        state = agent._sessions[session_id]

        print()
        print(f"agent > {response['message']}")
        if response["ask_attribute"]:
            print(f"        (asking about: {response['ask_attribute']})")
        print()
        for position, item in enumerate(response["recommendations"][:args.top], start=1):
            asin = item["parent_asin"]
            title = agent.index.titles[agent.index.id_to_doc[asin]]
            print(f"  {position}. {title[:66]}")
        pool = len(agent.index.bucket(state.category)) if state.category else agent.index.count
        print(f"\n        turn {turn} | {pool:,} candidates | "
              f"{len(state.active_constraints)} constraint(s) known\n")


if __name__ == "__main__":
    main()

"""Configuration sweep and ablation harness.

Runs the official scoring loop repeatedly against different agent configurations while
sharing a single catalog index, so a full 200-session evaluation costs a few seconds
instead of a rebuild each time.

Two modes:

* ``--ablation`` disables one mechanism at a time to measure what each contributes. This
  is the honest way to report which parts of the system actually earn their place.
* ``--sweep`` walks a weight grid to tune ranking parameters.

The official evaluator is never modified; configurations are injected through the agent's
own config object. The identity perturbation reproduces the official score exactly, which
is asserted by ``--verify``.
"""

from __future__ import annotations

import argparse
import json
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
from tools.robustness import identity, run  # noqa: E402

#: One mechanism disabled per row. Ordered roughly by expected impact.
ABLATIONS: list[tuple[str, dict]] = [
    ("full system", {}),
    ("no clarification", {"clarify_strategy": "none"}),
    ("no category lock", {"use_category_lock": False}),
    ("no constraint scoring", {"use_constraint_scoring": False}),
    ("no state tracking", {"use_state_tracking": False}),
    ("no override erasure", {"use_override_erasure": False}),
    ("no observed fallback", {"use_observed_fallback": False}),
    ("no top-10 padding", {"pad_to_top_k": False}),
    ("no MMR diversity", {"use_mmr_diversity": False}),
    ("no popularity prior", {"use_popularity_prior": False}),
    ("no profile prior", {"use_profile_prior": False}),
]

#: Clarification strategy comparison.
STRATEGIES: list[tuple[str, dict]] = [
    ("strategy: none", {"clarify_strategy": "none"}),
    ("strategy: open", {"clarify_strategy": "open"}),
    ("strategy: infogain", {"clarify_strategy": "infogain"}),
    ("strategy: hybrid", {"clarify_strategy": "hybrid"}),
]


def _grid() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for w_constraint in (1.8, 2.6, 3.4, 4.5, 6.0):
        for boost in (1.6, 2.2, 3.0):
            out.append(
                (
                    f"w_con={w_constraint} boost={boost}",
                    {"w_constraint": w_constraint, "constraint_term_boost": boost},
                )
            )
    return out


def _observed_grid() -> list[tuple[str, dict]]:
    return [
        (f"observed={value}", {"observed_term_boost": value})
        for value in (0.0, 0.25, 0.40, 0.55, 0.75, 1.0)
    ]


def _prior_grid() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for pop in (0.0, 0.08, 0.18, 0.30):
        for prof in (0.0, 0.12, 0.25):
            out.append((f"pop={pop} prof={prof}", {"w_popularity": pop, "w_profile": prof}))
    return out


def _profile_grid() -> list[tuple[str, dict]]:
    rows = [("profile off", {"use_profile_prior": False})]
    for value in (0.6, 1.0, 1.4, 2.0, 3.0):
        rows.append((f"cold-start w={value}", {"w_profile": value, "profile_cold_start_only": True}))
    return rows


def _pop_grid() -> list[tuple[str, dict]]:
    return [
        (f"w_pop={value}", {"w_popularity": value, "w_profile": 0.0})
        for value in (0.18, 0.30, 0.40, 0.55, 0.70, 0.90, 1.20, 1.60)
    ]


def _wcon_grid() -> list[tuple[str, dict]]:
    return [
        (f"w_con={value}", {"w_constraint": value})
        for value in (0.0, 0.25, 0.5, 0.9, 1.4, 2.6)
    ]


def _override_grid() -> list[tuple[str, dict]]:
    return [
        (f"override_decay={value}", {"override_decay": value})
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]


MODES = {
    "ablation": lambda: ABLATIONS,
    "override": _override_grid,
    "pop": _pop_grid,
    "profile": _profile_grid,
    "wcon": _wcon_grid,
    "strategy": lambda: STRATEGIES,
    "grid": _grid,
    "observed": _observed_grid,
    "prior": _prior_grid,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Configuration sweep / ablation harness")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--mode", default="ablation", choices=sorted(MODES))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--base", default=None,
        help="JSON object of AgentConfig overrides applied to every row",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    index = CatalogIndex(args.catalog)

    base: AgentConfig = DEFAULT_CONFIG
    if args.base:
        overrides = json.loads(args.base)
        allowed = set(AgentConfig.__dataclass_fields__)
        base = replace(base, **{k: v for k, v in overrides.items() if k in allowed})

    rows = MODES[args.mode]()
    results: dict[str, dict] = {}
    reference: float | None = None

    print(f"{'configuration':<26}{'score':>9}{'hit@10':>9}{'MRR':>9}{'MTTC':>8}{'delta':>9}")
    print("-" * 70)
    for label, overrides in rows:
        config = replace(base, **overrides)
        agent = ShoppingCopilot(args.catalog, config=config, index=index)
        outcome = run(agent, samples, catalog_ids, categories, products, identity)
        results[label] = {"overrides": overrides, **outcome}
        score = outcome["technical_score"]
        if reference is None:
            reference = score
        delta = score - reference
        print(
            f"{label:<26}{score:>9.4f}{outcome['hit_rate_at_10']:>9.3f}"
            f"{outcome['mrr']:>9.4f}{outcome['mttc']:>8.2f}{delta:>+9.4f}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

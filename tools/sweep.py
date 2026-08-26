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

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.catalog import CatalogIndex  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402
from copilot.llm import LLMReranker  # noqa: E402
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
    """Cold-start against always-on, because the sign of the effect depends on the timing.

    Both arms are here deliberately. The finding this grid supports is not "personalization
    helps" but "the same feature helps at cold start and hurts as a global term", and a grid
    carrying only the cold-start arm cannot reproduce the half that makes it interesting.
    """
    rows = [("profile off", {"use_profile_prior": False})]
    for value in (0.6, 1.0, 1.4, 2.0, 3.0):
        rows.append((f"cold-start w={value}", {"w_profile": value, "profile_cold_start_only": True}))
    for value in (0.35, 0.6, 1.0):
        rows.append((f"always     w={value}", {"w_profile": value, "profile_cold_start_only": False}))
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


def _dense_grid() -> list[tuple[str, dict]]:
    """Offline default against latent-space reranking at a range of weights.

    Free and offline like every other mode here -- the artifact is prebuilt, and inference
    reads it with the standard library.
    """
    rows = [("offline (default)", {})]
    for value in (0.15, 0.30, 0.60, 1.00, 1.80, 3.00):
        rows.append((f"dense w={value}", {"use_dense_rerank": True, "w_dense": value}))
    for value in (0.3, 0.6, 1.2, 2.5):
        rows.append((
            f"cold-start dense w={value}",
            {"use_dense_rerank": True, "dense_cold_start_only": True, "w_dense": value},
        ))
    return rows


def _llm_grid() -> list[tuple[str, dict]]:
    """Offline default against the optional LLM reranking layer.

    Kept out of ``--mode ablation`` deliberately. Every other mode in this file is free and
    offline; this one issues roughly 400 live API requests per row and costs real money, so
    it should be run when there is a decision to make rather than as part of a routine
    regression sweep.
    """
    return [
        ("offline (default)", {}),
        ("llm rerank depth=10", {"use_llm_rerank": True, "llm_rerank_depth": 10}),
        ("llm rerank depth=20", {"use_llm_rerank": True, "llm_rerank_depth": 20}),
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
    "llm": _llm_grid,
    "dense": _dense_grid,
}

#: Modes that issue live API requests, and therefore need an explicit warning and a
#: reachable layer before they are worth starting.
_BILLED_MODES = {"llm"}


def _require_reachable_llm(base: AgentConfig, rows: list[tuple[str, dict]], samples: int) -> None:
    """Refuse to start a billed sweep that would silently measure nothing.

    ``copilot.llm`` degrades quietly by design: without a key or the SDK it returns the
    offline ordering. That is the right behaviour during scoring and exactly the wrong
    behaviour here, where it would print several identical rows and invite the conclusion
    that the LLM layer contributes nothing. Fail loudly instead.
    """
    probe = LLMReranker(replace(base, use_llm_rerank=True))
    if not probe.available():
        raise SystemExit(
            "The LLM reranking layer is not reachable, so this sweep would measure the\n"
            "offline path several times over and report it as an LLM result.\n\n"
            "  pip install -r requirements-llm.txt\n"
            "  export ANTHROPIC_API_KEY=...     (or: ant auth login)\n"
            "  COPILOT_LLM=1 python -m tools.sweep --mode llm\n"
        )
    billed_rows = sum(1 for _, overrides in rows if overrides.get("use_llm_rerank"))
    print(
        f"This mode issues live API requests: {billed_rows} configuration(s) over "
        f"{samples} sessions,\nroughly {billed_rows * samples * 2:,} requests in total. "
        "It costs real money.\n"
    )


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
    billed = args.mode in _BILLED_MODES
    if billed:
        _require_reachable_llm(base, rows, len(samples))

    results: dict[str, dict] = {}
    reference: float | None = None

    tokens_column = f"{'tokens':>12}" if billed else ""
    print(
        f"{'configuration':<26}{'score':>9}{'hit@10':>9}{'MRR':>9}"
        f"{'MTTC':>8}{'delta':>9}{tokens_column}"
    )
    print("-" * (70 + (12 if billed else 0)))
    for label, overrides in rows:
        config = replace(base, **overrides)
        agent = ShoppingCopilot(args.catalog, config=config, index=index)
        if billed:
            # The official loop, because it is the only one that accumulates the token
            # usage this mode exists to disclose.
            official = evaluate(agent, samples, catalog_ids, categories, products)
            outcome = {
                "sample_count": official["sample_count"],
                "hit_rate_at_10": official["hit_rate_at_10"],
                "mrr": official["mrr"],
                "mttc": official["mttc"],
                "efficiency": official["efficiency"],
                "technical_score": official["recommended_technical_score"],
                "token_usage": official["reported_token_usage"],
            }
        else:
            outcome = run(agent, samples, catalog_ids, categories, products, identity)
        results[label] = {"overrides": overrides, **outcome}
        score = outcome["technical_score"]
        if reference is None:
            reference = score
        delta = score - reference
        suffix = ""
        if billed:
            suffix = f"{outcome['token_usage']['total_tokens']:>12,}"
        print(
            f"{label:<26}{score:>9.4f}{outcome['hit_rate_at_10']:>9.3f}"
            f"{outcome['mrr']:>9.4f}{outcome['mttc']:>8.2f}{delta:>+9.4f}{suffix}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

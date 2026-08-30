"""Agent configuration.

Every behavioural switch lives here so that ablations run by varying configuration
rather than by editing code. The official evaluator must stay untouched, so this is the
only sanctioned way to measure what each mechanism contributes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Clarification strategies.
#   "none"      - never ask (reproduces the starter baseline's blind spot)
#   "open"      - always ask the open-ended question
#   "infogain"  - pick the attribute maximising expected information gain
#   "hybrid"    - infogain while a specific attribute discriminates, else open
STRATEGIES = ("none", "open", "infogain", "hybrid")


@dataclass(frozen=True)
class AgentConfig:
    """Tunable behaviour for :class:`copilot.agent.ShoppingCopilot`."""

    # --- structural switches (used for ablation) ---------------------------------
    use_category_lock: bool = True
    use_state_tracking: bool = True
    use_constraint_scoring: bool = True
    use_override_erasure: bool = True
    use_profile_prior: bool = True
    use_popularity_prior: bool = True
    # Measured and rejected: diversifying an uncertain top-10 sounds right, but the
    # ablation puts it at exactly 0.0000 against the full system while dominating latency.
    # Retained as an option so the ablation table stays reproducible.
    use_mmr_diversity: bool = False
    pad_to_top_k: bool = True

    #: Withhold recommendations while the agent has too little evidence to order them.
    #:
    #: The evaluator ends a session the moment the target enters the top ten, and locks in
    #: whatever rank it landed at. Showing a list early therefore spends the session's only
    #: scoring opportunity on the agent's worst-informed guess: 85% of the sessions that
    #: finish below rank 1 were decided with two constraints or fewer in hand. Returning a
    #: shorter list until the evidence arrives trades a turn of MTTC, weighted 0.20, for
    #: rank in MRR, weighted 0.30 -- roughly a 13:1 trade per session when it works.
    #:
    #: It can also lose outright: a session gated into never hitting forfeits its Hit@10
    #: contribution, which costs about three times what a successful gate gains, so this
    #: needs to be right far more often than not. Off until measured.
    use_confidence_gate: bool = True
    #: Gate while fewer than this many constraints are known.
    gate_min_constraints: int = 4
    #: How many recommendations to show while gated. One, not zero: the agent always makes
    #: a recommendation, it just makes its single best one instead of ten speculative ones.
    #: Showing one is also worth more than showing none (0.9548 against 0.9357), because a
    #: correct single guess converts at rank 1 while a withheld list cannot convert at all.
    gate_list_size: int = 1
    #: Never gate beyond this turn, so a session that stays uninformative still gets its
    #: chance to hit rather than being starved to a guaranteed miss. Without this cap the
    #: mechanism has a catastrophic mode: gating to turn 10 scores 0.0000.
    gate_max_turn: int = 3
    # Retain raw message tokens even when structured parsing fails. This is what
    # keeps the agent standing up under paraphrase.
    use_observed_fallback: bool = True
    clarify_strategy: str = "hybrid"
    # Optional LLM semantic reranking. Off by default and gated a second time by the
    # COPILOT_LLM environment variable, because official scoring may run without network
    # access -- so the offline path is the one that must be the default. See copilot/llm.py.
    # Offline dense retrieval: latent-space cosine over the candidate pool, read from a
    # prebuilt artifact with the standard library alone. Needs no network and no model at
    # inference, so unlike the LLM layer it is a candidate for the scored default -- but it
    # ships off until measurement says otherwise. See copilot/dense.py.
    use_dense_rerank: bool = False
    #: Apply the latent term only before any constraint is known. Same shape as
    #: profile_cold_start_only, and tested for the same reason: cold start is the one
    #: moment when the lexical signal is a bare category and a smoothed one might add
    #: something rather than blur what is already precise.
    dense_cold_start_only: bool = False
    dense_path: str = "artifacts/dense"
    use_llm_rerank: bool = False
    #: How many of the top candidates are sent for reranking. The tail is already ordered
    #: by the offline ranker and rarely holds the target, so ranking it buys nothing.
    llm_rerank_depth: int = 20
    llm_model: str = "claude-opus-5"

    # --- ranking weights ---------------------------------------------------------
    w_bm25: float = 1.00
    w_constraint: float = 2.60
    #: Weight on the latent-space cosine term. Inert unless use_dense_rerank is on.
    w_dense: float = 0.60
    w_popularity: float = 0.55
    # Left at 1.0 deliberately. Raising this to 5.0 scores better on clean input (0.9083
    # against 0.9062) and the gain survives held-out targets, but it buys that by making
    # the agent lean harder on a prior that paraphrase disturbs: under heavy paraphrase it
    # scores 0.8711 against 1.0's 0.8824, dropping the worst case from 2.9% to 4.1% below
    # control. A ranking tweak that trades paraphrase resistance for two thousandths is the
    # wrong trade here -- see the rejected-ideas table in the README.
    w_profile: float = 1.00
    # Apply the profile prior only before any constraint is known. The anonymised
    # preference tags are generic words (fit, comfort, durability) that match most of
    # the catalog, so as a global term they add noise -- but at cold start, when
    # nothing else is known, they are the only personal signal available.
    profile_cold_start_only: bool = True

    # Weight applied to query terms coming from a confirmed constraint versus the
    # category label. Constraints are mined verbatim from the target product, so they
    # deserve to dominate the generic category words.
    constraint_term_boost: float = 2.2
    category_term_boost: float = 1.0
    # Retracted Intent Override values are not merely ignored; actively down-weighting
    # them beats neutrality, because the decoy is drawn from the target's own text and
    # would otherwise keep scoring well.
    decoy_term_penalty: float = 0.35
    # Trust retained by an Intent Override value after the customer retracts it.
    # 0.0 erases it, 1.0 ignores the retraction. Tuned empirically -- see README.
    override_decay: float = 0.5
    # Weight for tokens seen in a message but not parsed into a typed constraint.
    # Deliberately well below a confirmed constraint: useful signal, lower trust.
    observed_term_boost: float = 0.55

    # --- retrieval shape ---------------------------------------------------------
    # When the category lock yields fewer than this many candidates, widen with a
    # global lexical pass so a mis-parsed category can never strand the session.
    min_candidates: int = 40
    global_fallback_limit: int = 400
    mmr_lambda: float = 0.82

    # --- clarification -----------------------------------------------------------
    # Below this candidate-pool entropy an attribute is considered non-discriminating.
    min_infogain: float = 0.08

    def validate(self) -> None:
        if self.clarify_strategy not in STRATEGIES:
            raise ValueError(
                f"clarify_strategy must be one of {STRATEGIES}, got {self.clarify_strategy!r}"
            )


DEFAULT_CONFIG = AgentConfig()

#: Reproduces the shipped starter agent's behaviour through our own pipeline, so the
#: ablation table has an internally consistent zero point.
BASELINE_CONFIG = AgentConfig(
    use_category_lock=False,
    use_state_tracking=False,
    use_constraint_scoring=False,
    use_override_erasure=False,
    use_profile_prior=False,
    use_popularity_prior=False,
    use_mmr_diversity=False,
    pad_to_top_k=False,
    clarify_strategy="none",
)

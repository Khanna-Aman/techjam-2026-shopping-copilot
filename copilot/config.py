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
    use_mmr_diversity: bool = True
    pad_to_top_k: bool = True
    # Retain raw message tokens even when structured parsing fails. This is what
    # keeps the agent standing up under paraphrase.
    use_observed_fallback: bool = True
    clarify_strategy: str = "hybrid"

    # --- ranking weights ---------------------------------------------------------
    w_bm25: float = 1.00
    w_constraint: float = 2.60
    w_popularity: float = 0.18
    w_profile: float = 0.12

    # Weight applied to query terms coming from a confirmed constraint versus the
    # category label. Constraints are mined verbatim from the target product, so they
    # deserve to dominate the generic category words.
    constraint_term_boost: float = 2.2
    category_term_boost: float = 1.0
    # Retracted Intent Override values are not merely ignored; actively down-weighting
    # them beats neutrality, because the decoy is drawn from the target's own text and
    # would otherwise keep scoring well.
    decoy_term_penalty: float = 0.35
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

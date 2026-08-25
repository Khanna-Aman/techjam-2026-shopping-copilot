"""Candidate generation, constraint satisfaction, and ranking.

The pipeline is deliberately two-stage:

1. **Candidate generation** narrows the catalog. The category lock does the heavy lifting
   (50,000 -> ~180 median), with a global lexical pass as a safety net so a category that
   fails to resolve can never strand a session at zero candidates.
2. **Scoring** blends four signals: weighted BM25 over the accumulated query, explicit
   constraint satisfaction, a popularity prior, and profile affinity. Constraint
   satisfaction dominates by design, because constraints are mined verbatim from the
   target product and are therefore near-identifying when they are specific.

Constraint satisfaction is typed rather than lexical. A colour is a set-membership test,
a budget is numeric proximity, and a feature sentence is phrase containment. Treating a
180-character feature sentence as a bag of words throws away almost all of its selectivity.
"""

from __future__ import annotations

import math
from array import array

from copilot.catalog import _COLOR_BIT, _COLOR_CANON, _MATERIAL_BIT, CatalogIndex
from copilot.config import AgentConfig
from copilot.slots import BUDGET, COLOR, MATERIAL, Constraint, ConversationState
from copilot.text import MATERIALS, match_text, terms

# Products with no listed price cannot satisfy a budget constraint on their own, but the
# catalog omits price for ~79% of rows, so a hard zero would be far too aggressive.
_MISSING_PRICE_CREDIT = 0.25


def satisfies(index: CatalogIndex, doc_id: int, item: Constraint) -> float:
    """Score in [0, 1] for how well one product meets one constraint."""
    if item.kind == MATERIAL and item.value:
        # Graded, not binary: the simulator names the product's *first* material, so a
        # first-position match is much stronger evidence than a passing mention.
        position = MATERIALS.index(item.value) if item.value in MATERIALS else -1
        if position >= 0 and index.first_material[doc_id] == position:
            return 1.0
        bit = _MATERIAL_BIT.get(item.value, 0)
        return 0.6 if index.material_bits[doc_id] & bit else 0.0
    if item.kind == COLOR and item.value:
        canon = "gray" if item.value == "grey" else item.value
        position = _COLOR_CANON.index(canon) if canon in _COLOR_CANON else -1
        if position >= 0 and index.first_color[doc_id] == position:
            return 1.0
        bit = _COLOR_BIT.get(item.value, 0)
        return 0.6 if index.color_bits[doc_id] & bit else 0.0
    if item.kind == BUDGET and item.value:
        try:
            target = float(item.value)
        except (TypeError, ValueError):
            return _MISSING_PRICE_CREDIT
        price = index.price[doc_id]
        if price is None or price < 0:
            return _MISSING_PRICE_CREDIT
        if target <= 0:
            return _MISSING_PRICE_CREDIT
        relative = abs(price - target) / target
        return max(0.0, 1.0 - relative)

    # Phrase constraint: exact containment first, then partial token credit.
    needle = match_text(item.text)
    if not needle:
        return 0.0
    blob = index.blob[doc_id]
    if needle in blob:
        return 1.0
    tokens = item.tokens
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in blob)
    return 0.7 * (hits / len(tokens))


def constraint_score(
    index: CatalogIndex, doc_id: int, constraints: list[Constraint]
) -> float:
    """Mean satisfaction across active constraints, weighted toward specific ones.

    Longer constraints carry more information, so they get more say. A product that
    matches a distinctive feature sentence is a far better candidate than one that merely
    happens to be black.
    """
    if not constraints:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for item in constraints:
        weight = 1.0 + min(2.0, len(item.tokens) / 6.0)
        total += weight * satisfies(index, doc_id, item)
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


def profile_affinity(index: CatalogIndex, doc_id: int, tags: list[str]) -> float:
    if not tags:
        return 0.0
    blob = index.blob[doc_id]
    hits = sum(1 for tag in tags if tag and tag in blob)
    return hits / len(tags)


def candidate_pool(
    index: CatalogIndex, state: ConversationState, config: AgentConfig
) -> set[int]:
    """Build the working candidate set for this turn."""
    pool: set[int] = set()
    if config.use_category_lock and state.category:
        bucket = index.bucket(state.category)
        pool.update(bucket)

    if len(pool) >= config.min_candidates:
        return pool

    # Widen: lexical pass over the whole catalog so we always have something to rank.
    tokens, weights = state.query_terms(
        constraint_boost=config.constraint_term_boost,
        category_boost=config.category_term_boost,
        decoy_penalty=config.decoy_term_penalty,
    )
    if not tokens:
        tokens = terms(state.category or "")
        weights = {}
    scores = index.bm25(tokens, candidates=None, weights=weights or None)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: config.global_fallback_limit]
    pool.update(doc_id for doc_id, _ in ranked)
    return pool


def _normalise_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / top for key, value in scores.items()}


def rank(
    index: CatalogIndex,
    state: ConversationState,
    config: AgentConfig,
    *,
    limit: int = 10,
) -> list[int]:
    """Produce the ranked candidate list for this turn."""
    pool = candidate_pool(index, state, config)
    if not pool:
        return []

    tokens, weights = state.query_terms(
        constraint_boost=config.constraint_term_boost,
        category_boost=config.category_term_boost,
        decoy_penalty=config.decoy_term_penalty,
    )
    lexical = _normalise_scores(index.bm25(tokens, candidates=pool, weights=weights or None))

    constraints = state.active_constraints if config.use_constraint_scoring else []
    tags = state.profile_tags() if config.use_profile_prior else []

    combined: dict[int, float] = {}
    for doc_id in pool:
        score = config.w_bm25 * lexical.get(doc_id, 0.0)
        if constraints:
            score += config.w_constraint * constraint_score(index, doc_id, constraints)
        if config.use_popularity_prior:
            score += config.w_popularity * index.popularity(doc_id)
        if tags:
            score += config.w_profile * profile_affinity(index, doc_id, tags)
        combined[doc_id] = score

    ordered = sorted(combined.items(), key=lambda kv: (-kv[1], index.ids[kv[0]]))

    if not config.use_mmr_diversity or state.has_hard_signal():
        return [doc_id for doc_id, _ in ordered[:limit]]

    # With no constraints yet (early Browsing turns) the ranking is near-arbitrary, so
    # spend the ten slots covering the space instead of stacking near-duplicates.
    return _mmr(index, ordered[: limit * 6], limit, config.mmr_lambda)


def _title_tokens(index: CatalogIndex, doc_id: int) -> set[str]:
    return set(terms(index.titles[doc_id]))


def _mmr(
    index: CatalogIndex,
    ordered: list[tuple[int, float]],
    limit: int,
    lam: float,
) -> list[int]:
    """Maximal Marginal Relevance over titles, to diversify an uncertain top-10."""
    if not ordered:
        return []
    selected: list[int] = []
    selected_tokens: list[set[str]] = []
    remaining = list(ordered)
    top = max(score for _, score in ordered) or 1.0

    while remaining and len(selected) < limit:
        best_index = 0
        best_value = -math.inf
        for position, (doc_id, score) in enumerate(remaining):
            relevance = score / top
            tokens = _title_tokens(index, doc_id)
            penalty = 0.0
            for other in selected_tokens:
                if not tokens and not other:
                    continue
                union = len(tokens | other) or 1
                penalty = max(penalty, len(tokens & other) / union)
            value = lam * relevance - (1.0 - lam) * penalty
            if value > best_value:
                best_value = value
                best_index = position
        doc_id, _ = remaining.pop(best_index)
        selected.append(doc_id)
        selected_tokens.append(_title_tokens(index, doc_id))
    return selected


def pad(index: CatalogIndex, ranked: list[int], pool: set[int], limit: int) -> list[int]:
    """Top up a short list with popular in-pool items.

    A session scores nothing unless the target lands in the top 10, so an under-filled
    list is pure waste: every empty slot is a free lottery ticket left unbought.
    """
    if len(ranked) >= limit:
        return ranked[:limit]
    chosen = list(ranked)
    seen = set(chosen)
    extras = sorted(
        (doc for doc in pool if doc not in seen),
        key=lambda doc: -index.popularity(doc),
    )
    for doc_id in extras:
        if len(chosen) >= limit:
            break
        chosen.append(doc_id)
    return chosen

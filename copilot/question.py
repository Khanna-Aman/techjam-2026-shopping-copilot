"""Adaptive clarification: which question is worth spending a turn on.

The starter agent never asks anything, and that single omission is why it converges at
turn 9.8 out of 10 -- the simulator only discloses information in response to a question,
so a silent agent is negotiating with itself.

Choosing *what* to ask is a real decision, and two structural facts constrain it:

* The simulator routes constraints to attributes with a fixed classifier that can only
  ever return budget, material, colour, size, style, use_case or feature. Asking
  ``category`` or ``brand`` therefore cannot yield anything, ever. Those two questions
  are pure turn-waste and are excluded outright.
* An attribute that every candidate shares tells us nothing even when answered. "Is it
  black?" is worthless in a pool where everything is black. So questions are scored by
  expected information gain over the live candidate pool, not by a fixed script.

The open-ended question is retained as a deliberate fallback for when no single attribute
discriminates -- which is the common case early in a Browsing session, where the pool is
still wide and almost any disclosure helps.
"""

from __future__ import annotations

import math
from array import array

from copilot.catalog import CatalogIndex
from copilot.config import AgentConfig
from copilot.slots import ConversationState

#: Attributes the simulator's constraint classifier can actually produce.
PRODUCTIVE_ATTRIBUTES: tuple[str, ...] = (
    "material", "color", "budget", "size", "style", "use_case", "feature",
)

#: Provably unproductive: the classifier never emits these labels.
UNPRODUCTIVE_ATTRIBUTES: frozenset[str] = frozenset({"category", "brand"})

#: The open question. Matches any undisclosed constraint regardless of type.
OPEN_ATTRIBUTE = "other"

# Trigger vocabularies mirrored from the simulator's classifier, used to approximate how
# a lexical attribute would partition the candidate pool.
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "size": ("size", "sizing", "width", "wide", "narrow"),
    "style": ("department", "style", "fit", "sleeve", "neck"),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work"),
}

#: Feature sentences are the most identifying constraints available but cannot be
#: modelled as a small partition, so they carry a calibrated constant prior instead.
_FEATURE_PRIOR = 0.90

#: P(the customer's hidden card contains at least one constraint of this type), measured
#: over all 50,000 catalog products by applying the published card-construction rule.
#: This uses the catalog only -- no session labels -- and it is the term a pure-entropy
#: model omits. Omitting it is a real error: a budget question splits the candidate pool
#: beautifully but goes unanswered 99.5% of the time, so its true value is near zero.
YIELD_PROBABILITY: dict[str, float] = {
    "feature": 0.958,
    "material": 0.573,
    "color": 0.427,
    "style": 0.162,
    "size": 0.076,
    "use_case": 0.016,
    "budget": 0.005,
}

#: Mean constraints returned when a specific question does land.
_SPECIFIC_YIELD = 1.5

#: The open question returns up to two undisclosed constraints of any type, so it is
#: answered essentially whenever anything remains undisclosed.
_OPEN_YIELD = 2.0

#: Average discriminative power of whatever the open question happens to return. It is
#: a mixture: early answers are weak (material, colour), later ones are feature sentences.
_OPEN_POWER = 0.75

#: An attribute rarely carries a second constraint once it has already answered.
_REPEAT_DISCOUNT = 0.2

#: Rotating phrasings for the open question. The field is customer-facing natural
#: language, and repeating one identical sentence for ten turns reads as a broken bot
#: even when the underlying decision is right. Selection is deterministic on turn index
#: so runs stay reproducible.
OPEN_VARIANTS: tuple[str, ...] = (
    "Tell me the one detail that matters most and I will narrow it down.",
    "What else should I know about what you are after?",
    "Anything specific I should be matching on?",
    "Give me one more detail and I can tighten these up.",
    "What would make one of these the right pick for you?",
)

QUESTION_TEXT: dict[str, str] = {
    "material": "What material are you hoping for?",
    "color": "Any particular colour you have in mind?",
    "budget": "Roughly what budget are you working with?",
    "size": "Is there a size or fit you need?",
    "style": "What style or cut are you going for?",
    "use_case": "What will you mainly be using it for?",
    "feature": "Is there a specific feature that matters most to you?",
    OPEN_ATTRIBUTE: "Tell me the one detail that matters most and I will narrow it down.",
}


def _normalised_entropy(counts: list[int]) -> float:
    """Shannon entropy of a partition, scaled to [0, 1]."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    present = [count for count in counts if count > 0]
    if len(present) <= 1:
        return 0.0
    entropy = 0.0
    for count in present:
        p = count / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(len(present))


def _partition_typed(values: array, pool: set[int], buckets: int) -> float:
    counts = [0] * (buckets + 1)
    for doc_id in pool:
        value = values[doc_id]
        counts[value + 1 if value >= 0 else 0] += 1
    return _normalised_entropy(counts)


def _partition_price(index: CatalogIndex, pool: set[int]) -> float:
    prices = [index.price[doc_id] for doc_id in pool]
    known = sorted(price for price in prices if price is not None and price >= 0)
    missing = len(prices) - len(known)
    if len(known) < 4:
        return 0.0
    quartiles = [known[len(known) // 4], known[len(known) // 2], known[3 * len(known) // 4]]
    counts = [missing, 0, 0, 0, 0]
    for price in known:
        if price <= quartiles[0]:
            counts[1] += 1
        elif price <= quartiles[1]:
            counts[2] += 1
        elif price <= quartiles[2]:
            counts[3] += 1
        else:
            counts[4] += 1
    return _normalised_entropy(counts)


def _partition_lexical(index: CatalogIndex, pool: set[int], triggers: tuple[str, ...]) -> float:
    present = 0
    for doc_id in pool:
        blob = index.blob[doc_id]
        if any(word in blob for word in triggers):
            present += 1
    return _normalised_entropy([present, len(pool) - present])


def discriminative_power(
    index: CatalogIndex, pool: set[int], attribute: str
) -> float:
    """How well an answer about ``attribute`` would split the candidate pool, in [0, 1]."""
    if not pool:
        return 0.0
    if attribute == "material":
        return _partition_typed(index.first_material, pool, 9)
    if attribute == "color":
        return _partition_typed(index.first_color, pool, 11)
    if attribute == "budget":
        return _partition_price(index, pool)
    if attribute == "feature":
        return _FEATURE_PRIOR
    triggers = _TRIGGERS.get(attribute)
    if triggers:
        return _partition_lexical(index, pool, triggers)
    return 0.0


def expected_gain(
    index: CatalogIndex, state: ConversationState, pool: set[int], attribute: str
) -> float:
    """Expected value of asking about ``attribute``, in expected-constraints-of-signal.

    Value is the product of three terms, not entropy alone:

        P(the customer can answer) x E[constraints returned] x how well they split the pool

    Dropping the first term is what makes a naive information-gain policy pick questions
    that are beautifully discriminating and almost never answered.
    """
    if attribute in state.exhausted or attribute in UNPRODUCTIVE_ATTRIBUTES:
        return 0.0
    if not pool:
        return 0.0
    probability = YIELD_PROBABILITY.get(attribute, 0.0)
    if any(item.attribute == attribute for item in state.constraints):
        probability *= _REPEAT_DISCOUNT
    return probability * _SPECIFIC_YIELD * discriminative_power(index, pool, attribute)


def open_gain(state: ConversationState, pool: set[int]) -> float:
    """Expected value of the open question, on the same scale as ``expected_gain``."""
    if OPEN_ATTRIBUTE in state.exhausted or not pool:
        return 0.0
    return _OPEN_YIELD * _OPEN_POWER


def choose_attribute(
    index: CatalogIndex,
    state: ConversationState,
    pool: set[int],
    config: AgentConfig,
) -> str | None:
    """Select the attribute to ask about this turn, or ``None`` to stay silent."""
    strategy = config.clarify_strategy
    if strategy == "none":
        return None

    open_available = OPEN_ATTRIBUTE not in state.exhausted
    if strategy == "open":
        return OPEN_ATTRIBUTE if open_available else _best_specific(index, state, pool)[0]

    best, gain = _best_specific(index, state, pool)

    if strategy == "infogain":
        # Specific questions only. Retained as an ablation contrast: it shows what the
        # policy gives up by refusing to ask an open question.
        if best is not None and gain >= config.min_infogain:
            return best
        return OPEN_ATTRIBUTE if open_available else None

    # hybrid: compare every option on one expected-value scale, the open question
    # included, and take the best. In most states the open question wins outright,
    # because it is answered whenever anything remains undisclosed and returns two
    # constraints at once. It yields to a specific question once the open channel is
    # exhausted or a typed attribute becomes sharply discriminating.
    open_value = open_gain(state, pool) if open_available else 0.0
    if best is not None and gain > open_value and gain >= config.min_infogain:
        return best
    if open_available:
        return OPEN_ATTRIBUTE
    return best


def _best_specific(
    index: CatalogIndex, state: ConversationState, pool: set[int]
) -> tuple[str | None, float]:
    best: str | None = None
    best_gain = 0.0
    for attribute in PRODUCTIVE_ATTRIBUTES:
        if attribute in state.exhausted:
            continue
        gain = expected_gain(index, state, pool, attribute)
        if gain > best_gain:
            best_gain = gain
            best = attribute
    return best, best_gain


def question_text(attribute: str | None, turn: int = 0) -> str:
    """Render the clarification question as customer-facing prose."""
    if attribute is None:
        return "Here are the closest matches I found."
    if attribute == OPEN_ATTRIBUTE:
        return OPEN_VARIANTS[max(turn - 1, 0) % len(OPEN_VARIANTS)]
    return QUESTION_TEXT.get(attribute, OPEN_VARIANTS[0])

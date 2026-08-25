"""Conversational state: constraint slots, accumulation, and override erasure.

This is the module the challenge brief calls the "Dynamic State Machine". It tracks two
things the starter agent ignores entirely:

* **Information accumulation.** Constraints disclosed across turns are additive, and
  later turns carry the most discriminative information, because the simulator reveals
  weak attributes (material, colour) before distinctive product features.
* **Intent Override.** On turn 3 or 4 of an override session the customer retracts an
  earlier preference. Merely ignoring the retracted value is not enough: it was mined
  verbatim from the target product, so it keeps scoring well on exactly the wrong items.
  It has to be actively penalised.

Constraints are also *typed*, because their type determines how a product can satisfy
them: a material or colour is a set-membership test, a budget is a numeric proximity
test, and a feature sentence is a phrase-containment test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.text import (
    COLORS,
    MATERIALS,
    classify_constraint,
    normalise,
    parse_price,
    terms,
)

# Constraint kinds
MATERIAL = "material"
COLOR = "color"
BUDGET = "budget"
PHRASE = "phrase"

_MATERIAL_SET = frozenset(MATERIALS)
_COLOR_SET = frozenset(COLORS)


def classify_kind(text: str) -> tuple[str, str | None]:
    """Determine how a constraint string must be tested against a product.

    Returns ``(kind, value)`` where value is the normalised material name, colour name,
    or price, and ``None`` for free-text phrases.
    """
    lowered = normalise(text)
    if lowered.startswith("color:"):
        candidate = lowered.split(":", 1)[1].strip()
        if candidate in _COLOR_SET:
            return COLOR, candidate
    if lowered in _MATERIAL_SET:
        return MATERIAL, lowered
    if lowered in _COLOR_SET:
        return COLOR, lowered
    if "budget" in lowered:
        price = parse_price(lowered)
        if price is not None:
            return BUDGET, str(price)
    return PHRASE, None


@dataclass
class Constraint:
    """One piece of disclosed customer intent."""

    text: str
    turn: int
    kind: str
    value: str | None
    attribute: str
    active: bool = True
    provisional: bool = False

    @property
    def tokens(self) -> list[str]:
        return terms(self.text)


@dataclass
class ConversationState:
    """Per-session slot memory. One instance per ``reset`` call."""

    session_id: str
    profile: dict = field(default_factory=dict)
    category: str | None = None
    scenario: str = "browsing"
    turn: int = 0

    constraints: list[Constraint] = field(default_factory=list)
    retracted: list[Constraint] = field(default_factory=list)

    asked: list[str] = field(default_factory=list)
    pending_attribute: str | None = None
    exhausted: set[str] = field(default_factory=set)
    boundary_seen: bool = False
    override_applied: bool = False

    # ------------------------------------------------------------------ accumulation
    def _seen(self) -> set[str]:
        return {normalise(item.text) for item in self.constraints}

    def add_constraints(
        self, values: list[str], *, turn: int, provisional: bool = False
    ) -> list[Constraint]:
        """Record newly disclosed constraints, ignoring exact repeats."""
        seen = self._seen()
        added: list[Constraint] = []
        for raw in values:
            text = raw.strip()
            if not text:
                continue
            key = normalise(text)
            if key in seen:
                continue
            seen.add(key)
            kind, value = classify_kind(text)
            item = Constraint(
                text=text,
                turn=turn,
                kind=kind,
                value=value,
                attribute=classify_constraint(text),
                provisional=provisional,
            )
            self.constraints.append(item)
            added.append(item)
        return added

    def retract_provisional(self) -> list[Constraint]:
        """Erase the Intent Override decoy and move it to the penalised set."""
        moved: list[Constraint] = []
        for item in list(self.constraints):
            if item.provisional:
                item.active = False
                self.constraints.remove(item)
                self.retracted.append(item)
                moved.append(item)
        return moved

    def mark_exhausted(self, attribute: str | None) -> None:
        if attribute:
            self.exhausted.add(attribute)

    # -------------------------------------------------------------------- accessors
    @property
    def active_constraints(self) -> list[Constraint]:
        return [item for item in self.constraints if item.active]

    def has_hard_signal(self) -> bool:
        """True once any constraint beyond the bare category is known."""
        return bool(self.active_constraints)

    def profile_tags(self) -> list[str]:
        tags = self.profile.get("preference_tags") if isinstance(self.profile, dict) else None
        if not isinstance(tags, list):
            return []
        return [str(tag).lower() for tag in tags if isinstance(tag, (str, int, float))]

    def query_terms(
        self,
        *,
        constraint_boost: float,
        category_boost: float,
        decoy_penalty: float,
    ) -> tuple[list[str], dict[str, float]]:
        """Build the weighted bag of query terms for the current belief state.

        Retracted values receive a *negative* weight rather than being dropped, which is
        what makes override erasure actually move the ranking.
        """
        tokens: list[str] = []
        weights: dict[str, float] = {}

        if self.category:
            for token in terms(self.category):
                tokens.append(token)
                weights[token] = max(weights.get(token, 0.0), category_boost)

        for item in self.active_constraints:
            for token in item.tokens:
                tokens.append(token)
                weights[token] = max(weights.get(token, 0.0), constraint_boost)

        for item in self.retracted:
            for token in item.tokens:
                # Never penalise a token that an active constraint also relies on.
                if weights.get(token, 0.0) > 0.0:
                    continue
                tokens.append(token)
                weights[token] = -abs(decoy_penalty)

        return tokens, weights

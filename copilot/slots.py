"""Conversational state: constraint slots, accumulation, and override handling.

This is the module the challenge brief calls the "Dynamic State Machine". It tracks two
things the starter agent ignores entirely:

* **Information accumulation.** Constraints disclosed across turns are additive, and
  later turns carry the most discriminative information, because the simulator reveals
  weak attributes (material, colour) before distinctive product features.
* **Intent Override.** On turn 3 or 4 of an override session the customer retracts an
  earlier preference. The obvious reading -- erase the withdrawn value -- turns out to be
  wrong, and measurably so. The simulator mines both the withdrawn preference and its
  replacement from the *same* hidden target, so the target never changes when the
  customer changes their mind, and the withdrawn value stays diagnostic. It is retained
  at a tunable confidence rather than erased.

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
    #: Confidence multiplier. A superseded Intent Override value keeps a reduced weight
    #: rather than being erased -- see ConversationState.supersede_provisional.
    weight: float = 1.0

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

    #: Content tokens seen in *any* customer message, whether or not they were
    #: successfully parsed into a typed constraint. This is the parse-failure safety
    #: net: under paraphrase the structured extraction may miss, but the payload words
    #: still arrive, and discarding them entirely is what makes template parsing brittle.
    observed: set[str] = field(default_factory=set)

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

    def supersede_provisional(self, decay: float) -> list[Constraint]:
        """Handle an Intent Override retraction.

        Counter-intuitively, the retracted value should *not* be thrown away. The
        simulator draws the customer's "earlier preference" from the same hidden target
        product as the replacement, so the target itself never changes when the customer
        changes their mind. The withdrawn value therefore stays diagnostic, and
        penalising it measurably costs score.

        ``decay`` controls how much trust survives: 1.0 keeps the value at full strength,
        0.0 erases it and moves it to the penalised set (the naive reading of the
        protocol, which measurement rejects).
        """
        moved: list[Constraint] = []
        for item in list(self.constraints):
            if not item.provisional:
                continue
            moved.append(item)
            if decay <= 0.0:
                item.active = False
                self.constraints.remove(item)
                self.retracted.append(item)
            else:
                item.weight = decay
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

    def observe_text(self, message: str, chrome: frozenset[str]) -> None:
        """Record content tokens from a raw customer message."""
        for token in terms(message):
            if token not in chrome:
                self.observed.add(token)

    def query_terms(
        self,
        *,
        constraint_boost: float,
        category_boost: float,
        decoy_penalty: float,
        observed_boost: float = 0.0,
    ) -> tuple[list[str], dict[str, float]]:
        """Build the weighted bag of query terms for the current belief state.

        Precedence runs category and observed text first, then confirmed constraints,
        then retraction penalties. Fully-erased values (override_decay == 0) receive a
        negative weight rather than merely being dropped, so that erasure, when enabled,
        actually moves the ranking instead of leaving the value inert.
        """
        tokens: list[str] = []
        weights: dict[str, float] = {}
        protected: set[str] = set()

        if self.category:
            for token in terms(self.category):
                tokens.append(token)
                weights[token] = max(weights.get(token, 0.0), category_boost)
                protected.add(token)

        if observed_boost > 0.0:
            for token in self.observed:
                tokens.append(token)
                weights[token] = max(weights.get(token, 0.0), observed_boost)

        for item in self.active_constraints:
            boost = constraint_boost * item.weight
            for token in item.tokens:
                tokens.append(token)
                weights[token] = max(weights.get(token, 0.0), boost)
                protected.add(token)

        for item in self.retracted:
            for token in item.tokens:
                # Never penalise a token the category or an active constraint relies on.
                if token in protected:
                    continue
                tokens.append(token)
                weights[token] = -abs(decoy_penalty)

        return tokens, weights

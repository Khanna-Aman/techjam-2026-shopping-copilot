"""Text normalisation and simulator-protocol vocabulary.

The customer simulator in this challenge is a *published, deterministic* policy: the
participant kit ships the exact message templates, the attribute taxonomy, and the
regexes used to mine constraints out of product metadata.  This module re-states that
vocabulary so the agent can model the environment it is talking to.

It is deliberately a re-implementation rather than an import of ``evaluator``:
the agent must never depend on the harness module layout, because the official scoring
run may vendor the evaluator differently.  Nothing here reads ground truth.
"""

from __future__ import annotations

import re

# --- Attribute taxonomy -------------------------------------------------------------
# The set the simulator accepts in ``ask_attribute``.  Anything outside it is coerced
# to "other" by the simulator, so we never emit a value that is not in this set.
ALLOWED_ATTRIBUTES: tuple[str, ...] = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

# --- Constraint mining vocabulary ---------------------------------------------------
MATERIALS: tuple[str, ...] = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
)
COLORS: tuple[str, ...] = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)

# Field order used to build a product's searchable blob.
SEARCH_FIELDS: tuple[str, ...] = (
    "title", "features", "details", "description", "categories", "store",
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Stopwords: the starter's list plus the closed-class words the simulator's own
# templates inject.  Template words carry zero discriminative signal but would
# otherwise dominate a bag-of-words query, because they appear in every message.
STOPWORDS: frozenset[str] = frozenset({
    # starter baseline list
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    # simulator template chrome
    "im", "still", "exploring", "key", "requirement", "what", "matters",
    "dont", "have", "additional", "preference", "actually", "ignore",
    "earlier", "need", "those", "options", "not", "quite", "right", "yet",
    "ask", "about", "one", "specific", "attribute", "judgment", "use",
    "your", "prefer", "different", "style", "prioritize", "target",
    "requirements", "around",
})


def normalise(value: str) -> str:
    """Collapse whitespace and lowercase, for stable substring comparison."""
    return re.sub(r"\s+", " ", value).strip().lower()


def flatten_value(value: object) -> list[str]:
    """Render a raw catalog field as a list of strings (mirrors the kit's helper)."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def searchable_text(product: dict) -> str:
    """Concatenate the participant-visible fields into one blob."""
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def coarse_category(values: list[str]) -> str:
    """Reduce a product's category path to the simulator's coarse label.

    The opening customer message embeds exactly this string for the hidden target, so
    it is the single strongest turn-1 signal available to the agent.
    """
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def terms(text: str) -> list[str]:
    """Tokenise to lowercase content words, dropping stopwords and single chars."""
    out: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) > 1 and lowered not in STOPWORDS:
            out.append(lowered)
    return out


def classify_constraint(value: str) -> str:
    """Map a constraint string to the attribute that would elicit it.

    Mirrors the simulator's published routing rule.  The agent uses it in reverse: to
    predict which question is capable of unlocking which piece of hidden information,
    and therefore to avoid asking questions that provably cannot pay out.
    """
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


_PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")


def parse_price(value: object) -> float | None:
    """Best-effort numeric price from a catalog value or a 'budget around $X' string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        price = float(value)
        return price if price > 0 else None
    text = str(value)
    match = _PRICE_RE.search(text)
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        price = float(match.group(1))
    except ValueError:
        return None
    return price if price > 0 else None


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def match_text(value: str) -> str:
    """Punctuation-flattened lowercase text for phrase-containment tests.

    Constraints are mined from product metadata but rendered with different separators
    than the search blob: a details dict becomes ``"Department: Womens"`` in a constraint
    and ``"Department Womens"`` in the blob. Flattening every non-alphanumeric run to a
    single space makes the two directly comparable, so an exact phrase match works.
    """
    return _NON_ALNUM_RE.sub(" ", value.lower()).strip()

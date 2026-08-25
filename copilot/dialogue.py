"""Parsing of the customer simulator's message protocol.

The simulator emits a small, fixed set of message templates. Parsing them exactly, rather
than throwing the raw string into a bag of words, is what converts this from a keyword
search problem into a state-tracking problem.

Two consequences matter most:

* The three opening templates are mutually distinguishable, so the scenario
  (Buying / Browsing / Intent Override) is known *exactly* from turn 1 rather than
  guessed. That is the dual-track routing signal the challenge asks for.
* Refusals come in two distinct forms. "no preference for X" is the one-shot Boundary
  behaviour; "no additional preference for X" means that attribute is exhausted. Both
  mean never ask X again, and conflating them wastes turns.

No ground truth is read here; every field comes from the message text the agent is
handed at runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.text import ALLOWED_ATTRIBUTES, normalise

# Scenario labels. "browsing" also covers Boundary sessions, which open identically;
# Boundary reveals itself later via the one-shot "no preference" reply.
BUYING = "buying"
BROWSING = "browsing"
INTENT_OVERRIDE = "intent_override"

_PREFIX = "i'm looking for "
_EXPLORING_SUFFIX = ", but i'm still exploring."
_REQUIREMENT_MARKER = ". a key requirement is: "

_DISCLOSURE_RE = re.compile(r"^for that,\s*what matters is:\s*(.+?)\.?$", re.I)
_NO_ADDITIONAL_RE = re.compile(r"^i don't have an additional preference for\s+(.+?)\.?$", re.I)
_NO_PREFERENCE_RE = re.compile(
    r"^i don't have a preference for\s+(.+?);\s*please use your judgment\.?$", re.I
)
_ASK_MORE_RE = re.compile(r"ask me about one specific attribute", re.I)
_OVERRIDE_RE = re.compile(
    r"ignore my earlier preference\.\s*what i need is:\s*(.+?)\.?$", re.I
)

# Reply kinds
DISCLOSURE = "disclosure"
NO_ADDITIONAL = "no_additional"
BOUNDARY = "boundary"
ASK_MORE = "ask_more"
OVERRIDE = "override"
UNKNOWN = "unknown"

MAX_CATEGORY_WORDS = 10


@dataclass
class Opening:
    """Structured view of the first customer message."""

    category: str | None
    scenario: str
    constraints: list[str] = field(default_factory=list)
    # True when the opening constraint is the Intent Override decoy, which will be
    # retracted on turn 3 or 4 and must not be allowed to anchor the ranking.
    provisional: bool = False


@dataclass
class Reply:
    """Structured view of a follow-up customer message."""

    kind: str
    attribute: str | None = None
    constraints: list[str] = field(default_factory=list)


def resolve_category(fragment: str, bucket_lookup: dict[str, str]) -> tuple[str | None, str]:
    """Match the longest known catalog category that prefixes ``fragment``.

    Returns ``(category, remainder)``. Matching against the catalog's own category
    vocabulary avoids brittle punctuation splitting: category labels can themselves
    contain separators, so a naive split on ". " mis-parses a real subset of sessions.
    """
    cleaned = normalise(fragment)
    direct = bucket_lookup.get(cleaned.rstrip(" .,"))
    if direct is not None:
        return direct, ""

    words = cleaned.split(" ")
    for size in range(min(MAX_CATEGORY_WORDS, len(words)), 0, -1):
        candidate = " ".join(words[:size]).rstrip(" .,;:")
        name = bucket_lookup.get(candidate)
        if name is not None:
            remainder = " ".join(words[size:]).lstrip(" .,;:")
            return name, remainder
    return None, cleaned


def parse_opening(message: str, bucket_lookup: dict[str, str]) -> Opening:
    """Classify the opening message and extract its category and any disclosed constraint."""
    lowered = normalise(message)
    body = lowered[len(_PREFIX):] if lowered.startswith(_PREFIX) else lowered

    if body.endswith(_EXPLORING_SUFFIX):
        fragment = body[: -len(_EXPLORING_SUFFIX)]
        category, _ = resolve_category(fragment, bucket_lookup)
        return Opening(category=category or fragment or None, scenario=BROWSING)

    marker = body.find(_REQUIREMENT_MARKER)
    if marker != -1:
        fragment = body[:marker]
        constraint = body[marker + len(_REQUIREMENT_MARKER):].strip().rstrip(".")
        category, _ = resolve_category(fragment, bucket_lookup)
        return Opening(
            category=category or fragment or None,
            scenario=BUYING,
            constraints=[constraint] if constraint else [],
        )

    # Remaining shape is "<category>. <old_value>", the Intent Override opener. The
    # trailing value is a decoy that gets retracted, so it is flagged provisional.
    category, remainder = resolve_category(body, bucket_lookup)
    constraints = [remainder.strip().rstrip(".")] if remainder.strip() else []
    return Opening(
        category=category,
        scenario=INTENT_OVERRIDE,
        constraints=constraints,
        provisional=True,
    )


def _split_constraints(payload: str) -> list[str]:
    return [part.strip().rstrip(".") for part in payload.split(";") if part.strip()]


def _clean_attribute(value: str) -> str | None:
    candidate = normalise(value).strip(" .;:")
    return candidate if candidate in ALLOWED_ATTRIBUTES else None


def parse_reply(message: str) -> Reply:
    """Classify a follow-up customer message into one of the simulator's reply kinds."""
    text = normalise(message)
    if not text:
        return Reply(kind=UNKNOWN)

    match = _OVERRIDE_RE.search(text)
    if match:
        return Reply(kind=OVERRIDE, constraints=_split_constraints(match.group(1)))

    match = _NO_PREFERENCE_RE.match(text)
    if match:
        return Reply(kind=BOUNDARY, attribute=_clean_attribute(match.group(1)))

    match = _NO_ADDITIONAL_RE.match(text)
    if match:
        return Reply(kind=NO_ADDITIONAL, attribute=_clean_attribute(match.group(1)))

    match = _DISCLOSURE_RE.match(text)
    if match:
        return Reply(kind=DISCLOSURE, constraints=_split_constraints(match.group(1)))

    if _ASK_MORE_RE.search(text):
        return Reply(kind=ASK_MORE)

    return Reply(kind=UNKNOWN)

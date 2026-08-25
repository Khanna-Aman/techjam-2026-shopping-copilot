"""Parsing of the customer's messages, with graceful degradation.

The simulator emits a small set of fixed templates, and parsing them exactly turns this
from a keyword-search problem into a state-tracking one. But the competition
specification warns that the organiser may add natural-language paraphrasing, noting only
that paraphrasing "cannot decide correctness". An agent that *only* matches templates is
therefore betting the whole score on wording that is explicitly declared unstable.

So parsing runs in two tiers:

* **Exact tier** - the known templates, which give perfect scenario and category recovery
  when the wording is untouched.
* **Fuzzy tier** - content-driven recovery used whenever the exact tier misses. The
  category is found by scanning every n-gram of the message against the catalog's own
  category vocabulary rather than by splitting on punctuation; intent is read from
  negation and retraction cues rather than fixed phrases.

The design assumption behind the fuzzy tier is that a paraphraser rewrites *chrome*, not
*content*: product attributes like "cotton" or a feature sentence survive rewording,
because they are the payload the message exists to carry.

No ground truth is read here; every field comes from the runtime message text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.text import ALLOWED_ATTRIBUTES, normalise, terms

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
_OVERRIDE_RE = re.compile(r"ignore my earlier preference\.\s*what i need is:\s*(.+?)\.?$", re.I)

# --- fuzzy cue vocabularies ---------------------------------------------------------
# Retraction: the customer is replacing an earlier stated preference.
_RETRACT_CUES = (
    "ignore my earlier", "ignore what i", "scratch that", "forget what i",
    "forget that", "instead of", "actually, i", "actually i need",
    "changed my mind", "never mind", "disregard",
)
# Still-exploring: no hard constraint is being offered.
_BROWSE_CUES = (
    "exploring", "browsing", "just looking", "still making up", "haven't decided",
    "havent decided", "not decided", "making up my mind", "for now", "no rush",
)
# A hard requirement is being stated.
_REQUIRE_CUES = (
    "requirement", "must be", "must have", "must-have", "has to be", "have to be",
    "it has to", "i need", "needs to be", "key thing", "essential", "important that",
)
# Negation of a preference, i.e. the attribute is spent.
_NEGATION_CUES = (
    "don't have", "dont have", "do not have", "no preference", "no strong",
    "nothing particular", "nothing specific", "up to you", "your call",
    "no opinion", "doesn't matter", "doesnt matter", "either is fine",
    "your judgment", "your judgement", "not fussed", "no feelings",
)
_ASK_MORE_CUES = (
    "not quite", "none of those", "ask me", "not right", "what do you want to know",
    "try again",
)

# Words a paraphraser adds as connective tissue. Stripped before treating residual text
# as a constraint, so filler never becomes a phantom requirement.
_CHROME_TOKENS = frozenset({
    "hi", "hey", "hello", "um", "uh", "so", "well", "honestly", "guess", "know",
    "show", "shopping", "browsing", "after", "want", "wanted", "need", "needed",
    "care", "caring", "mainly", "really", "thing", "things", "one", "must",
    "haven", "decided", "making", "mind", "now", "though", "there", "let",
    "scratch", "forget", "disregard", "actually", "instead", "said", "call",
    "judgment", "judgement", "matter", "matters", "preference", "preferences",
    "additional", "particular", "specific", "strong", "opinion", "fine", "either",
    "please", "thanks", "ok", "okay", "sure", "maybe", "just", "still", "yet",
})

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
    exact: bool = True


@dataclass
class Reply:
    """Structured view of a follow-up customer message."""

    kind: str
    attribute: str | None = None
    constraints: list[str] = field(default_factory=list)
    exact: bool = True


def _contains(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def scan_category(text: str, bucket_lookup: dict[str, str]) -> tuple[str | None, tuple[int, int]]:
    """Find the longest catalog category appearing anywhere in ``text``.

    Scanning every n-gram against the catalog's own category vocabulary is what makes
    category recovery survive rewording: it never depends on a prefix, a delimiter, or
    where in the sentence the category happens to land. Messages are short, so the
    n-gram sweep is a few hundred dictionary probes.
    """
    words = normalise(text).split(" ")
    best: str | None = None
    best_span = (0, 0)
    best_len = 0
    for start in range(len(words)):
        upper = min(len(words), start + MAX_CATEGORY_WORDS)
        for end in range(upper, start, -1):
            if end - start <= best_len:
                break
            candidate = " ".join(words[start:end]).strip(" .,;:!?-")
            name = bucket_lookup.get(candidate)
            if name is not None:
                best = name
                best_span = (start, end)
                best_len = end - start
                break
    return best, best_span


def _residual(text: str, span: tuple[int, int]) -> str:
    """Message text with the matched category span removed."""
    words = normalise(text).split(" ")
    start, end = span
    if end > start:
        words = words[:start] + words[end:]
    return " ".join(words).strip(" .,;:-")


def _clean_segment(segment: str) -> str:
    return segment.strip(" .,;:!?-\"'")


def _content_segments(text: str) -> list[str]:
    """Split residual text into candidate constraint strings.

    Semicolons are the simulator's own separator and are trusted. Commas are not split
    on, because real constraint strings contain them; over-splitting would shatter a
    distinctive feature sentence into useless fragments.
    """
    parts = [p for p in text.split(";")] if ";" in text else [text]
    out: list[str] = []
    for part in parts:
        cleaned = _clean_segment(part)
        if not cleaned:
            continue
        content = [token for token in terms(cleaned) if token not in _CHROME_TOKENS]
        if not content:
            continue
        out.append(cleaned)
    return out


def resolve_category(fragment: str, bucket_lookup: dict[str, str]) -> tuple[str | None, str]:
    """Match the longest known catalog category that prefixes ``fragment``."""
    cleaned = normalise(fragment)
    direct = bucket_lookup.get(cleaned.rstrip(" .,"))
    if direct is not None:
        return direct, ""
    words = cleaned.split(" ")
    for size in range(min(MAX_CATEGORY_WORDS, len(words)), 0, -1):
        candidate = " ".join(words[:size]).rstrip(" .,;:")
        name = bucket_lookup.get(candidate)
        if name is not None:
            return name, " ".join(words[size:]).lstrip(" .,;:")
    return None, cleaned


def parse_opening(message: str, bucket_lookup: dict[str, str]) -> Opening:
    """Classify the opening message and extract its category and any disclosed constraint.

    Opening constraints are always treated as retractable. In an Intent Override session
    the opening value is a decoy that gets withdrawn; in a Buying session no retraction
    ever arrives, so the flag costs nothing. Marking unconditionally removes the need to
    distinguish the two openings under paraphrase, where that distinction is unreliable.
    """
    lowered = normalise(message)

    # --- exact tier ---------------------------------------------------------------
    if lowered.startswith(_PREFIX):
        body = lowered[len(_PREFIX):]
        if body.endswith(_EXPLORING_SUFFIX):
            fragment = body[: -len(_EXPLORING_SUFFIX)]
            category, _ = resolve_category(fragment, bucket_lookup)
            if category:
                return Opening(category=category, scenario=BROWSING)
        marker = body.find(_REQUIREMENT_MARKER)
        if marker != -1:
            fragment = body[:marker]
            constraint = _clean_segment(body[marker + len(_REQUIREMENT_MARKER):])
            category, _ = resolve_category(fragment, bucket_lookup)
            if category:
                return Opening(
                    category=category,
                    scenario=BUYING,
                    constraints=[constraint] if constraint else [],
                )
        category, remainder = resolve_category(body, bucket_lookup)
        if category:
            residual = _clean_segment(remainder)
            return Opening(
                category=category,
                scenario=INTENT_OVERRIDE,
                constraints=[residual] if residual else [],
            )

    # --- fuzzy tier ---------------------------------------------------------------
    category, span = scan_category(message, bucket_lookup)
    residual = _residual(message, span) if category else normalise(message)

    if _contains(lowered, _BROWSE_CUES):
        scenario = BROWSING
        constraints: list[str] = []
    else:
        constraints = _content_segments(residual)
        scenario = BUYING if _contains(lowered, _REQUIRE_CUES) else INTENT_OVERRIDE

    return Opening(
        category=category,
        scenario=scenario,
        constraints=constraints,
        exact=False,
    )


def _split_constraints(payload: str) -> list[str]:
    return [_clean_segment(part) for part in payload.split(";") if _clean_segment(part)]


def _clean_attribute(value: str) -> str | None:
    candidate = normalise(value).strip(" .;:")
    return candidate if candidate in ALLOWED_ATTRIBUTES else None


def _find_attribute(text: str) -> str | None:
    """Locate any allowed attribute name mentioned in the message."""
    for attribute in ALLOWED_ATTRIBUTES:
        if re.search(rf"\b{re.escape(attribute.replace('_', ' '))}\b", text):
            return attribute
        if re.search(rf"\b{re.escape(attribute)}\b", text):
            return attribute
    return None


def parse_reply(message: str) -> Reply:
    """Classify a follow-up customer message into one of the simulator's reply kinds."""
    text = normalise(message)
    if not text:
        return Reply(kind=UNKNOWN)

    # --- exact tier ---------------------------------------------------------------
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

    # --- fuzzy tier ---------------------------------------------------------------
    # Retraction is checked first: it carries a replacement value and must not be
    # mistaken for an ordinary disclosure, or the decoy would never be erased.
    if _contains(text, _RETRACT_CUES):
        payload = text
        for cue in _RETRACT_CUES:
            index = payload.find(cue)
            if index != -1:
                payload = payload[index + len(cue):]
                break
        payload = re.sub(r"^(?:.*?\bis\b|.*?\bneed\b)[:\s]*", "", payload, count=1)
        return Reply(
            kind=OVERRIDE, constraints=_content_segments(_clean_segment(payload)), exact=False
        )

    if _contains(text, _NEGATION_CUES):
        attribute = _find_attribute(text)
        # "please use your judgment" is the Boundary tell; a plain negation is exhaustion.
        boundary = any(
            cue in text
            for cue in ("judgment", "judgement", "up to you", "your call", "your choice")
        )
        return Reply(
            kind=BOUNDARY if boundary else NO_ADDITIONAL, attribute=attribute, exact=False
        )

    if _contains(text, _ASK_MORE_CUES):
        return Reply(kind=ASK_MORE, exact=False)

    # Anything else that still carries content is treated as a disclosure. A colon is a
    # strong hint that the payload follows it.
    payload = text.split(":", 1)[1] if ":" in text else text
    segments = _content_segments(_clean_segment(payload))
    if segments:
        return Reply(kind=DISCLOSURE, constraints=segments, exact=False)
    return Reply(kind=UNKNOWN, exact=False)

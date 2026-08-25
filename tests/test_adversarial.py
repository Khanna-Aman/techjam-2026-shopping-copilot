"""Adversarial and contract tests.

The evaluator converts any exception the agent raises into an empty response, which
silently forfeits a turn rather than reporting a bug. A crash therefore does not look
like a crash -- it looks like a slightly lower score. These tests exist to make that
class of failure loud.

Every test here asserts the same two things: the agent does not raise, and whatever it
returns still satisfies the published Agent contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402
from copilot.text import ALLOWED_ATTRIBUTES  # noqa: E402

TOP_K = 10


def assert_valid_response(response: object, *, top_k: int = TOP_K) -> None:
    """Assert the published Agent output contract."""
    assert isinstance(response, dict), "response must be a dict"
    assert isinstance(response["message"], str), "message must be a string"

    attribute = response["ask_attribute"]
    assert attribute is None or attribute in ALLOWED_ATTRIBUTES, (
        f"ask_attribute must be null or one of {ALLOWED_ATTRIBUTES}, got {attribute!r}"
    )

    recommendations = response["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) <= top_k, "must not exceed top_k recommendations"
    seen: set[str] = set()
    for item in recommendations:
        assert isinstance(item, dict), "each recommendation must be an object"
        asin = item["parent_asin"]
        assert isinstance(asin, str) and asin, "parent_asin must be a non-empty string"
        assert asin not in seen, f"duplicate parent_asin {asin!r} in recommendations"
        seen.add(asin)

    usage = response.get("usage")
    if usage is not None:
        assert isinstance(usage, dict)
        for key in ("prompt_tokens", "completion_tokens"):
            assert isinstance(usage[key], int) and usage[key] >= 0


@pytest.fixture()
def agent(synthetic_catalog, synthetic_index) -> ShoppingCopilot:
    return ShoppingCopilot(synthetic_catalog, index=synthetic_index)


# --------------------------------------------------------------------- hostile input
HOSTILE_MESSAGES = [
    pytest.param("", id="empty"),
    pytest.param("   \t\n  ", id="whitespace-only"),
    pytest.param("?" * 5000, id="very-long-punctuation"),
    pytest.param("\x00\x01\x02 null bytes", id="control-characters"),
    pytest.param("찾고 있어요 면 티셔츠", id="non-latin-script"),
    pytest.param("🛍️👗👠 emoji only 🧣", id="emoji"),
    pytest.param("'; DROP TABLE products; --", id="sql-injection-shape"),
    pytest.param('{"injected": "json"}', id="json-payload"),
    pytest.param("<script>alert(1)</script>", id="html-payload"),
    pytest.param("../../etc/passwd", id="path-traversal-shape"),
    pytest.param("I'm looking for " + "shirts " * 400, id="repetition-flood"),
    pytest.param("NULL", id="literal-null-word"),
    pytest.param("I don't have a preference for ; please use your judgment.", id="empty-attribute"),
    pytest.param("For that, what matters is: .", id="empty-disclosure"),
    pytest.param("Actually, ignore my earlier preference. What I need is: .", id="empty-override"),
    pytest.param(";;;;;;;;", id="separators-only"),
]


@pytest.mark.parametrize("message", HOSTILE_MESSAGES)
def test_hostile_messages_never_break_the_contract(agent: ShoppingCopilot, message: str) -> None:
    agent.reset("hostile", {})
    for turn in range(1, 4):
        response = agent.respond("hostile", message, turn, TOP_K)
        assert_valid_response(response)


MALFORMED_PROFILES = [
    pytest.param({}, id="empty-dict"),
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param("a string", id="string"),
    pytest.param(42, id="int"),
    pytest.param({"preference_tags": None}, id="tags-none"),
    pytest.param({"preference_tags": "not-a-list"}, id="tags-string"),
    pytest.param({"preference_tags": [None, 1, {"x": 1}]}, id="tags-mixed-junk"),
    pytest.param({"preference_tags": ["fit"] * 10000}, id="tags-flood"),
    pytest.param({"average_prior_rating": float("nan")}, id="nan-rating"),
]


@pytest.mark.parametrize("profile", MALFORMED_PROFILES)
def test_malformed_profiles_are_survivable(agent: ShoppingCopilot, profile: object) -> None:
    agent.reset("profile", profile)  # type: ignore[arg-type]
    response = agent.respond("profile", "I'm looking for Shirts T-Shirts.", 1, TOP_K)
    assert_valid_response(response)


BAD_TURNS = [
    pytest.param(0, id="zero"),
    pytest.param(-5, id="negative"),
    pytest.param(999, id="beyond-max"),
    pytest.param(None, id="none"),
    pytest.param("3", id="string"),
]


@pytest.mark.parametrize("turn", BAD_TURNS)
def test_out_of_contract_turn_values(agent: ShoppingCopilot, turn: object) -> None:
    agent.reset("turns", {})
    response = agent.respond("turns", "I'm looking for Shirts T-Shirts.", turn, TOP_K)  # type: ignore[arg-type]
    assert_valid_response(response)


BAD_TOP_K = [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(1, id="one"),
    pytest.param(3, id="three"),
    pytest.param(None, id="none"),
    pytest.param(10**6, id="huge"),
]


@pytest.mark.parametrize("top_k", BAD_TOP_K)
def test_top_k_is_respected_or_safely_defaulted(agent: ShoppingCopilot, top_k: object) -> None:
    agent.reset("topk", {})
    response = agent.respond("topk", "I'm looking for Shirts T-Shirts.", 1, top_k)  # type: ignore[arg-type]
    limit = top_k if isinstance(top_k, int) and top_k > 0 else TOP_K
    # A catalog smaller than top_k simply yields fewer rows; never more.
    assert_valid_response(response, top_k=max(limit, TOP_K))
    if isinstance(top_k, int) and 0 < top_k < 5:
        assert len(response["recommendations"]) <= top_k


def test_respond_without_reset_does_not_raise(agent: ShoppingCopilot) -> None:
    """The harness always calls reset first, but a missed reset must not cost a turn."""
    response = agent.respond("never-reset", "I'm looking for Shirts T-Shirts.", 1, TOP_K)
    assert_valid_response(response)


def test_sessions_are_isolated(agent: ShoppingCopilot) -> None:
    """State from one session must never leak into another."""
    agent.reset("a", {})
    agent.respond("a", "I'm looking for Shirts T-Shirts. A key requirement is: cotton.", 1, TOP_K)
    agent.reset("b", {})
    agent.respond("b", "I'm looking for Women Shoes, but I'm still exploring.", 1, TOP_K)

    state_a = agent._sessions["a"]
    state_b = agent._sessions["b"]
    assert state_a.category != state_b.category
    assert not state_b.constraints, "fresh session must not inherit constraints"


def test_reset_clears_previous_state(agent: ShoppingCopilot) -> None:
    agent.reset("reuse", {})
    agent.respond("reuse", "I'm looking for Shirts T-Shirts. A key requirement is: cotton.", 1, TOP_K)
    assert agent._sessions["reuse"].constraints
    agent.reset("reuse", {})
    assert not agent._sessions["reuse"].constraints
    assert agent._sessions["reuse"].category is None


def test_full_ten_turn_session_stays_valid(agent: ShoppingCopilot) -> None:
    """Drive a session to the hard 10-turn limit with varied replies."""
    agent.reset("long", {"preference_tags": ["fit", "comfort"]})
    replies = [
        "I'm looking for Shirts T-Shirts, but I'm still exploring.",
        "For that, what matters is: cotton; color: black.",
        "I don't have an additional preference for material.",
        "Actually, ignore my earlier preference. What I need is: polyester.",
        "I don't have a preference for size; please use your judgment.",
        "Those options are not quite right yet. Ask me about one specific attribute.",
        "For that, what matters is: Moisture wicking polyester.",
        "",
        "🤷",
        "For that, what matters is: Reflective trim for running.",
    ]
    for turn, message in enumerate(replies, start=1):
        assert_valid_response(agent.respond("long", message, turn, TOP_K))


def test_internal_failure_degrades_instead_of_raising(
    agent: ShoppingCopilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a fault in the ranking path; the turn must still produce a valid answer."""
    import copilot.agent as agent_module

    def explode(*args, **kwargs):
        raise RuntimeError("injected ranking failure")

    monkeypatch.setattr(agent_module, "rank", explode)
    agent.reset("fault", {})
    response = agent.respond("fault", "I'm looking for Shirts T-Shirts.", 1, TOP_K)
    assert_valid_response(response)
    assert response["recommendations"], "fallback must still recommend something"


def test_every_recommended_id_exists_in_the_catalog(agent: ShoppingCopilot) -> None:
    """Invalid IDs are silently dropped by the scorer, so emitting one wastes a slot."""
    agent.reset("valid", {})
    known = set(agent.index.ids)
    for turn in range(1, 4):
        response = agent.respond("valid", "I'm looking for Shirts T-Shirts.", turn, TOP_K)
        for item in response["recommendations"]:
            assert item["parent_asin"] in known


def test_agent_reports_zero_token_usage(agent: ShoppingCopilot) -> None:
    """The agent is fully offline: token usage is zero by construction, not estimate."""
    agent.reset("usage", {})
    response = agent.respond("usage", "I'm looking for Shirts T-Shirts.", 1, TOP_K)
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(DEFAULT_CONFIG, id="default"),
        pytest.param(AgentConfig(clarify_strategy="none"), id="no-clarification"),
        pytest.param(AgentConfig(clarify_strategy="open"), id="open"),
        pytest.param(AgentConfig(clarify_strategy="infogain"), id="infogain"),
        pytest.param(AgentConfig(use_category_lock=False), id="no-category-lock"),
        pytest.param(AgentConfig(use_state_tracking=False), id="no-state"),
        pytest.param(AgentConfig(use_mmr_diversity=True), id="mmr-on"),
        pytest.param(AgentConfig(pad_to_top_k=False), id="no-padding"),
    ],
)
def test_every_configuration_honours_the_contract(
    synthetic_catalog, synthetic_index, config: AgentConfig
) -> None:
    """Ablation configs are run for real numbers, so they must be valid too."""
    configured = ShoppingCopilot(synthetic_catalog, config=config, index=synthetic_index)
    configured.reset("cfg", {"preference_tags": ["fit"]})
    for turn, message in enumerate(
        [
            "I'm looking for Shirts T-Shirts. A key requirement is: cotton.",
            "For that, what matters is: color: black.",
            "Actually, ignore my earlier preference. What I need is: leather.",
        ],
        start=1,
    ):
        assert_valid_response(configured.respond("cfg", message, turn, TOP_K))


def test_invalid_strategy_is_rejected_loudly(synthetic_catalog, synthetic_index) -> None:
    """Configuration errors should fail fast, unlike runtime faults which degrade."""
    with pytest.raises(ValueError, match="clarify_strategy"):
        ShoppingCopilot(
            synthetic_catalog,
            config=AgentConfig(clarify_strategy="nonsense"),
            index=synthetic_index,
        )

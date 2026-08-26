"""Tests for the optional LLM reranking layer.

No network is touched. The point of these tests is not that reranking works when the model
cooperates -- it is that nothing the model or the network does can damage a session. The
agent's contract is that a turn is never forfeited, and this layer is the only part of the
system that can fail for reasons outside the process.
"""

from __future__ import annotations

import json

import pytest

from copilot.agent import ShoppingCopilot
from copilot.config import DEFAULT_CONFIG, AgentConfig
from copilot.llm import ENV_ENABLE, LLMReranker, _apply_order
from copilot.slots import ConversationState


# ------------------------------------------------------------------------- test doubles
class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.input_tokens = prompt
        self.output_tokens = completion


class _Response:
    def __init__(self, payload, prompt=120, completion=30, stop_reason="end_turn") -> None:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        self.content = [_Block(body)]
        self.usage = _Usage(prompt, completion)
        self.stop_reason = stop_reason


class _Messages:
    def __init__(self, outcome) -> None:
        self._outcome = outcome
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FakeClient:
    def __init__(self, outcome) -> None:
        self.messages = _Messages(outcome)


def _reranker(outcome, monkeypatch, *, depth: int = 20) -> LLMReranker:
    """A reranker wired to a fake client, with both gates deliberately open."""
    monkeypatch.setenv(ENV_ENABLE, "1")
    config = AgentConfig(use_llm_rerank=True, llm_rerank_depth=depth)
    reranker = LLMReranker(config)
    reranker._client = _FakeClient(outcome)
    return reranker


def _state() -> ConversationState:
    state = ConversationState(session_id="t", category="boots")
    state.add_constraints(["leather", "platform measures approximately 0.5 inches"], turn=1)
    return state


class _Index:
    """Only ``titles`` is read by the reranker."""

    titles = [f"product {i}" for i in range(50)]


# ------------------------------------------------------------------------------ gating
def test_disabled_by_default():
    assert DEFAULT_CONFIG.use_llm_rerank is False
    assert LLMReranker(DEFAULT_CONFIG).enabled() is False


def test_config_flag_alone_does_not_enable(monkeypatch):
    """The environment gate exists so a scoring run cannot start spending money."""
    monkeypatch.delenv(ENV_ENABLE, raising=False)
    assert LLMReranker(AgentConfig(use_llm_rerank=True)).enabled() is False


def test_environment_variable_alone_does_not_enable(monkeypatch):
    monkeypatch.setenv(ENV_ENABLE, "1")
    assert LLMReranker(AgentConfig(use_llm_rerank=False)).enabled() is False


def test_both_gates_open_enables(monkeypatch):
    monkeypatch.setenv(ENV_ENABLE, "1")
    assert LLMReranker(AgentConfig(use_llm_rerank=True)).enabled() is True


def test_disabled_reranker_makes_no_call_and_reports_zero_usage(monkeypatch):
    monkeypatch.delenv(ENV_ENABLE, raising=False)
    reranker = LLMReranker(AgentConfig(use_llm_rerank=True))
    client = _FakeClient(_Response({"order": [2, 1, 0]}))
    reranker._client = client
    ids, usage = reranker.rerank(_state(), _Index(), [10, 11, 12])
    assert ids == [10, 11, 12]
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}
    assert client.messages.calls == []


def test_unavailable_sdk_is_cached_not_retried(monkeypatch):
    """A misconfigured run must pay the discovery cost once, not on all 600 turns."""
    monkeypatch.setenv(ENV_ENABLE, "1")
    reranker = LLMReranker(AgentConfig(use_llm_rerank=True))
    attempts = []

    def _explode(*_args, **_kwargs):
        attempts.append(1)
        raise RuntimeError("no credentials")

    monkeypatch.setattr("builtins.__import__", _explode)
    assert reranker.available() is False
    assert reranker.available() is False
    assert len(attempts) == 1


# ------------------------------------------------------------------------- happy path
def test_valid_ordering_is_applied(monkeypatch):
    reranker = _reranker(_Response({"order": [2, 0, 1]}), monkeypatch)
    ids, usage = reranker.rerank(_state(), _Index(), [10, 11, 12])
    assert ids == [12, 10, 11]
    assert usage == {"prompt_tokens": 120, "completion_tokens": 30}


def test_only_the_head_is_sent_and_the_tail_is_preserved(monkeypatch):
    reranker = _reranker(_Response({"order": [1, 0]}), monkeypatch, depth=2)
    ids, _ = reranker.rerank(_state(), _Index(), [10, 11, 12, 13])
    assert ids == [11, 10, 12, 13]
    prompt = reranker._client.messages.calls[0]["messages"][0]["content"]
    assert "0. product 10" in prompt and "1. product 11" in prompt
    assert "product 12" not in prompt


def test_request_uses_the_configured_model_and_low_effort(monkeypatch):
    reranker = _reranker(_Response({"order": [0, 1]}), monkeypatch)
    reranker.rerank(_state(), _Index(), [10, 11])
    call = reranker._client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_withdrawn_constraints_are_marked_rather_than_hidden(monkeypatch):
    """The retention finding has to survive into the prompt, or the layer contradicts it."""
    reranker = _reranker(_Response({"order": [0, 1]}), monkeypatch)
    state = _state()
    state.constraints[0].weight = 0.5
    reranker.rerank(state, _Index(), [10, 11])
    prompt = reranker._client.messages.calls[0]["messages"][0]["content"]
    assert "(withdrawn)" in prompt


# ------------------------------------------------------ hostile and degenerate responses
@pytest.mark.parametrize(
    "payload",
    [
        {"order": [0, 0, 0]},                 # repeats
        {"order": [9, 8, 7]},                 # all out of range
        {"order": [0]},                       # incomplete
        {"order": []},                        # empty
        {"order": [-1, 2, 1, 0]},             # negative index
        {"order": ["0", 1.5, None, True]},    # wrong types, including bool-as-int
        {"order": "not a list"},
        {"wrong_key": [0, 1, 2]},
        {},
        [0, 1, 2],                            # not an object
        "plain text, not json at all",
    ],
)
def test_malformed_responses_still_yield_a_full_permutation(payload, monkeypatch):
    """A dropped slot can never hit, so a bad response must not shorten the list."""
    head = [10, 11, 12]
    outcome = _Response(payload) if not isinstance(payload, str) else _Response(payload)
    reranker = _reranker(outcome, monkeypatch)
    ids, _ = reranker.rerank(_state(), _Index(), head)
    assert sorted(ids) == sorted(head)
    assert len(ids) == len(head)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("rate limited"),
        TimeoutError("connection timed out"),
        ValueError("bad request"),
        KeyError("unexpected shape"),
    ],
)
def test_any_exception_returns_the_offline_ordering(error, monkeypatch):
    reranker = _reranker(error, monkeypatch)
    ids, usage = reranker.rerank(_state(), _Index(), [10, 11, 12])
    assert ids == [10, 11, 12]
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_a_refusal_is_treated_as_a_failure(monkeypatch):
    """A refusal is HTTP 200 with no usable ordering, so it must be checked explicitly."""
    reranker = _reranker(_Response({"order": [2, 1, 0]}, stop_reason="refusal"), monkeypatch)
    ids, usage = reranker.rerank(_state(), _Index(), [10, 11, 12])
    assert ids == [10, 11, 12]
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_negative_usage_counts_are_clamped(monkeypatch):
    """The rules require non-negative token counts; never forward a nonsense figure."""
    reranker = _reranker(_Response({"order": [0, 1]}, prompt=-5, completion=-2), monkeypatch)
    _, usage = reranker.rerank(_state(), _Index(), [10, 11])
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_empty_and_single_candidate_lists_make_no_call(monkeypatch):
    reranker = _reranker(_Response({"order": [0]}), monkeypatch)
    assert reranker.rerank(_state(), _Index(), [])[0] == []
    assert reranker.rerank(_state(), _Index(), [10])[0] == [10]
    assert reranker._client.messages.calls == []


# ----------------------------------------------------------------------- _apply_order
def test_apply_order_appends_omitted_candidates_in_original_position():
    assert _apply_order({"order": [2]}, [10, 11, 12]) == [12, 10, 11]


def test_apply_order_rejects_booleans_masquerading_as_indices():
    # bool is a subclass of int; True would otherwise be read as index 1.
    assert _apply_order({"order": [True, False]}, [10, 11]) == [10, 11]


# ------------------------------------------------------------------ agent integration
def test_agent_reports_zero_usage_on_the_default_offline_path(synthetic_catalog):
    agent = ShoppingCopilot(synthetic_catalog, index=None)
    agent.reset("s", {})
    response = agent.respond("s", "I'm looking for T-Shirts, but I'm still exploring.", 1, 10)
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    assert len(response["recommendations"]) > 0


def test_agent_constructs_the_reranker_without_importing_the_sdk(synthetic_index):
    """Constructing an agent must never pull in a third-party package."""
    agent = ShoppingCopilot("unused", index=synthetic_index)
    assert agent._reranker.enabled() is False
    assert agent._reranker._client is None

"""ShoppingCopilot: the conversational retrieval agent.

Implements the challenge's required Agent contract (``reset`` / ``respond``) on top of the
catalog index, the slot state machine, the clarification policy, and the ranking pipeline.

Two invariants drive most of the design:

* **Never waste a turn.** The protocol allows a response to carry a clarification question
  *and* a ranked list simultaneously, so the agent always does both. There is no
  ask-versus-recommend trade-off to manage, and treating it as though there were is what
  caps the starter baseline.
* **Never raise.** The evaluator converts an exception into an empty response, which
  silently burns a turn. Every entry point is therefore wrapped, and any internal failure
  degrades to a still-valid, still-populated answer rather than an empty one.
"""

from __future__ import annotations

from pathlib import Path

from copilot.catalog import CatalogIndex
from copilot.config import DEFAULT_CONFIG, AgentConfig
from copilot.dialogue import (
    _CHROME_TOKENS,
    ASK_MORE,
    BOUNDARY,
    DISCLOSURE,
    NO_ADDITIONAL,
    OVERRIDE,
    parse_opening,
    scan_category,
    parse_reply,
)
from copilot.llm import LLMReranker
from copilot.question import choose_attribute, question_text
from copilot.retrieval import candidate_pool, pad, rank
from copilot.slots import ConversationState

DEFAULT_CATALOG = "data/catalog.jsonl"


class ShoppingCopilot:
    """Stateful, offline, dependency-free conversational shopping agent."""

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG,
        *,
        config: AgentConfig | None = None,
        index: CatalogIndex | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        self.index = index if index is not None else CatalogIndex(catalog_path)
        self._sessions: dict[str, ConversationState] = {}
        # Constructed unconditionally, but inert unless explicitly enabled: it imports no
        # third-party package and opens no connection until the first request it is
        # allowed to make.
        self._reranker = LLMReranker(self.config)

    # ------------------------------------------------------------------ Agent contract
    def reset(self, session_id: str, user_profile: dict) -> None:
        """Begin a new session. Profile is the anonymised aggregate, never raw history."""
        profile = user_profile if isinstance(user_profile, dict) else {}
        self._sessions[session_id] = ConversationState(session_id=session_id, profile=profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            # Degrade to a valid, populated response rather than forfeiting the turn.
            return self._fallback(session_id, top_k)

    # ------------------------------------------------------------------------ internals
    def _state(self, session_id: str) -> ConversationState:
        state = self._sessions.get(session_id)
        if state is None:
            # Defensive: respond() before reset() should not cost a turn.
            state = ConversationState(session_id=session_id)
            self._sessions[session_id] = state
        return state

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._state(session_id)
        state.turn = turn
        message = user_message if isinstance(user_message, str) else ""
        limit = top_k if isinstance(top_k, int) and top_k > 0 else 10

        is_opening = state.category is None and not state.constraints and turn <= 1
        if is_opening:
            self._observe_opening(state, message)
        else:
            self._observe_reply(state, message)

        # Retain the raw payload regardless of how parsing went. Under paraphrase the
        # structured extraction can miss entirely, and dropping the message text is what
        # turns a partial parse failure into a total one.
        if self.config.use_observed_fallback:
            state.observe_text(message, _CHROME_TOKENS)

        pool = candidate_pool(self.index, state, self.config)
        ranked = rank(self.index, state, self.config, limit=limit)
        if self.config.pad_to_top_k:
            ranked = pad(self.index, ranked, pool, limit)

        # Optional semantic reranking. Disabled by default, and a no-op returning zero
        # usage unless both the config flag and COPILOT_LLM are set. Any failure inside
        # returns the offline ordering untouched, so this can never cost a turn.
        ranked, usage = self._reranker.rerank(state, self.index, ranked)

        attribute = choose_attribute(self.index, state, pool, self.config)
        state.pending_attribute = attribute
        if attribute is not None:
            state.asked.append(attribute)

        return {
            "message": self._compose_message(state, attribute, len(ranked)),
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": self.index.ids[doc_id]} for doc_id in ranked[:limit]
            ],
            # Zero by construction on the default offline path -- no model is called. With
            # the reranking layer enabled these are the counts the API actually charged,
            # read from its usage field rather than estimated.
            "usage": usage,
        }

    def _observe_opening(self, state: ConversationState, message: str) -> None:
        opening = parse_opening(message, self.index.bucket_lookup)
        state.category = opening.category
        state.scenario = opening.scenario
        if self.config.use_state_tracking:
            # Opening constraints are always retractable. In an Intent Override session
            # the opening value is a decoy that gets withdrawn; in a Buying session no
            # retraction ever arrives, so the flag is inert. Marking unconditionally
            # avoids having to tell the two openings apart under paraphrase, where that
            # distinction is exactly what stops being reliable.
            state.add_constraints(
                opening.constraints,
                turn=state.turn,
                provisional=self.config.use_override_erasure,
            )

    def _observe_reply(self, state: ConversationState, message: str) -> None:
        if not self.config.use_state_tracking:
            return

        # A category missed at turn 1 (a heavily reworded opener) is still recoverable:
        # the customer keeps naming the product type as the conversation goes on.
        if state.category is None and self.config.use_category_lock:
            found, _ = scan_category(message, self.index.bucket_lookup)
            if found:
                state.category = found

        reply = parse_reply(message)

        if reply.kind == OVERRIDE:
            if self.config.use_override_erasure:
                state.supersede_provisional(self.config.override_decay)
            state.override_applied = True
            state.add_constraints(reply.constraints, turn=state.turn)
            return

        if reply.kind == DISCLOSURE:
            state.add_constraints(reply.constraints, turn=state.turn)
            return

        if reply.kind == NO_ADDITIONAL:
            # That attribute is spent; asking again would burn another turn.
            state.mark_exhausted(reply.attribute or state.pending_attribute)
            return

        if reply.kind == BOUNDARY:
            state.boundary_seen = True
            state.mark_exhausted(reply.attribute or state.pending_attribute)
            return

        if reply.kind == ASK_MORE:
            # We stayed silent last turn and learned nothing. No state change.
            return

    def _compose_message(
        self, state: ConversationState, attribute: str | None, count: int
    ) -> str:
        if count == 0:
            return (
                "I could not find a close match yet. "
                + question_text(attribute or "other", state.turn)
            )
        if attribute is None:
            return "Here are the closest matches I found."
        lead = (
            "Here are my best matches so far."
            if state.active_constraints
            else "Here are some options to start from."
        )
        return f"{lead} {question_text(attribute, state.turn)}"

    def _fallback(self, session_id: str, top_k: int) -> dict:
        """Last-resort response: popular items, still schema-valid, still ten of them."""
        limit = top_k if isinstance(top_k, int) and top_k > 0 else 10
        try:
            state = self._sessions.get(session_id)
            pool: set[int] = set()
            if state is not None and state.category:
                pool = set(self.index.bucket(state.category))
            if len(pool) < limit:
                pool.update(range(min(self.index.count, 2000)))
            ordered = sorted(pool, key=lambda doc: -self.index.popularity(doc))[:limit]
            recommendations = [{"parent_asin": self.index.ids[doc]} for doc in ordered]
        except Exception:
            recommendations = []
        return {
            "message": "Let me try again. Which detail matters most to you?",
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


#: The evaluator imports ``Agent`` and constructs it with the catalog path.
Agent = ShoppingCopilot

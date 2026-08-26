"""Optional LLM semantic reranking. Off by default, and deliberately so.

The Track 4 brief asks for an "LLM Semantic Ranking" stage. This is it -- implemented,
measurable, and disabled for scoring.

That is a choice, not an omission. The submission rules state that "for official final
scoring, organizer policy may disable network access", so an agent that *needs* a key is a
liability rather than a feature. The offline path is the one that gets scored; this layer
exists so the comparison against it is measured rather than assumed, and so the architecture
the brief describes is actually present in the repository.

Three properties matter more than the reranking quality:

* **The default path never imports this module's dependency.** ``anthropic`` is imported
  lazily inside the call, so with the layer disabled the agent remains standard-library
  only and ``requirements-llm.txt`` is never needed.
* **Every failure is silent and total.** A missing key, a missing package, a rate limit, a
  malformed response, a network timeout -- all return the input ordering unchanged. The
  agent's contract is that a turn is never forfeited, and a reranker is the last place that
  should be allowed to break it.
* **Token usage is reported honestly.** The rules require disclosing token usage. With this
  layer off the agent reports zero by construction; with it on it reports what the API
  actually charged, taken from ``response.usage`` rather than estimated.

Enable with ``COPILOT_LLM=1`` in the environment *and* ``use_llm_rerank`` in the config.
Both are required: the environment variable alone will not silently start spending money on
a scoring run.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from copilot.catalog import CatalogIndex
    from copilot.config import AgentConfig
    from copilot.slots import ConversationState

#: Environment switch. The config flag alone is not enough to fire a request.
ENV_ENABLE = "COPILOT_LLM"

#: Titles are truncated before they reach the prompt. A ranking decision needs the
#: distinguishing part of a product name, not the keyword tail that Amazon sellers append.
_TITLE_LIMIT = 110
_CONSTRAINT_LIMIT = 180

_SYSTEM = """You re-rank candidate products for a shopping assistant.

The shopper has disclosed a set of constraints across a conversation. Exactly one candidate
is the product they are looking for. Order the candidates so the most likely target is
first.

Weigh the specific constraints far above the generic ones. A distinctive feature sentence
that names a measurement, a construction detail or a named component is near-identifying; a
material or a colour is shared by thousands of products and separates almost nothing.

A constraint marked (withdrawn) was retracted by the shopper. Treat it as weakened evidence
rather than as a negative signal: it still describes the product they were looking at.

Return every candidate index exactly once."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Candidate indices, best first. Every index exactly once.",
        }
    },
    "required": ["order"],
    "additionalProperties": False,
}

_EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}


class LLMReranker:
    """A reranking stage that is allowed to fail, and does so quietly."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = None
        self._unavailable = False

    # ---------------------------------------------------------------------- gating
    def enabled(self) -> bool:
        """Both switches must agree before a single request is made."""
        return bool(self.config.use_llm_rerank) and os.environ.get(ENV_ENABLE) == "1"

    def available(self) -> bool:
        """True only if a request could actually succeed.

        Resolving the client is what proves the SDK is installed and credentials exist.
        A failure here is cached, so a misconfigured run pays the cost once rather than
        on all 600 turns of an evaluation.
        """
        if not self.enabled() or self._unavailable:
            return False
        if self._client is not None:
            return True
        try:
            import anthropic  # noqa: PLC0415 - deliberately lazy; see module docstring

            # The zero-argument constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN
            # or an `ant auth login` profile, so an unset key does not imply no credentials.
            self._client = anthropic.Anthropic(max_retries=1)
        except Exception:
            self._unavailable = True
            return False
        return True

    # ----------------------------------------------------------------- the request
    def rerank(
        self,
        state: ConversationState,
        index: CatalogIndex,
        doc_ids: list[int],
    ) -> tuple[list[int], dict]:
        """Reorder the head of ``doc_ids``. Returns the ordering and this call's usage.

        Only the first ``llm_rerank_depth`` candidates are sent. The tail is already
        ordered by the offline ranker and is unlikely to contain the target, so paying to
        rank it would buy nothing.
        """
        if not doc_ids or not self.available():
            return doc_ids, dict(_EMPTY_USAGE)

        depth = max(2, int(self.config.llm_rerank_depth))
        head, tail = doc_ids[:depth], doc_ids[depth:]
        if len(head) < 2:
            return doc_ids, dict(_EMPTY_USAGE)

        try:
            payload, usage = self._call(state, index, head)
        except Exception:
            # Rate limit, timeout, refusal, transport error, malformed JSON - all the same
            # from here: keep the offline ordering and carry on.
            return doc_ids, dict(_EMPTY_USAGE)

        return _apply_order(payload, head) + tail, usage

    def _call(
        self, state: ConversationState, index: CatalogIndex, head: list[int]
    ) -> tuple[object, dict]:
        lines = [
            f"{position}. {index.titles[doc_id][:_TITLE_LIMIT]}"
            for position, doc_id in enumerate(head)
        ]
        prompt = (
            f"Product type: {state.category or 'unknown'}\n\n"
            f"Constraints the shopper has disclosed:\n{_describe(state)}\n\n"
            f"Candidates:\n" + "\n".join(lines)
        )

        response = self._client.messages.create(
            model=self.config.llm_model,
            max_tokens=4096,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            # Ranking a short list is not a reasoning-heavy task, and low effort keeps both
            # latency and cost down. Thinking is left at its default rather than disabled:
            # disabling it on Opus 5 risks tool-call and tag leakage into the visible text.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
        )

        # A refusal returns HTTP 200 with no usable ordering, so check before reading.
        if getattr(response, "stop_reason", None) == "refusal":
            raise ValueError("model declined to rank")

        text = next(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return json.loads(text), _usage_of(response)


# --------------------------------------------------------------------------- helpers
def _describe(state: ConversationState) -> str:
    """Render slot memory as prompt text, withdrawn constraints included but marked."""
    lines: list[str] = []
    for item in state.constraints:
        if not item.active:
            continue
        mark = "" if item.weight >= 1.0 else " (withdrawn)"
        lines.append(f"- {item.text[:_CONSTRAINT_LIMIT]}{mark}")
    return "\n".join(lines) if lines else "- (none disclosed yet)"


def _usage_of(response: object) -> dict:
    """Read the charged token counts, defaulting to zero rather than guessing."""
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "output_tokens", 0) or 0
    return {
        "prompt_tokens": int(prompt) if int(prompt) >= 0 else 0,
        "completion_tokens": int(completion) if int(completion) >= 0 else 0,
    }


def _apply_order(payload: object, head: list[int]) -> list[int]:
    """Map a model-supplied ordering back onto document ids, defensively.

    The schema constrains the response shape but not its contents: the model can still
    repeat an index, invent one, or omit several. Anything unusable is dropped and anything
    missing is appended in its original position, so the result is always a permutation of
    ``head`` -- never shorter, never longer, never containing a product that was not a
    candidate. A dropped recommendation slot can never hit, so a malformed response must
    not be able to shrink the list.
    """
    if not isinstance(payload, dict):
        return head
    order = payload.get("order")
    if not isinstance(order, list):
        return head

    seen: set[int] = set()
    result: list[int] = []
    for position in order:
        if not isinstance(position, int) or isinstance(position, bool):
            continue
        if position < 0 or position >= len(head) or position in seen:
            continue
        seen.add(position)
        result.append(head[position])

    result.extend(doc_id for pos, doc_id in enumerate(head) if pos not in seen)
    return result

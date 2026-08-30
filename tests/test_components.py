"""Unit tests for the parsing, state, policy and ranking layers.

These pin down the behaviour the score depends on: that the simulator's templates are
recovered exactly, that reworded versions of the same messages still yield the same
information, and that the ranking signals mean what the design says they mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from copilot import dialogue  # noqa: E402
from copilot.config import DEFAULT_CONFIG  # noqa: E402
from copilot.question import (  # noqa: E402
    OPEN_ATTRIBUTE,
    UNPRODUCTIVE_ATTRIBUTES,
    YIELD_PROBABILITY,
    choose_attribute,
    expected_gain,
)
from copilot.retrieval import candidate_pool, constraint_score, rank, satisfies  # noqa: E402
from copilot.slots import BUDGET, COLOR, MATERIAL, PHRASE, ConversationState, classify_kind  # noqa: E402
from copilot.text import classify_constraint, match_text, terms  # noqa: E402

BUCKETS = {
    "shirts t-shirts": "Shirts T-Shirts",
    "women shoes": "Women Shoes",
    "coats overcoats": "Coats Overcoats",
    "accessories scarves": "Accessories Scarves",
}


# ------------------------------------------------------------------ opening parsing
class TestOpeningParsing:
    def test_buying_opening_exact(self) -> None:
        opening = dialogue.parse_opening(
            "I'm looking for Shirts T-Shirts. A key requirement is: cotton.", BUCKETS
        )
        assert opening.category == "Shirts T-Shirts"
        assert opening.scenario == dialogue.BUYING
        assert opening.constraints == ["cotton"]
        assert opening.exact

    def test_browsing_opening_exact(self) -> None:
        opening = dialogue.parse_opening(
            "I'm looking for Women Shoes, but I'm still exploring.", BUCKETS
        )
        assert opening.category == "Women Shoes"
        assert opening.scenario == dialogue.BROWSING
        assert opening.constraints == []

    def test_override_opening_exact(self) -> None:
        opening = dialogue.parse_opening(
            "I'm looking for Coats Overcoats. I prefer a lighter shell.", BUCKETS
        )
        assert opening.category == "Coats Overcoats"
        assert opening.scenario == dialogue.INTENT_OVERRIDE
        assert opening.constraints == ["i prefer a lighter shell"]

    @pytest.mark.parametrize(
        "message",
        [
            "I need Shirts T-Shirts. It has to be: cotton.",
            "Hi, I want Shirts T-Shirts. One thing I really need: cotton.",
            "um, shopping for shirts t-shirts. must-have: cotton",
            "Looking at Shirts T-Shirts -- key thing is cotton",
        ],
    )
    def test_reworded_buying_openings_still_recover_category(self, message: str) -> None:
        """A paraphraser rewrites chrome, not payload; the category must survive."""
        opening = dialogue.parse_opening(message, BUCKETS)
        assert opening.category == "Shirts T-Shirts"

    @pytest.mark.parametrize(
        "message",
        [
            "Just browsing Women Shoes for now.",
            "I'm after Women Shoes, though I haven't decided yet.",
            "show me some women shoes - still making up my mind",
        ],
    )
    def test_reworded_browsing_openings_recognised(self, message: str) -> None:
        opening = dialogue.parse_opening(message, BUCKETS)
        assert opening.category == "Women Shoes"
        assert opening.scenario == dialogue.BROWSING
        assert opening.constraints == [], "browsing discloses nothing to latch onto"

    def test_category_found_mid_sentence(self) -> None:
        """Scanning n-grams means the category need not be in a fixed position."""
        opening = dialogue.parse_opening(
            "honestly, my sister said Accessories Scarves would suit me", BUCKETS
        )
        assert opening.category == "Accessories Scarves"

    def test_unknown_category_degrades_without_raising(self) -> None:
        opening = dialogue.parse_opening("I'm looking for a hovercraft.", BUCKETS)
        assert opening.category is None

    def test_longest_category_wins(self) -> None:
        buckets = {**BUCKETS, "shirts": "Shirts"}
        opening = dialogue.parse_opening("I'm looking for Shirts T-Shirts.", buckets)
        assert opening.category == "Shirts T-Shirts"


# -------------------------------------------------------------------- reply parsing
class TestReplyParsing:
    def test_disclosure_splits_on_semicolon(self) -> None:
        reply = dialogue.parse_reply("For that, what matters is: cotton; color: black.")
        assert reply.kind == dialogue.DISCLOSURE
        assert reply.constraints == ["cotton", "color: black"]

    def test_disclosure_does_not_split_on_comma(self) -> None:
        """Real constraint strings contain commas; splitting on them shreds the signal."""
        reply = dialogue.parse_reply(
            "For that, what matters is: Lightweight hoops, approximately 2 inches."
        )
        assert reply.constraints == ["lightweight hoops, approximately 2 inches"]

    def test_exhaustion_vs_boundary_are_distinguished(self) -> None:
        exhausted = dialogue.parse_reply("I don't have an additional preference for material.")
        assert exhausted.kind == dialogue.NO_ADDITIONAL
        assert exhausted.attribute == "material"

        boundary = dialogue.parse_reply(
            "I don't have a preference for color; please use your judgment."
        )
        assert boundary.kind == dialogue.BOUNDARY
        assert boundary.attribute == "color"

    def test_override_extracts_replacement(self) -> None:
        reply = dialogue.parse_reply(
            "Actually, ignore my earlier preference. What I need is: leather."
        )
        assert reply.kind == dialogue.OVERRIDE
        assert reply.constraints == ["leather"]

    def test_ask_more_recognised(self) -> None:
        reply = dialogue.parse_reply(
            "Those options are not quite right yet. Ask me about one specific attribute."
        )
        assert reply.kind == dialogue.ASK_MORE

    @pytest.mark.parametrize(
        "message",
        [
            "Scratch that - what I actually need is leather.",
            "Forget what I said. I really need: leather.",
            "changed my mind, i need leather",
        ],
    )
    def test_reworded_override_still_detected(self, message: str) -> None:
        """Retraction must outrank disclosure, or the decoy is never handled."""
        reply = dialogue.parse_reply(message)
        assert reply.kind == dialogue.OVERRIDE
        assert any("leather" in c for c in reply.constraints)

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("No strong feelings about material.", dialogue.NO_ADDITIONAL),
            ("Nothing particular on color.", dialogue.NO_ADDITIONAL),
            ("Up to you on size.", dialogue.BOUNDARY),
            ("No preference on style - your call.", dialogue.BOUNDARY),
        ],
    )
    def test_reworded_refusals_classified(self, message: str, expected: str) -> None:
        reply = dialogue.parse_reply(message)
        assert reply.kind == expected

    def test_reworded_disclosure_keeps_payload(self) -> None:
        reply = dialogue.parse_reply("What I care about there: Moisture wicking polyester.")
        assert reply.kind == dialogue.DISCLOSURE
        assert reply.constraints == ["moisture wicking polyester"]

    def test_pure_filler_yields_nothing(self) -> None:
        """Chrome must not be promoted into a phantom requirement."""
        reply = dialogue.parse_reply("um, well, honestly")
        assert reply.constraints == []


# ------------------------------------------------------------------- constraint types
class TestConstraintTyping:
    @pytest.mark.parametrize(
        ("text", "kind", "value"),
        [
            ("cotton", MATERIAL, "cotton"),
            ("color: black", COLOR, "black"),
            ("blue", COLOR, "blue"),
            ("budget around $24.99", BUDGET, "24.99"),
            ("Reflective trim for running", PHRASE, None),
        ],
    )
    def test_classify_kind(self, text: str, kind: str, value: str | None) -> None:
        assert classify_kind(text) == (kind, value)

    def test_match_text_flattens_separators(self) -> None:
        """Constraints arrive as 'Department: Womens'; the blob stores 'Department Womens'."""
        assert match_text("Department: Womens") == match_text("Department Womens")

    def test_template_chrome_is_not_a_search_term(self) -> None:
        assert terms("I'm looking for shoes, but I'm still exploring.") == ["shoes"]


# ------------------------------------------------------------------------ satisfaction
class TestSatisfaction:
    def _constraint(self, state: ConversationState, text: str):
        return state.add_constraints([text], turn=1)[0]

    def test_exact_phrase_beats_partial_overlap(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        item = self._constraint(state, "Reflective trim for running")
        running = synthetic_index.id_to_doc["B000000002"]
        tee = synthetic_index.id_to_doc["B000000001"]
        assert satisfies(synthetic_index, running, item) == 1.0
        assert satisfies(synthetic_index, tee, item) < 1.0

    def test_first_material_outranks_incidental_mention(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        item = self._constraint(state, "cotton")
        tee = synthetic_index.id_to_doc["B000000001"]
        assert satisfies(synthetic_index, tee, item) == 1.0

    def test_material_mismatch_scores_zero(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        item = self._constraint(state, "silk")
        boot = synthetic_index.id_to_doc["B000000003"]
        assert satisfies(synthetic_index, boot, item) == 0.0

    def test_missing_price_gets_partial_budget_credit(self, synthetic_index) -> None:
        """79% of the catalog has no price; a hard zero would be far too aggressive."""
        state = ConversationState(session_id="s")
        item = self._constraint(state, "budget around $20")
        scarf = synthetic_index.id_to_doc["B000000004"]  # price is None
        assert 0.0 < satisfies(synthetic_index, scarf, item) < 1.0

    def test_closer_price_scores_higher(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        item = self._constraint(state, "budget around $20")
        tee = synthetic_index.id_to_doc["B000000001"]     # 19.99
        coat = synthetic_index.id_to_doc["B000000005"]    # 210.00
        assert satisfies(synthetic_index, tee, item) > satisfies(synthetic_index, coat, item)

    def test_longer_constraints_carry_more_weight(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        state.add_constraints(["cotton", "Ribbed crew neckline"], turn=1)
        tee = synthetic_index.id_to_doc["B000000001"]
        assert constraint_score(synthetic_index, tee, state.active_constraints) == 1.0


# --------------------------------------------------------------------- state machine
class TestStateMachine:
    def test_constraints_accumulate_and_deduplicate(self) -> None:
        state = ConversationState(session_id="s")
        state.add_constraints(["cotton"], turn=1)
        state.add_constraints(["cotton", "color: black"], turn=2)
        assert [c.text for c in state.active_constraints] == ["cotton", "color: black"]

    def test_supersede_retains_value_at_reduced_confidence(self) -> None:
        """The withdrawn preference still describes the target, so it is not erased."""
        state = ConversationState(session_id="s")
        state.add_constraints(["wool"], turn=1, provisional=True)
        state.supersede_provisional(0.5)
        assert [c.text for c in state.active_constraints] == ["wool"]
        assert state.active_constraints[0].weight == 0.5

    def test_full_erasure_moves_to_penalised_set(self) -> None:
        state = ConversationState(session_id="s")
        state.add_constraints(["wool"], turn=1, provisional=True)
        state.supersede_provisional(0.0)
        assert state.active_constraints == []
        assert [c.text for c in state.retracted] == ["wool"]

    def test_erased_terms_get_negative_query_weight(self) -> None:
        state = ConversationState(session_id="s")
        state.category = "Coats Overcoats"
        state.add_constraints(["wool"], turn=1, provisional=True)
        state.supersede_provisional(0.0)
        _, weights = state.query_terms(
            constraint_boost=2.2, category_boost=1.0, decoy_penalty=0.35
        )
        assert weights["wool"] < 0

    def test_category_terms_are_never_penalised(self) -> None:
        """A retracted value that overlaps the category must not poison the category."""
        state = ConversationState(session_id="s")
        state.category = "Coats Overcoats"
        state.add_constraints(["overcoats"], turn=1, provisional=True)
        state.supersede_provisional(0.0)
        _, weights = state.query_terms(
            constraint_boost=2.2, category_boost=1.0, decoy_penalty=0.35
        )
        assert weights["overcoats"] > 0

    def test_observed_tokens_survive_parse_failure(self) -> None:
        state = ConversationState(session_id="s")
        state.observe_text("moisture wicking polyester", dialogue._CHROME_TOKENS)
        assert "polyester" in state.observed
        tokens, weights = state.query_terms(
            constraint_boost=2.2, category_boost=1.0, decoy_penalty=0.35, observed_boost=0.55
        )
        assert weights["polyester"] == 0.55

    def test_confirmed_constraint_outweighs_observed_token(self) -> None:
        state = ConversationState(session_id="s")
        state.observe_text("cotton", dialogue._CHROME_TOKENS)
        state.add_constraints(["cotton"], turn=1)
        _, weights = state.query_terms(
            constraint_boost=2.2, category_boost=1.0, decoy_penalty=0.35, observed_boost=0.55
        )
        assert weights["cotton"] == 2.2


# ------------------------------------------------------------------ question policy
class TestQuestionPolicy:
    def test_unproductive_attributes_are_never_asked(self, synthetic_index) -> None:
        """The simulator's classifier cannot emit these, so asking always wastes a turn."""
        state = ConversationState(session_id="s")
        pool = set(range(synthetic_index.count))
        for attribute in UNPRODUCTIVE_ATTRIBUTES:
            assert expected_gain(synthetic_index, state, pool, attribute) == 0.0
        assert not (UNPRODUCTIVE_ATTRIBUTES & set(YIELD_PROBABILITY))

    def test_budget_is_valued_near_zero_despite_splitting_well(self, synthetic_index) -> None:
        """The failure the expected-value model exists to prevent."""
        state = ConversationState(session_id="s")
        pool = set(range(synthetic_index.count))
        assert expected_gain(synthetic_index, state, pool, "budget") < expected_gain(
            synthetic_index, state, pool, "feature"
        )

    def test_exhausted_attributes_are_dropped(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        pool = set(range(synthetic_index.count))
        state.mark_exhausted("material")
        assert expected_gain(synthetic_index, state, pool, "material") == 0.0

    def test_default_policy_opens_the_floor(self, synthetic_index) -> None:
        """With nothing known, the open question is the highest-value move."""
        state = ConversationState(session_id="s")
        pool = set(range(synthetic_index.count))
        assert choose_attribute(synthetic_index, state, pool, DEFAULT_CONFIG) == OPEN_ATTRIBUTE

    def test_policy_falls_back_when_open_is_exhausted(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        state.mark_exhausted(OPEN_ATTRIBUTE)
        pool = set(range(synthetic_index.count))
        chosen = choose_attribute(synthetic_index, state, pool, DEFAULT_CONFIG)
        assert chosen != OPEN_ATTRIBUTE
        assert chosen not in UNPRODUCTIVE_ATTRIBUTES

    def test_strategy_none_stays_silent(self, synthetic_index) -> None:
        from dataclasses import replace

        state = ConversationState(session_id="s")
        pool = set(range(synthetic_index.count))
        config = replace(DEFAULT_CONFIG, clarify_strategy="none")
        assert choose_attribute(synthetic_index, state, pool, config) is None


# ----------------------------------------------------------------------- retrieval
class TestRetrieval:
    def test_category_lock_restricts_the_pool(self, synthetic_index) -> None:
        from dataclasses import replace

        state = ConversationState(session_id="s")
        state.category = "Shirts T-Shirts"
        config = replace(DEFAULT_CONFIG, min_candidates=1)
        pool = candidate_pool(synthetic_index, state, config)
        assert {synthetic_index.ids[d] for d in pool} == {"B000000001", "B000000002"}

    def test_tiny_category_still_fills_every_slot(self, synthetic_catalog, synthetic_index) -> None:
        """A one-product category must not produce a one-item answer.

        An empty slot can never hit, so a short list forfeits free chances. Padding has
        to be able to reach past the candidate pool into the catalog at large.
        """
        from dataclasses import replace

        from copilot.agent import ShoppingCopilot

        # The confidence gate deliberately truncates an under-informed turn, so it is
        # switched off here: this test is about padding reaching past the candidate pool,
        # and leaving the gate on would test the gate instead. The gate has its own tests.
        config = replace(DEFAULT_CONFIG, use_confidence_gate=False)
        agent = ShoppingCopilot(synthetic_catalog, config=config, index=synthetic_index)
        agent.reset("tiny", {})
        response = agent.respond(
            "tiny", "I'm looking for Accessories Scarves, but I'm still exploring.", 1, 10
        )
        # The synthetic catalog holds only five products, so that is the ceiling here.
        assert len(response["recommendations"]) == min(10, synthetic_index.count)

    def test_constraints_drive_the_ranking(self, synthetic_index) -> None:
        state = ConversationState(session_id="s")
        state.category = "Shirts T-Shirts"
        state.add_constraints(["Moisture wicking polyester"], turn=1)
        ranked = rank(synthetic_index, state, DEFAULT_CONFIG, limit=2)
        assert synthetic_index.ids[ranked[0]] == "B000000002"

    def test_ranking_is_deterministic(self, synthetic_index) -> None:
        """Ties break on a stable key, so repeated runs reproduce exactly."""
        state = ConversationState(session_id="s")
        state.category = "Shirts T-Shirts"
        first = rank(synthetic_index, state, DEFAULT_CONFIG, limit=5)
        second = rank(synthetic_index, state, DEFAULT_CONFIG, limit=5)
        assert first == second

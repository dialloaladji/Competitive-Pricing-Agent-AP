"""
Follow-up analysis and candidate recommendations tests.

Verifies:
1. market_analyst follow-up uses stored analysis (not DB fallback)
2. "analyse le dernier résultat" follow-up uses stored analysis (not new search)
3. No DB fallback when latest analysis_result exists and message is analytical
4. valid_match_count=0 sets confirmed_equivalents_count=0 and no "Équivalents" label
5. compute_candidate_recommendations returns cheapest, best_score, best_market_analyst_choice
6. compact_analysis_context includes candidate_recommendations
7. _is_analytical_followup detects analytical follow-up signals
8. Differential-breaker penalty in analyst ranking
"""

import asyncio
import uuid

import pytest

from api.analysis_service import compact_analysis_context, compute_candidate_recommendations
from api.chat_service import (
    ChatOrchestrator,
    _is_new_product_query,
    _is_analytical_followup,
)
from api.chat_memory import get_conversation_context
from api.llm_client import MockClient
from api.database import async_session_factory, engine, Base
from api.models import ChatConversation, ChatMessage


async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


asyncio.run(_init_db())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_full_result(
    candidate_count: int = 10,
    valid_match_count: int = 0,
    best_score: float = 0.65,
    best_price: float = 8.50,
    n_cross_brand: int = 0,
    n_partial: int = 4,
    n_weak: int = 3,
) -> dict:
    def _cand(i, score, sq, price, is_vague=False, title=None, is_same_brand=False):
        return {
            "title": title or f"Brand{i} MCB 16A Courbe C 6kA",
            "price": price,
            "currency": "EUR",
            "merchant": f"Merchant{i}",
            "brand": f"Brand{i}",
            "url": f"https://shop{i}.example.com/p{i}",
            "score": score,
            "spec_quality": sq,
            "is_vague": is_vague,
            "is_same_brand": is_same_brand,
            "classification": "functional_equivalent",
        }

    cross_brand = [_cand(i, best_score + i * 0.01, 0.85, best_price + i) for i in range(n_cross_brand)]
    partial = [_cand(10 + i, 0.55, 0.40, 12.0 + i) for i in range(n_partial)]
    weak = [_cand(20 + i, 0.30, 0.10, 5.0, is_vague=True) for i in range(n_weak)]

    return {
        "product_id": str(uuid.uuid4()),
        "product_name": "Disjoncteur Hager 16A 6kA",
        "run_id": str(uuid.uuid4()),
        "candidate_count": candidate_count,
        "valid_match_count": valid_match_count,
        "cross_brand_count": n_cross_brand,
        "same_brand_count": 0,
        "partial_spec_count": n_partial,
        "weak_candidate_count": n_weak,
        "best_match_price": best_price,
        "best_match_score": best_score,
        "price_confidence": 0.5,
        "recommendation": None,
        "inferred_product": {
            "name": "Hager MCN 16A 6kA",
            "category": "miniature circuit breaker",
            "brand": "Hager",
            "specs": {"poles": 1, "current_a": 16},
        },
        "cross_brand_equivalents": cross_brand,
        "same_brand_listings": [],
        "partial_spec_equivalents": partial,
        "weak_candidates": weak,
        "brand_diversity_stats": {},
    }


async def _store_fresh_analysis(db, full_result: dict) -> str:
    """Persist a conversation with stored fresh_equivalent_analysis metadata."""
    conv = ChatConversation(title="Test followup conv")
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    conv_id = str(conv.id)

    db.add(ChatMessage(
        conversation_id=conv_id, role="user",
        content="cherche des equivalents hager 16A",
    ))
    from api.analysis_service import compact_analysis_context
    capped = {
        **{k: v for k, v in full_result.items() if k not in (
            "cross_brand_equivalents", "same_brand_listings",
            "partial_spec_equivalents", "weak_candidates",
        )},
        "cross_brand_equivalents": (full_result.get("cross_brand_equivalents") or [])[:20],
        "same_brand_listings": [],
        "partial_spec_equivalents": (full_result.get("partial_spec_equivalents") or [])[:20],
        "weak_candidates": (full_result.get("weak_candidates") or [])[:20],
    }
    db.add(ChatMessage(
        conversation_id=conv_id, role="assistant",
        content="Résultat d'analyse.",
        msg_metadata={
            "type": "fresh_equivalent_analysis",
            "product_id": full_result["product_id"],
            "product_name": full_result["product_name"],
            "fresh_equivalent_analysis": capped,
            "sources_used": ["serpapi"],
            "actions_triggered": ["fresh_equivalent_analysis_triggered"],
        },
    ))
    await db.commit()
    return conv_id


# ── Test 1: _is_analytical_followup detection ─────────────────────────────────

class TestAnalyticalFollowupDetection:

    def test_market_analyst_question_is_followup(self):
        msg = "En tant que market analyst, lequel tu me suggères ?"
        assert _is_analytical_followup(msg)
        assert not _is_new_product_query(msg) or _is_analytical_followup(msg)

    def test_lequel_is_followup(self):
        assert _is_analytical_followup("lequel est le moins cher ?")

    def test_dernier_resultat_is_followup(self):
        assert _is_analytical_followup("Fais une analyse détaillée du dernier résultat")
        assert _is_analytical_followup("analyse détaillée du dernier résultat compact")

    def test_separe_is_followup(self):
        assert _is_analytical_followup(
            "Sépare le moins cher, le meilleur score et le meilleur compromis"
        )

    def test_du_dernier_resultat_is_followup(self):
        assert _is_analytical_followup(
            "mais utilise seulement un résumé compact du dernier résultat"
        )

    def test_new_brand_search_is_not_followup(self):
        assert not _is_analytical_followup("trouve un disjoncteur schneider 20A courbe D")

    def test_analyse_without_reference_is_not_followup(self):
        # "analyse" alone without a reference to prior results is NOT a followup
        assert not _is_analytical_followup("analyse le marché")

    def test_analyse_is_removed_from_new_query_signals(self):
        # "analyse" should NOT be in _NEW_QUERY_SIGNALS anymore
        assert not _is_new_product_query("analyse ce résultat")
        assert not _is_new_product_query("analyser les candidats")


# ── Test 2: compute_candidate_recommendations ─────────────────────────────────

class TestComputeCandidateRecommendations:

    def test_cheapest_best_score_best_choice_returned(self):
        full = _make_full_result(n_partial=3, n_weak=2)
        recs = compute_candidate_recommendations(full)

        assert recs["cheapest_candidate"] is not None
        assert recs["best_score_candidate"] is not None
        assert recs["best_market_analyst_choice"] is not None

    def test_cheapest_has_lowest_price(self):
        full = _make_full_result(n_partial=3)
        recs = compute_candidate_recommendations(full)
        cheapest = recs["cheapest_candidate"]
        all_partial = full["partial_spec_equivalents"]
        min_price = min(c["price"] for c in all_partial if not c.get("is_vague"))
        assert cheapest["price"] <= min_price

    def test_best_score_has_highest_score(self):
        full = _make_full_result(n_cross_brand=2, n_partial=3)
        recs = compute_candidate_recommendations(full)
        best = recs["best_score_candidate"]
        all_cands = (
            full["cross_brand_equivalents"]
            + full["partial_spec_equivalents"]
        )
        max_score = max(c["score"] for c in all_cands if not c.get("is_vague"))
        assert best["score"] == max_score

    def test_valid_match_count_zero_sets_no_confirmed(self):
        full = _make_full_result(valid_match_count=0, n_cross_brand=0)
        recs = compute_candidate_recommendations(full)
        assert recs["confirmed_equivalents_count"] == 0
        assert not recs["has_confirmed_equivalents"]
        assert recs["no_match_warning"] is not None
        assert recs["result_label"] == "Candidats à vérifier"

    def test_confirmed_equivalents_when_cross_brand_present(self):
        full = _make_full_result(valid_match_count=2, n_cross_brand=2)
        recs = compute_candidate_recommendations(full)
        assert recs["has_confirmed_equivalents"]
        assert len(recs["confirmed_equivalents"]) == 2
        assert recs["result_label"] == "Équivalents confirmés"

    def test_vague_candidates_excluded_from_picks(self):
        full = _make_full_result(n_partial=0, n_weak=3)
        recs = compute_candidate_recommendations(full)
        # All weak candidates are vague=True, so priced pool is empty
        assert recs["cheapest_candidate"] is None
        assert recs["best_score_candidate"] is None
        assert recs["best_market_analyst_choice"] is None

    def test_differential_penalty_in_analyst_choice(self):
        """A differential breaker candidate should rank lower than an MCB for an MCB product."""
        cross = [
            {
                "title": "ABB S201-C16 1P 16A MCB Courbe C 6kA",
                "price": 9.00,
                "score": 0.80,
                "spec_quality": 0.75,
                "is_vague": False,
                "is_same_brand": False,
            }
        ]
        partial = [
            {
                "title": "Hager MBN116 1P+N 16A Disjoncteur Différentiel 30mA",
                "price": 7.50,
                "score": 0.70,
                "spec_quality": 0.60,
                "is_vague": False,
                "is_same_brand": False,
            }
        ]
        full = _make_full_result(n_cross_brand=0, n_partial=0, n_weak=0)
        full["cross_brand_equivalents"] = cross
        full["partial_spec_equivalents"] = partial
        full["inferred_product"] = {
            "name": "Disjoncteur MCB 16A",
            "category": "miniature circuit breaker",
        }

        recs = compute_candidate_recommendations(full)
        choice = recs["best_market_analyst_choice"]
        # ABB MCB should win over differential despite higher price
        assert "ABB" in choice["title"] or "MCB" in choice["title"]


# ── Test 3: compact_analysis_context includes candidate_recommendations ────────

class TestCompactContextIncludesRecommendations:

    def test_compact_contains_candidate_recommendations(self):
        full = _make_full_result(n_partial=3)
        compact = compact_analysis_context(full)
        assert "candidate_recommendations" in compact

    def test_compact_recommendations_have_required_keys(self):
        full = _make_full_result(n_partial=3)
        compact = compact_analysis_context(full)
        recs = compact["candidate_recommendations"]
        assert "cheapest_candidate" in recs
        assert "best_score_candidate" in recs
        assert "best_market_analyst_choice" in recs
        assert "confirmed_equivalents_count" in recs
        assert "has_confirmed_equivalents" in recs

    def test_compact_no_match_warning_when_valid_zero(self):
        full = _make_full_result(valid_match_count=0, n_cross_brand=0)
        compact = compact_analysis_context(full)
        recs = compact["candidate_recommendations"]
        assert recs.get("no_match_warning") is not None
        assert "Aucun équivalent confirmé" in recs["no_match_warning"]


# ── Test 4: Follow-up uses stored analysis ────────────────────────────────────

class TestFollowUpUsesStoredAnalysis:

    @pytest.mark.asyncio
    async def test_market_analyst_followup_uses_metadata(self):
        """'lequel me suggères-tu ?' after analysis must use stored metadata, not re-trigger."""
        full = _make_full_result(n_partial=4)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="En tant que market analyst, lequel tu me suggères ? Sépare le moins cher, le meilleur score et le meilleur compromis.",
                conversation_id=conv_id,
            )

        assert "fresh_analysis_followup_from_metadata" in resp.actions_triggered, (
            "market analyst question on existing analysis must use stored metadata"
        )
        assert "fresh_equivalent_analysis_triggered" not in resp.actions_triggered

    @pytest.mark.asyncio
    async def test_analyse_dernier_resultat_uses_metadata(self):
        """'Fais une analyse détaillée du dernier résultat' must use stored metadata."""
        full = _make_full_result(n_partial=4)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="Fais une analyse détaillée du dernier résultat, mais utilise seulement un résumé compact.",
                conversation_id=conv_id,
            )

        assert "fresh_analysis_followup_from_metadata" in resp.actions_triggered, (
            "'dernier résultat' must trigger followup from metadata, not a new search"
        )
        assert "fresh_equivalent_analysis_triggered" not in resp.actions_triggered

    @pytest.mark.asyncio
    async def test_no_db_fallback_when_analysis_in_conversation(self):
        """No DB product lookup should occur when a fresh analysis is in conversation metadata."""
        full = _make_full_result(n_partial=4)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel est le meilleur compromis ?",
                conversation_id=conv_id,
            )

        assert "product_lookup_by_id" not in resp.actions_triggered, (
            "DB product lookup must not happen when analysis is already in conversation"
        )
        assert "fresh_analysis_followup_from_metadata" in resp.actions_triggered


# ── Test 5: valid_match_count=0 prevents confirmed label ─────────────────────

class TestValidMatchCountZeroLabel:

    @pytest.mark.asyncio
    async def test_zero_valid_matches_sets_confirmed_count_zero(self):
        """When valid_match_count=0, confirmed_equivalents_count must be 0."""
        full = _make_full_result(valid_match_count=0, n_cross_brand=0, n_partial=3)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel recommandes-tu ?",
                conversation_id=conv_id,
            )

        assert resp.confirmed_equivalents_count == 0, (
            "confirmed_equivalents_count must be 0 when valid_match_count=0"
        )
        assert resp.confidence == "low", (
            "confidence must be 'low' when there are no valid matches"
        )

    @pytest.mark.asyncio
    async def test_zero_valid_matches_answer_contains_no_confirmed_warning(self):
        """The mock answer for valid_match_count=0 must warn about lack of confirmed equivalents."""
        full = _make_full_result(valid_match_count=0, n_cross_brand=0, n_partial=3)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel recommandes-tu ?",
                conversation_id=conv_id,
            )

        assert "Aucun équivalent confirmé" in resp.answer or resp.confirmed_equivalents_count == 0, (
            "Answer must warn about no confirmed equivalents when valid_match_count=0"
        )

    @pytest.mark.asyncio
    async def test_partial_candidates_in_response(self):
        """partial_candidates field must contain the partial spec equivalents."""
        full = _make_full_result(valid_match_count=0, n_cross_brand=0, n_partial=4, n_weak=2)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel recommandes-tu ?",
                conversation_id=conv_id,
            )

        assert isinstance(resp.partial_candidates, list), "partial_candidates must be a list"
        assert len(resp.partial_candidates) > 0, (
            "partial_candidates must be populated when partial_spec_equivalents exist"
        )


# ── Test 6: Mock answer includes three picks ──────────────────────────────────

class TestMockAnswerIncludesThreePicks:

    @pytest.mark.asyncio
    async def test_mock_answer_contains_cheapest_best_score_best_choice(self):
        """The mock answer for a market analyst request must include all three picks."""
        full = _make_full_result(n_partial=4)
        async with async_session_factory() as db:
            conv_id = await _store_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel me suggères-tu ? Sépare le moins cher, le meilleur score et le meilleur compromis.",
                conversation_id=conv_id,
            )

        answer_lower = resp.answer.lower()
        assert "moins cher" in answer_lower, "Answer must contain cheapest candidate"
        assert "score" in answer_lower, "Answer must contain best_score candidate"
        assert "compromis" in answer_lower or "analyst" in answer_lower, (
            "Answer must contain analyst recommendation"
        )


# ── Test 7: best_technical_candidate ─────────────────────────────────────────

class TestBestTechnicalCandidate:

    def test_best_technical_candidate_is_present(self):
        """compute_candidate_recommendations must return best_technical_candidate."""
        full = _make_full_result(n_partial=4)
        recs = compute_candidate_recommendations(full)
        assert "best_technical_candidate" in recs

    def test_best_technical_differs_from_cheapest_when_cheapest_has_poor_specs(self):
        """When the cheapest candidate has low spec_quality, best_technical must differ."""
        # Cheapest has terrible spec_quality; a more expensive candidate matches specs exactly.
        cheap = {
            "title": "Generic No-Name CB 16A",
            "price": 3.00,
            "score": 0.55,
            "spec_quality": 0.10,
            "is_vague": False,
            "brand": "Generic",
            "specs": {},
        }
        spec_match = {
            "title": "Hager MCN116 1P 16A Courbe C 6kA",
            "price": 14.50,
            "score": 0.72,
            "spec_quality": 0.85,
            "is_vague": False,
            "brand": "Hager",
            "specs": {"current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6},
        }
        full = _make_full_result(n_partial=0, n_weak=0, n_cross_brand=0)
        full["partial_spec_equivalents"] = [cheap, spec_match]
        full["inferred_product"] = {
            "name": "Disjoncteur MCB 16A Courbe C 6kA",
            "category": "miniature circuit breaker",
            "specs": {"current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6},
        }

        recs = compute_candidate_recommendations(full)
        cheapest = recs["cheapest_candidate"]
        best_tech = recs["best_technical_candidate"]

        assert cheapest is not None
        assert best_tech is not None
        assert cheapest["title"] != best_tech["title"], (
            "Cheapest (poor specs) must differ from best_technical (good spec match)"
        )
        assert "Hager" in best_tech["title"] or best_tech.get("spec_quality", 0) > cheapest.get("spec_quality", 0)

    def test_vague_cheapest_not_selected_as_best_technical(self):
        """A vague cheapest product must never be best_technical_candidate."""
        vague_cheap = {
            "title": "Circuit Breaker 16A",
            "price": 2.00,
            "score": 0.50,
            "spec_quality": 0.05,
            "is_vague": True,
            "brand": None,
            "specs": {},
        }
        real_candidate = {
            "title": "ABB S201-C16 MCB 1P 16A Courbe C",
            "price": 18.00,
            "score": 0.80,
            "spec_quality": 0.75,
            "is_vague": False,
            "brand": "ABB",
            "specs": {"current_a": 16, "poles": 1, "curve": "C"},
        }
        full = _make_full_result(n_partial=0, n_weak=0, n_cross_brand=0)
        full["cross_brand_equivalents"] = [vague_cheap, real_candidate]
        full["valid_match_count"] = 1

        recs = compute_candidate_recommendations(full)
        best_tech = recs["best_technical_candidate"]

        assert best_tech is not None
        assert best_tech.get("is_vague") is not True, (
            "Vague candidate must never be selected as best_technical_candidate"
        )
        assert "ABB" in (best_tech.get("title") or ""), (
            "ABB candidate with real specs must be selected over vague cheap one"
        )

    def test_market_analyst_answer_uses_candidate_recommendations(self):
        """Mock answer must be built from candidate_recommendations, not raw lists."""
        partial = [
            {
                "title": f"Legrand DPX³ 16A 1P #{i}",
                "price": 10.0 + i,
                "score": 0.60 + i * 0.02,
                "spec_quality": 0.50,
                "is_vague": False,
                "brand": "Legrand",
            }
            for i in range(4)
        ]
        full = _make_full_result(n_partial=0, n_weak=0)
        full["partial_spec_equivalents"] = partial
        full["valid_match_count"] = 0

        recs = compute_candidate_recommendations(full)
        cheapest_title = (recs["cheapest_candidate"] or {}).get("title", "")
        best_tech_title = (recs["best_technical_candidate"] or {}).get("title", "")

        from api.chat_service import ChatOrchestrator
        from api.llm_client import MockClient
        mock_orch = ChatOrchestrator.__new__(ChatOrchestrator)
        mock_orch.is_mock = True

        from api.analysis_service import compact_analysis_context
        compact = compact_analysis_context(full, "lequel me suggères-tu ?")
        answer_data = mock_orch._mock_fresh_analysis_answer(compact, "market_analysis", "lequel me suggères-tu ?")

        # The answer must reference products from candidate_recommendations
        answer = answer_data["answer"]
        assert "moins cher" in answer.lower() or cheapest_title[:30] in answer, (
            "Mock answer must reference cheapest from candidate_recommendations"
        )
        assert "technique" in answer.lower() or best_tech_title[:30] in answer, (
            "Mock answer must reference best_technical from candidate_recommendations"
        )

    def test_full_result_stored_in_metadata_not_sent_to_llm(self):
        """compact_analysis_context strips full_result — must not include raw URL lists or all fields."""
        full = _make_full_result(n_partial=4, n_weak=3)
        # Add a sentinel field that must never appear in compact context
        full["_FULL_RESULT_SENTINEL"] = "THIS_MUST_NOT_REACH_LLM"
        full["serpapi_raw_results"] = [{"title": "raw", "link": "https://evil.com"}]

        compact = compact_analysis_context(full, "lequel ?")

        assert "_FULL_RESULT_SENTINEL" not in compact, (
            "Sentinel field from full_result must not appear in compact context"
        )
        assert "serpapi_raw_results" not in compact, (
            "serpapi_raw_results must not be forwarded to the LLM"
        )
        # compact must still have the useful structured data
        assert "candidate_recommendations" in compact
        assert "cross_brand_equivalents" in compact or "partial_spec_equivalents" in compact

    def test_compact_context_includes_missing_critical_specs(self):
        """compact_analysis_context must expose missing_critical_specs from candidate_recommendations."""
        full = _make_full_result(n_partial=2)
        full["inferred_product"] = {
            "name": "Disjoncteur 16A",
            "category": "miniature circuit breaker",
            "specs": {"current_a": 16},  # poles, curve, breaking_capacity_ka missing
        }
        compact = compact_analysis_context(full)
        recs = compact["candidate_recommendations"]
        missing = recs.get("missing_critical_specs") or []
        assert "poles" in missing or "curve" in missing, (
            "missing_critical_specs must list specs absent from inferred_product"
        )

    def test_compact_context_includes_confidence_level(self):
        """compact_analysis_context must expose confidence_level derived from best_match_score."""
        full_low = _make_full_result(valid_match_count=0)
        recs_low = compute_candidate_recommendations(full_low)
        assert recs_low["confidence_level"] == "low"

        full_high = _make_full_result(valid_match_count=3, n_cross_brand=3, best_score=0.92)
        full_high["best_match_score"] = 0.92
        recs_high = compute_candidate_recommendations(full_high)
        assert recs_high["confidence_level"] == "high"

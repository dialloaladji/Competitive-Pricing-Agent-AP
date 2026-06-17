"""
Fresh analysis chat architecture tests.

Verifies:
1. compact_analysis_context() produces smaller, correct output
2. _is_new_product_query() detects new vs follow-up messages
3. FRESH_ANALYSIS_INTENTS triggers the new path in process()
4. Follow-up messages reuse stored fresh_equivalent_analysis from metadata
5. New product queries re-run the analysis
6. Full result is stored in metadata; compact is passed to LLM
7. Confidence override works for fresh analysis path
8. process_stream() also follows the fresh analysis path
9. compact_analysis_context() handles empty / missing buckets
10. chat_memory extracts fresh_equivalent_analysis from assistant message metadata
"""

import asyncio
import json
import uuid

import pytest

from api.analysis_service import compact_analysis_context, _compact_candidate
from api.chat_service import (
    ChatOrchestrator,
    FRESH_ANALYSIS_INTENTS,
    _is_new_product_query,
    LOW_CONFIDENCE_SCORE_THRESHOLD,
    EXACT_MATCH_SCORE_THRESHOLD,
)
from api.chat_memory import get_conversation_context
from api.llm_client import MockClient
from api.database import async_session_factory, engine, Base
from api.models import ChatConversation, ChatMessage


# ── DB bootstrap ──────────────────────────────────────────────────────────────

async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


asyncio.run(_init_db())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_full_result(
    candidate_count: int = 5,
    valid_match_count: int = 3,
    best_score: float = 0.91,
    best_price: float = 8.50,
    n_cross_brand: int = 3,
    n_partial: int = 1,
    n_weak: int = 1,
) -> dict:
    """Build a synthetic full analysis result as returned by run_equivalent_analysis_from_text()."""

    def _cand(i: int, score: float, bucket: str) -> dict:
        return {
            "title": f"Brand{i} MCB 16A Courbe C 6kA {'x' * 40}",  # intentionally long
            "price": 8.50 + i,
            "currency": "EUR",
            "merchant": f"Merchant{i}",
            "brand": f"Brand{i}",
            "url": f"https://shop{i}.example.com/p{i}",
            "score": score,
            "price_score": 0.8,
            "relevance_score": 0.7,
            "trust_score": 0.85,
            "spec_quality": 0.9 if bucket != "weak" else 0.1,
            "is_vague": bucket == "weak",
            "classification": "functional_equivalent",
            "spec_match": "poles=1, curve=C, breaking_capacity=6kA",
            "specs": {"poles": 1, "curve": "C"},
            "quality_bucket": bucket,
        }

    cross_brand = [_cand(i, best_score - i * 0.02, "reliable") for i in range(n_cross_brand)]
    partial = [_cand(10 + i, 0.6, "partial") for i in range(n_partial)]
    weak = [_cand(20 + i, 0.3, "weak") for i in range(n_weak)]

    return {
        "product_id": str(uuid.uuid4()),
        "product_name": "Disjoncteur Legrand 16A 6kA",
        "run_id": str(uuid.uuid4()),
        "total_latency_ms": 1234.5,
        "candidate_count": candidate_count,
        "valid_match_count": valid_match_count,
        "cross_brand_count": n_cross_brand,
        "same_brand_count": 0,
        "partial_spec_count": n_partial,
        "weak_candidate_count": n_weak,
        "best_match_price": best_price,
        "best_match_score": best_score,
        "price_confidence": 0.85,
        "recommendation": "Best option: Brand0 at €8.50",
        "brand_diversity_warning": None,
        "brand_diversity_stats": {"needs_supplemental_search": False, "selected_brand_count": 3},
        "inferred_product": {"name": "Disjoncteur 16A", "category": "miniature circuit breaker",
                             "brand": "Legrand", "specs": {"poles": 1, "curve": "C"}},
        "cross_brand_equivalents": cross_brand,
        "same_brand_listings": [],
        "partial_spec_equivalents": partial,
        "weak_candidates": weak,
    }


async def _make_conv_with_fresh_analysis(db, full_result: dict) -> str:
    """Create a conversation that already has a fresh analysis stored in assistant message metadata."""
    conv = ChatConversation(title="Test fresh analysis")
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    conv_id = str(conv.id)

    user_msg = ChatMessage(
        conversation_id=conv_id, role="user",
        content="Trouve des equivalents pour disjoncteur Legrand 16A",
    )
    db.add(user_msg)
    assistant_msg = ChatMessage(
        conversation_id=conv_id, role="assistant",
        content="J'ai trouvé 5 candidats.",
        msg_metadata={
            "type": "fresh_equivalent_analysis",
            "product_id": full_result["product_id"],
            "product_name": full_result["product_name"],
            "fresh_equivalent_analysis": full_result,
            "sources_used": ["serpapi"],
            "actions_triggered": ["fresh_equivalent_analysis_triggered"],
        },
    )
    db.add(assistant_msg)
    await db.commit()
    return conv_id


# ── Test 1: compact_analysis_context produces smaller, correct output ──────────

class TestCompactAnalysisContext:
    def test_compact_strips_internal_fields(self):
        full = _make_full_result()
        compact = compact_analysis_context(full)

        # Fields that should be present
        assert compact["candidate_count"] == full["candidate_count"]
        assert compact["valid_match_count"] == full["valid_match_count"]
        assert compact["best_match_score"] == full["best_match_score"]
        assert compact["best_match_price"] == full["best_match_price"]
        assert compact["recommendation"] == full["recommendation"]

        # Internal/large fields that should NOT be in compact
        assert "total_latency_ms" not in compact
        assert "brand_diversity_stats" not in compact
        assert "run_metadata" not in compact

    def test_compact_candidates_are_stripped(self):
        full = _make_full_result(n_cross_brand=3)
        compact = compact_analysis_context(full)

        cands = compact["cross_brand_equivalents"]
        assert len(cands) == 3
        for c in cands:
            # These internal/large fields should be absent
            assert "price_score" not in c
            assert "relevance_score" not in c
            assert "trust_score" not in c
            assert "quality_bucket" not in c
            # These should be present
            assert "title" in c
            assert "price" in c
            assert "score" in c

    def test_compact_title_truncated_at_80_chars(self):
        full = _make_full_result(n_cross_brand=1)
        compact = compact_analysis_context(full)
        title = compact["cross_brand_equivalents"][0]["title"]
        assert len(title) <= 80

    def test_compact_respects_max_per_bucket(self):
        full = _make_full_result(n_cross_brand=5, n_partial=4, n_weak=3)
        compact = compact_analysis_context(full, max_cross_brand=2, max_partial=1, max_weak=1)
        assert len(compact.get("cross_brand_equivalents", [])) <= 2
        assert len(compact.get("partial_spec_equivalents", [])) <= 1
        assert len(compact.get("weak_candidates", [])) <= 1

    def test_compact_handles_empty_buckets(self):
        full = _make_full_result(n_cross_brand=0, n_partial=0, n_weak=0)
        compact = compact_analysis_context(full)
        assert compact.get("cross_brand_equivalents", []) == []
        assert compact.get("partial_spec_equivalents", []) == []
        assert compact.get("weak_candidates", []) == []
        assert compact["candidate_count"] == full["candidate_count"]

    def test_compact_omits_none_top_level_values(self):
        full = _make_full_result()
        full["brand_diversity_warning"] = None
        compact = compact_analysis_context(full)
        assert "brand_diversity_warning" not in compact


# ── Test 2: _is_new_product_query ─────────────────────────────────────────────

class TestIsNewProductQuery:
    def test_electrical_brand_is_new_query(self):
        assert _is_new_product_query("cherche un disjoncteur schneider 20A") is True

    def test_product_category_is_new_query(self):
        assert _is_new_product_query("disjoncteur 16A courbe C") is True

    def test_followup_question_is_not_new_query(self):
        assert _is_new_product_query("lequel est le moins cher ?") is False
        assert _is_new_product_query("what is the best match?") is False
        assert _is_new_product_query("donne moi plus de détails") is False

    def test_new_search_verb_is_new_query(self):
        assert _is_new_product_query("cherche encore un autre produit") is True
        assert _is_new_product_query("recherche un contacteur 25A") is True


# ── Test 3: FRESH_ANALYSIS_INTENTS set ────────────────────────────────────────

class TestFreshAnalysisIntents:
    def test_contains_expected_intents(self):
        expected = {
            "equivalent_products_search",
            "product_lookup",
            "product_comparison",
            "price_analysis",
            "market_analysis",
            "stock_analysis",
        }
        assert expected == FRESH_ANALYSIS_INTENTS

    def test_price_history_not_in_fresh_intents(self):
        # price_history_analysis uses DB price snapshots, not live search
        assert "price_history_analysis" not in FRESH_ANALYSIS_INTENTS

    def test_general_question_not_in_fresh_intents(self):
        assert "general_question" not in FRESH_ANALYSIS_INTENTS


# ── Test 4: process() takes the fresh analysis path for FRESH_ANALYSIS_INTENTS ─

class TestProcessFreshAnalysisPath:
    @pytest.mark.asyncio
    async def test_fresh_path_triggered_for_equivalent_search(self):
        async with async_session_factory() as db:
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="cherche des equivalents pour disjoncteur Legrand DX3 16A 6kA courbe C",
            )
        assert "fresh_equivalent_analysis_triggered" in resp.actions_triggered
        assert resp.intent == "equivalent_products_search"

    @pytest.mark.asyncio
    async def test_fresh_path_for_price_analysis(self):
        async with async_session_factory() as db:
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="prix disjoncteur schneider easy9 16A",
            )
        assert "fresh_equivalent_analysis_triggered" in resp.actions_triggered
        assert resp.intent == "price_analysis"

    @pytest.mark.asyncio
    async def test_fresh_path_stores_metadata(self):
        """After a fresh analysis, the assistant message metadata must contain fresh_equivalent_analysis."""
        async with async_session_factory() as db:
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="disjoncteur ABB S201 16A courbe C 6kA — trouve equivalents",
            )
            conv_id = resp.conversation_id
            ctx = await get_conversation_context(db, conv_id)

        assert ctx["fresh_equivalent_analysis"] is not None
        fresh = ctx["fresh_equivalent_analysis"]
        assert "candidate_count" in fresh
        assert "cross_brand_equivalents" in fresh
        assert "product_name" in fresh


# ── Test 5: Follow-up uses stored analysis, does not re-run ───────────────────

class TestFollowUpReusesStoredAnalysis:
    @pytest.mark.asyncio
    async def test_followup_uses_metadata_not_serpapi(self):
        """A follow-up question must load stored fresh_equivalent_analysis from metadata."""
        full = _make_full_result()
        async with async_session_factory() as db:
            conv_id = await _make_conv_with_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel est le moins cher ?",
                conversation_id=conv_id,
            )
        assert "fresh_analysis_followup_from_metadata" in resp.actions_triggered
        assert "fresh_equivalent_analysis_triggered" not in resp.actions_triggered

    @pytest.mark.asyncio
    async def test_new_product_query_reruns_analysis(self):
        """A message with product keywords should trigger a fresh analysis even with stored data."""
        full = _make_full_result()
        async with async_session_factory() as db:
            conv_id = await _make_conv_with_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="maintenant cherche un disjoncteur schneider 20A courbe D",
                conversation_id=conv_id,
            )
        assert "fresh_equivalent_analysis_triggered" in resp.actions_triggered


# ── Test 6: Confidence override works on fresh path ───────────────────────────

class TestFreshAnalysisConfidenceOverride:
    @pytest.mark.asyncio
    async def test_low_confidence_when_no_valid_matches(self):
        """valid_match_count=0 must force confidence='low' regardless of LLM output."""
        full = _make_full_result(valid_match_count=0, best_score=0.5, n_cross_brand=0)
        async with async_session_factory() as db:
            conv_id = await _make_conv_with_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="quel est le meilleur ?",
                conversation_id=conv_id,
            )
        assert resp.confidence == "low"

    @pytest.mark.asyncio
    async def test_medium_confidence_with_strong_matches(self):
        """valid_match_count>0 and score>=threshold → confidence can be 'medium'."""
        full = _make_full_result(valid_match_count=3, best_score=0.92, n_cross_brand=3)
        async with async_session_factory() as db:
            conv_id = await _make_conv_with_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            resp = await orch.process(
                message="lequel est le meilleur ?",
                conversation_id=conv_id,
            )
        assert resp.confidence in ("medium", "high")


# ── Test 7: process_stream() also follows the fresh analysis path ─────────────

class TestProcessStreamFreshPath:
    @pytest.mark.asyncio
    async def test_stream_fresh_path_emits_events(self):
        async with async_session_factory() as db:
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            events = []
            async for event in orch.process_stream(
                message="trouve equivalents disjoncteur Hager 16A 6kA",
            ):
                events.append(event)

        types = [e["type"] for e in events]
        assert "thinking" in types
        assert "token" in types
        assert "done" in types

        done_event = next(e for e in events if e["type"] == "done")
        actions = done_event["data"]["actions_triggered"]
        assert "fresh_equivalent_analysis_triggered" in actions

    @pytest.mark.asyncio
    async def test_stream_followup_uses_metadata(self):
        full = _make_full_result()
        async with async_session_factory() as db:
            conv_id = await _make_conv_with_fresh_analysis(db, full)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            events = []
            async for event in orch.process_stream(
                message="quel est le prix le plus bas ?",
                conversation_id=conv_id,
            ):
                events.append(event)

        done_event = next(e for e in events if e["type"] == "done")
        actions = done_event["data"]["actions_triggered"]
        assert "fresh_analysis_followup_from_metadata" in actions
        assert "fresh_equivalent_analysis_triggered" not in actions


# ── Test 8: chat_memory extracts fresh_equivalent_analysis ────────────────────

class TestChatMemoryFreshAnalysis:
    @pytest.mark.asyncio
    async def test_get_conversation_context_returns_fresh_analysis(self):
        full = _make_full_result()
        async with async_session_factory() as db:
            conv_id = await _make_conv_with_fresh_analysis(db, full)
            ctx = await get_conversation_context(db, conv_id)

        assert ctx["fresh_equivalent_analysis"] is not None
        fresh = ctx["fresh_equivalent_analysis"]
        assert fresh["candidate_count"] == full["candidate_count"]
        assert fresh["product_name"] == full["product_name"]
        assert len(fresh["cross_brand_equivalents"]) == len(full["cross_brand_equivalents"])

    @pytest.mark.asyncio
    async def test_get_conversation_context_returns_none_when_no_fresh_analysis(self):
        async with async_session_factory() as db:
            conv = ChatConversation(title="No analysis")
            db.add(conv)
            await db.commit()
            await db.refresh(conv)
            ctx = await get_conversation_context(db, str(conv.id))

        assert ctx["fresh_equivalent_analysis"] is None

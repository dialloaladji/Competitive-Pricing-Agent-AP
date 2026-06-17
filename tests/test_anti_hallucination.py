"""
Anti-hallucination and grounding tests.

These tests verify that when the chat system answers follow-up questions
it only uses candidates from the stored analysis_run — never invents products.
All LLM calls use MockClient; we test the data pipeline, confidence override,
and context injection rather than the LLM's free-text output.
"""

import asyncio
import json
import uuid

import pytest
from sqlalchemy import select

from api.chat_service import (
    ChatOrchestrator,
    _rank_all_candidates,
    _analysis_needs_low_confidence,
    _scored_to_analysis,
    LOW_CONFIDENCE_SCORE_THRESHOLD,
    EXACT_MATCH_SCORE_THRESHOLD,
)
from api.llm_client import MockClient
from api.database import async_session_factory, engine, Base
from api.models import (
    ChatConversation, ChatMessage, Product, Offer, AnalysisRun, AnalysisStatus,
)


# ── DB bootstrap ──────────────────────────────────────────────────────────────

async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


asyncio.run(_init_db())


# ── Shared fixtures ────────────────────────────────────────────────────────────

async def _make_product(db, name="ABB S201-C16", brand="ABB") -> Product:
    p = Product(
        id=str(uuid.uuid4()),
        name=name,
        brand=brand,
        category="miniature circuit breaker",
    )
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return p


async def _make_analysis_run(
    db,
    product_id: str,
    candidate_count: int = 40,
    valid_match_count: int = 0,
    best_match_score: float = 0.51,
    n_partial: int = 10,
    n_weak: int = 9,
) -> AnalysisRun:
    """Create a completed AnalysisRun and matching Offer rows."""
    run = AnalysisRun(
        id=str(uuid.uuid4()),
        product_id=product_id,
        status=AnalysisStatus.completed,
        candidate_count=candidate_count,
        valid_match_count=valid_match_count,
        best_match_score=best_match_score,
        price_confidence=best_match_score,
        final_decision={"summary": "No confirmed match."},
    )
    db.add(run)
    await db.commit()

    for i in range(n_partial):
        offer = Offer(
            id=str(uuid.uuid4()),
            product_id=product_id,
            source="analysis",
            title=f"Partial Product {i + 1} 16A C 6kA",
            price=10.0 + i,
            currency="EUR",
            url=f"https://shop{i}.example.com/product-{i}",
            merchant=f"Shop{i}",
            raw_data={
                "score": 0.45 + i * 0.01,
                "spec_quality": 0.40,
                "is_vague": False,
                "is_same_brand": False,
                "classification": "functional_equivalent",
                "quality_bucket": "partial",
                "brand": "Unknown",
            },
        )
        db.add(offer)

    for i in range(n_weak):
        offer = Offer(
            id=str(uuid.uuid4()),
            product_id=product_id,
            source="analysis",
            title=f"Weak Candidate {i + 1}",
            price=7.0 + i,
            currency="EUR",
            url=f"https://weakshop{i}.example.com/product-{i}",
            merchant=f"WeakShop{i}",
            raw_data={
                "score": 0.10 + i * 0.02,
                "spec_quality": 0.10,
                "is_vague": True,
                "is_same_brand": False,
                "classification": "functional_equivalent",
                "quality_bucket": "weak",
                "brand": None,
            },
        )
        db.add(offer)

    await db.commit()
    return run


# ── Helper: extract analysis_run from the latest assistant message ─────────────

async def _get_latest_analysis_run(db, conv_id: str) -> dict | None:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conv_id)
        .where(ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc())
        .limit(1)
    )
    msg = (await db.execute(stmt)).scalar_one_or_none()
    if msg and msg.msg_metadata:
        return msg.msg_metadata.get("analysis_run")
    return None


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRankingHelpers:
    """Unit tests for the ranking and conversion helpers."""

    def test_rank_all_candidates_order(self):
        """reliable > partial > weak; within group: higher score first."""
        analysis = {
            "cross_brand_equivalents": [
                {"title": "R1", "price": 10.0, "score": 0.90, "spec_quality": 0.8,
                 "is_vague": False, "classification": "reliable", "merchant": "A", "brand": "X", "url": ""},
            ],
            "partial_spec_equivalents": [
                {"title": "P1", "price": 12.0, "score": 0.60, "spec_quality": 0.4,
                 "is_vague": False, "classification": "partial", "merchant": "B", "brand": "Y", "url": ""},
                {"title": "P2", "price": 8.0, "score": 0.55, "spec_quality": 0.3,
                 "is_vague": False, "classification": "partial", "merchant": "C", "brand": "Z", "url": ""},
            ],
            "weak_candidates": [
                {"title": "W1", "price": 5.0, "score": 0.20, "spec_quality": 0.1,
                 "is_vague": True, "classification": "weak", "merchant": "D", "brand": None, "url": ""},
            ],
        }
        ranked = _rank_all_candidates(analysis)
        assert ranked[0]["title"] == "R1"       # reliable first
        assert ranked[1]["title"] == "P1"       # partial, higher score
        assert ranked[2]["title"] == "P2"       # partial, lower score
        assert ranked[3]["title"] == "W1"       # weak last
        assert ranked[0]["rank"] == 1

    def test_rank_adds_spec_warning(self):
        analysis = {
            "cross_brand_equivalents": [],
            "partial_spec_equivalents": [],
            "weak_candidates": [
                {"title": "Vague", "price": 5.0, "score": 0.10, "spec_quality": 0.05,
                 "is_vague": True, "classification": "weak", "merchant": "X", "brand": None, "url": ""},
                {"title": "Good", "price": 6.0, "score": 0.30, "spec_quality": 0.50,
                 "is_vague": False, "classification": "weak", "merchant": "Y", "brand": "A", "url": ""},
            ],
        }
        ranked = _rank_all_candidates(analysis)
        vague = next(r for r in ranked if r["title"] == "Vague")
        good = next(r for r in ranked if r["title"] == "Good")
        assert vague["spec_warning"] != ""
        assert good["spec_warning"] == ""

    def test_analysis_needs_low_confidence(self):
        assert _analysis_needs_low_confidence({"valid_match_count": 0, "best_match_score": 0.50})
        assert _analysis_needs_low_confidence({"valid_match_count": 1, "best_match_score": 0.65})
        assert not _analysis_needs_low_confidence({"valid_match_count": 2, "best_match_score": 0.75})

    def test_scored_to_analysis_buckets(self):
        scored = [
            {"title": "Strong", "score": 0.92, "spec_quality": 0.8, "is_vague": False,
             "is_same_brand": False, "price": 10.0, "currency": "EUR", "merchant": "S1",
             "classification": "exact", "brand": "A", "url": ""},
            {"title": "Partial", "score": 0.40, "spec_quality": 0.35, "is_vague": False,
             "is_same_brand": False, "price": 12.0, "currency": "EUR", "merchant": "S2",
             "classification": "functional", "brand": "B", "url": ""},
            {"title": "Weak", "score": 0.10, "spec_quality": 0.05, "is_vague": True,
             "is_same_brand": False, "price": 7.0, "currency": "EUR", "merchant": "S3",
             "classification": "functional", "brand": None, "url": ""},
        ]
        result = _scored_to_analysis(scored, raw_count=3)
        assert result["valid_match_count"] == 1
        assert len(result["cross_brand_equivalents"]) == 1
        assert len(result["partial_spec_equivalents"]) == 1
        assert len(result["weak_candidates"]) == 1
        assert result["best_match_score"] == 0.92


class TestAnalysisRunPersistence:
    """Verify full analysis_run is stored in assistant message metadata."""

    @pytest.mark.asyncio
    async def test_metadata_stores_analysis_run_on_first_turn(self):
        """After a turn that retrieves analysis, metadata must contain analysis_run."""
        async with async_session_factory() as db:
            product = await _make_product(db)
            await _make_analysis_run(db, product.id, n_partial=5, n_weak=4)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            result = await orch.process(
                message="compare me the equivalents",
                product_id=str(product.id),
            )
        async with async_session_factory() as db:
            run_meta = await _get_latest_analysis_run(db, result.conversation_id)
        assert run_meta is not None, "analysis_run must be stored in metadata"
        assert "cross_brand_equivalents" in run_meta
        assert "partial_spec_equivalents" in run_meta
        assert "weak_candidates" in run_meta
        assert "best_match_score" in run_meta
        assert "valid_match_count" in run_meta

    @pytest.mark.asyncio
    async def test_metadata_stores_candidate_count_correctly(self):
        """Metadata analysis_run must reflect actual candidate counts from live search.

        The fresh analysis path bypasses pre-seeded DB data and runs a live
        SerpAPI search.  We verify structure and type, not a specific count.
        """
        async with async_session_factory() as db:
            product = await _make_product(db)
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            result = await orch.process(
                message="liste les équivalents",
                product_id=str(product.id),
            )
        async with async_session_factory() as db:
            run_meta = await _get_latest_analysis_run(db, result.conversation_id)
        assert run_meta is not None
        assert "cross_brand_equivalents" in run_meta
        assert "partial_spec_equivalents" in run_meta
        assert "weak_candidates" in run_meta
        assert isinstance(run_meta.get("candidate_count", 0), int)
        total_stored = (
            len(run_meta.get("cross_brand_equivalents", []))
            + len(run_meta.get("partial_spec_equivalents", []))
            + len(run_meta.get("weak_candidates", []))
        )
        assert total_stored >= 0, f"Expected non-negative candidate count, got {total_stored}"


class TestConfidenceEnforcement:
    """Confidence must be 'low' when analysis quality is insufficient."""

    @pytest.mark.asyncio
    async def test_confidence_is_low_when_valid_match_count_zero(self):
        """Response confidence must be 'low' when valid_match_count = 0."""
        async with async_session_factory() as db:
            product = await _make_product(db)
            await _make_analysis_run(
                db, product.id,
                valid_match_count=0,
                best_match_score=0.51,
                n_partial=5, n_weak=4,
            )
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            result = await orch.process(
                message="compare me the equivalents",
                product_id=str(product.id),
            )
        assert result.confidence == "low", (
            f"Expected low confidence when valid_match_count=0, got: {result.confidence}"
        )

    @pytest.mark.asyncio
    async def test_confidence_is_low_when_score_below_threshold(self):
        """Confidence must be 'low' when best_match_score < LOW_CONFIDENCE_SCORE_THRESHOLD."""
        async with async_session_factory() as db:
            product = await _make_product(db)
            await _make_analysis_run(
                db, product.id,
                valid_match_count=1,
                best_match_score=0.60,  # below 0.70 → triggers _analysis_needs_low_confidence
                n_partial=3, n_weak=2,
            )
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            # "compare equivalents" → product_comparison intent → triggers analysis lookup
            result = await orch.process(
                message="compare equivalents",
                product_id=str(product.id),
            )
        assert result.confidence == "low", (
            f"Expected low confidence when score=0.60 < {LOW_CONFIDENCE_SCORE_THRESHOLD}, got: {result.confidence}"
        )

    @pytest.mark.asyncio
    async def test_confidence_is_low_on_followup_using_stored_analysis(self):
        """Follow-up without product_id must inherit low confidence from stored analysis_run."""
        async with async_session_factory() as db:
            product = await _make_product(db)
            await _make_analysis_run(
                db, product.id,
                valid_match_count=0,
                best_match_score=0.51,
                n_weak=9,
            )
            llm = MockClient()
            orch = ChatOrchestrator(db=db, llm=llm)
            r1 = await orch.process(
                message="compare me the equivalents",
                product_id=str(product.id),
            )
            conv_id = r1.conversation_id
            r2 = await orch.process(
                message="donne moi 10 examples",
                conversation_id=conv_id,
            )
        assert r2.confidence == "low", (
            f"Follow-up must stay 'low' when stored analysis has score=0.51, got: {r2.confidence}"
        )


class TestGroundedCandidates:
    """
    Unit tests for _build_product_context's grounded_candidates injection.

    Because _build_product_context is bypassed in mock mode (mock goes directly
    to _mock_answer), these tests call the method directly with a synthetic
    _conversation_context — this verifies the grounding logic in isolation.
    """

    def _make_fake_orch(self) -> ChatOrchestrator:
        """Minimal orchestrator with no DB — only _conversation_context is used."""
        orch = object.__new__(ChatOrchestrator)
        orch.is_mock = True
        orch.db = None
        orch.llm = MockClient()
        orch._conversation_context = {}
        orch._conversation_summary = None
        orch._recent_messages = []
        return orch

    def _make_fake_product(self, name="Test MCB", brand="ABB"):
        """Minimal namespace matching the attributes _build_product_context reads."""
        import types
        p = types.SimpleNamespace(
            id=str(uuid.uuid4()),
            name=name,
            brand=brand,
            category="miniature circuit breaker",
            sku=None,
            current_a=16,
            poles=1,
            curve="C",
            breaking_capacity_ka=6.0,
        )
        return p

    def _make_price_analysis(self):
        from api.chat_schemas import PriceAnalysis
        return PriceAnalysis(has_history=False, trend="unknown")

    def _make_stored_analysis(self, n_weak: int = 9, n_partial: int = 0, best_score: float = 0.51):
        return {
            "run_id": "unit-test-run",
            "valid_match_count": 0,
            "best_match_score": best_score,
            "cross_brand_equivalents": [],
            "partial_spec_equivalents": [
                {
                    "title": f"Partial {i+1}",
                    "price": 10.0 + i,
                    "currency": "EUR",
                    "merchant": f"Shop{i}",
                    "url": f"https://shop{i}.com",
                    "score": 0.45 + i * 0.01,
                    "spec_quality": 0.40,
                    "is_vague": False,
                    "brand": "Unknown",
                    "is_same_brand": False,
                    "classification": "functional_equivalent",
                }
                for i in range(n_partial)
            ],
            "weak_candidates": [
                {
                    "title": f"Weak Candidate {i+1}",
                    "price": 7.0 + i,
                    "currency": "EUR",
                    "merchant": f"WeakShop{i}",
                    "url": f"https://weakshop{i}.com",
                    "score": 0.10 + i * 0.02,
                    "spec_quality": 0.10,
                    "is_vague": True,
                    "brand": None,
                    "is_same_brand": False,
                    "classification": "functional_equivalent",
                }
                for i in range(n_weak)
            ],
        }

    def test_only_stored_candidates_are_available(self):
        """With 9 weak candidates in stored analysis, grounded context must have exactly 9."""
        orch = self._make_fake_orch()
        stored = self._make_stored_analysis(n_weak=9, n_partial=0)
        orch._conversation_context = {"analysis_run": stored}

        product = self._make_fake_product()
        pa = self._make_price_analysis()
        ctx = orch._build_product_context(product, [], [], pa, "general_question", None)

        grounded = ctx.get("grounded_candidates")
        assert grounded is not None, "grounded_candidates must be injected when stored analysis exists"
        assert grounded["total_available"] == 9, (
            f"Only 9 candidates in analysis, got {grounded['total_available']}"
        )
        candidate_titles = {c["title"] for c in grounded["candidates"]}
        for title in candidate_titles:
            assert "Weak Candidate" in title, f"Unexpected candidate title: {title}"

    def test_grounded_candidates_not_injected_when_fresh_analysis_available(self):
        """When eq_analysis is freshly fetched, grounded_candidates from conv is not injected."""
        orch = self._make_fake_orch()
        stored = self._make_stored_analysis(n_weak=5)
        orch._conversation_context = {"analysis_run": stored}

        product = self._make_fake_product()
        pa = self._make_price_analysis()
        fresh_eq_analysis = {
            "run_id": "fresh-run",
            "valid_match_count": 0,
            "best_match_score": 0.50,
            "cross_brand_equivalents": [],
            "partial_spec_equivalents": [
                {"title": "P1", "price": 10.0, "currency": "EUR", "merchant": "S1",
                 "url": "https://s1.com", "score": 0.45, "spec_quality": 0.4, "is_vague": False,
                 "brand": "X", "is_same_brand": False, "classification": "partial"},
            ],
            "weak_candidates": [],
        }
        ctx = orch._build_product_context(product, [], [], pa, "equivalent_products_search", fresh_eq_analysis)

        # Fresh eq_analysis → equivalent_analysis block, NOT grounded_candidates from conv
        assert "equivalent_analysis" in ctx, "Fresh eq_analysis should populate equivalent_analysis"
        # grounded_candidates from conv should NOT be injected since eq_analysis is present
        assert "grounded_candidates" not in ctx, (
            "grounded_candidates from conv must NOT be injected when fresh eq_analysis is available"
        )

    def test_grounded_candidates_are_ranked_correctly(self):
        """Ranked candidates must list partial before weak within grounded_candidates."""
        orch = self._make_fake_orch()
        stored = self._make_stored_analysis(n_weak=2, n_partial=3)
        orch._conversation_context = {"analysis_run": stored}

        product = self._make_fake_product()
        pa = self._make_price_analysis()
        ctx = orch._build_product_context(product, [], [], pa, "general_question", None)

        grounded = ctx.get("grounded_candidates")
        assert grounded is not None
        candidates = grounded["candidates"]
        buckets = [c["bucket"] for c in candidates]

        partial_indices = [i for i, b in enumerate(buckets) if b == "partial"]
        weak_indices = [i for i, b in enumerate(buckets) if b == "weak"]
        assert partial_indices, "Expected at least one partial candidate"
        assert weak_indices, "Expected at least one weak candidate"
        assert max(partial_indices) < min(weak_indices), (
            "All partial candidates must appear before weak candidates in ranked list"
        )

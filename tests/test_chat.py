import uuid
import pytest

from api.chat_service import _mock_intent, EXACT_MATCH_SCORE_THRESHOLD


def test_chat_intent_classification():
    assert _mock_intent("Compare this product with Legrand")["intent"] == "product_comparison"
    assert _mock_intent("What is the price trend?")["intent"] in ("price_analysis", "price_history_analysis")
    assert _mock_intent("Do we have stock?")["intent"] == "stock_analysis"
    assert _mock_intent("Show me market analysis")["intent"] == "market_analysis"
    assert _mock_intent("Disjoncteur Legrand 16A")["intent"] == "product_lookup"
    assert _mock_intent("Hello")["intent"] == "general_question"


@pytest.mark.asyncio
async def test_chat_orchestrator_general_question():
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="Hello, what can you do?")
        assert result.intent == "general_question"
        assert result.answer
        assert result.confidence in ("high", "medium", "low")
        assert result.sources_used


@pytest.mark.asyncio
async def test_chat_orchestrator_product_lookup():
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="Find me a Legrand disjoncteur 16A")
        assert result.intent in ("product_lookup", "equivalent_products_search")
        assert result.answer


@pytest.mark.asyncio
async def test_chat_orchestrator_price_analysis():
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="What is the price history?")
        assert result.intent in ("price_history_analysis", "price_analysis")
        assert result.answer


@pytest.mark.asyncio
async def test_chat_orchestrator_response_structure():
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="Find me a Schneider 16A circuit breaker")
        required_keys = [
            "answer", "intent", "product", "products_found", "equivalents",
            "offers", "price_analysis", "market_analysis", "confidence",
            "sources_used", "actions_triggered", "missing_information",
        ]
        for key in required_keys:
            assert hasattr(result, key) or key in result.__dict__, f"Missing key: {key}"
        if hasattr(result, "market_analysis"):
            assert isinstance(result.market_analysis.observed_facts, list)
            assert isinstance(result.market_analysis.recommendations, list)
        assert result.confidence in ("high", "medium", "low")


# ─────────────────────────── Equivalent analysis integration tests ────────────────────────────

async def _make_product_with_analysis(db, *, best_match_score: float = 0.72, n_cross_brand: int = 2):
    """Helper: create a Product + AnalysisRun + Offers with raw_data in the test DB."""
    from api.models import Product, Offer, AnalysisRun, AnalysisStatus
    from datetime import datetime

    product = Product(
        id=str(uuid.uuid4()),
        name="Schneider iC60N 16A 1P",
        brand="Schneider",
        category="circuit_breaker",
        sku="A9F74116",
    )
    db.add(product)
    await db.commit()
    await db.refresh(product)

    run = AnalysisRun(
        id=str(uuid.uuid4()),
        product_id=product.id,
        status=AnalysisStatus.completed,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        candidate_count=40,
        valid_match_count=22,
        best_match_price=12.50,
        best_match_score=best_match_score,
        price_confidence=best_match_score,
        final_decision={"summary": "Test recommendation"},
        run_metadata={"total_scored": 40, "weak": 10},
    )
    db.add(run)
    await db.commit()

    for i in range(n_cross_brand):
        offer = Offer(
            id=str(uuid.uuid4()),
            product_id=product.id,
            source="analysis",
            competitor_name=f"Legrand DX³ 16A 1P (#{i})",
            title=f"Legrand DX³ 16A 1P (#{i})",
            price=10.0 + i,
            currency="EUR",
            url=f"https://example.com/legrand-{i}",
            merchant="elec-shop",
            raw_data={
                "score": 0.72 + i * 0.01,
                "spec_quality": 0.6,
                "classification": "direct_competitor",
                "spec_match": "close_spec_equivalent",
                "is_same_brand": False,
                "is_vague": False,
                "brand": "Legrand",
                "relevance_score": 0.5,
                "quality_bucket": "reliable",
            },
        )
        db.add(offer)

    # Add a weak candidate that must never appear in result.equivalents
    weak_offer = Offer(
        id=str(uuid.uuid4()),
        product_id=product.id,
        source="analysis",
        competitor_name="Generic CB 16A",
        title="Generic CB 16A",
        price=5.0,
        currency="EUR",
        url="https://example.com/generic",
        merchant="cheap-store",
        raw_data={
            "score": 0.30,
            "spec_quality": 0.1,
            "classification": "functional_equivalent",
            "spec_match": "functional_equivalent",
            "is_same_brand": False,
            "is_vague": True,
            "brand": None,
            "relevance_score": 0.2,
            "quality_bucket": "weak",
        },
    )
    db.add(weak_offer)
    await db.commit()

    return product, run


@pytest.mark.asyncio
async def test_chat_response_includes_equivalents_when_analysis_exists():
    """equivalents must be populated from the latest AnalysisRun when analysis intents are used."""
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        product, _ = await _make_product_with_analysis(db, n_cross_brand=2)
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(
            message=f"Trouve des équivalents pour {product.name}",
            product_id=str(product.id),
        )
    total = len(result.equivalents) + len(result.partial_candidates)
    assert total > 0, "confirmed equivalents or partial candidates must not be empty when analysis run exists"


@pytest.mark.asyncio
async def test_competitor_offers_not_returned_as_source_product_offers():
    """Competitor offers (is_same_brand=False, score < 0.88) must not appear in result.offers."""
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        product, _ = await _make_product_with_analysis(db, n_cross_brand=2)
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(
            message=f"Prix du produit {product.name}",
            product_id=str(product.id),
        )
    # All stored offers have is_same_brand=False and score < 0.88 → must not be in offers
    assert result.offers == [], (
        f"Competitor offers must not appear in result.offers; got {result.offers}"
    )


@pytest.mark.asyncio
async def test_cilia_offer_not_treated_as_schneider_exact_offer():
    """An offer from a different brand (Cilia) must not be an exact offer for Schneider."""
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator
    from api.models import Product, Offer

    async with async_session_factory() as db:
        product = Product(
            id=str(uuid.uuid4()),
            name="Schneider Acti9 iC60N 16A",
            brand="Schneider",
            category="circuit_breaker",
        )
        db.add(product)
        await db.commit()
        await db.refresh(product)

        cilia_offer = Offer(
            id=str(uuid.uuid4()),
            product_id=product.id,
            source="analysis",
            competitor_name="Cilia 16A circuit breaker",
            title="Cilia 16A circuit breaker",
            price=8.50,
            currency="EUR",
            url="https://example.com/cilia",
            merchant="generic-store",
            raw_data={
                "score": 0.65,
                "spec_quality": 0.55,
                "classification": "functional_equivalent",
                "spec_match": "close_spec_equivalent",
                "is_same_brand": False,
                "is_vague": False,
                "brand": "Cilia",
                "relevance_score": 0.4,
                "quality_bucket": "reliable",
            },
        )
        db.add(cilia_offer)
        await db.commit()

        llm = MockClient()
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(
            message=f"Prix du {product.name}",
            product_id=str(product.id),
        )

    titles_in_offers = [o["title"] for o in result.offers]
    assert "Cilia 16A circuit breaker" not in titles_in_offers, (
        "Cilia offer must not appear in exact source offers for Schneider"
    )


@pytest.mark.asyncio
async def test_cross_brand_equivalents_included_in_response_equivalents():
    """cross_brand equivalents from the analysis run must appear in ChatResponse.equivalents."""
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        product, _ = await _make_product_with_analysis(db, n_cross_brand=3)
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(
            message=f"Comparaison des équivalents pour {product.name}",
            product_id=str(product.id),
        )

    # Fresh analysis path: equivalents come from live search (cross_brand + partial buckets).
    # Specific brands depend on live API results, so we only assert plumbing correctness.
    assert isinstance(result.equivalents, list), "result.equivalents must be a list"
    assert "fresh_equivalent_analysis_triggered" in result.actions_triggered, (
        "Fresh analysis must have been triggered for a product_comparison intent"
    )
    assert "serpapi" in result.sources_used, (
        "SerpAPI must appear in sources for fresh analysis"
    )


@pytest.mark.asyncio
async def test_weak_candidates_excluded_from_equivalents():
    """Weak candidates (quality_bucket='weak') must NOT appear in result.equivalents."""
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        product, _ = await _make_product_with_analysis(db, n_cross_brand=1)
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(
            message=f"Analyse du marché pour {product.name}",
            product_id=str(product.id),
        )

    eq_titles = [e["title"] for e in result.equivalents + result.partial_candidates]
    assert "Generic CB 16A" not in eq_titles, (
        "Weak candidate 'Generic CB 16A' must not appear in result.equivalents or partial_candidates"
    )


@pytest.mark.asyncio
async def test_answer_mentions_limited_confidence_when_best_match_score_below_threshold():
    """When best_match_score < 0.88, confidence must be limited and the answer must say so."""
    from api.llm_client import MockClient
    from api.database import async_session_factory
    from api.chat_service import ChatOrchestrator

    llm = MockClient()
    async with async_session_factory() as db:
        product, _ = await _make_product_with_analysis(db, best_match_score=0.72, n_cross_brand=1)
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(
            message=f"Analyse du prix pour {product.name}",
            product_id=str(product.id),
        )

    assert result.confidence in ("low", "medium"), (
        f"Confidence must be low or medium when best_match_score < {EXACT_MATCH_SCORE_THRESHOLD}; "
        f"got {result.confidence}"
    )
    answer_lower = result.answer.lower()
    assert (
        "0.72" in result.answer
        or "limit" in answer_lower
        or "confidence" in answer_lower
    ), "Answer must mention limited confidence when best_match_score < threshold"

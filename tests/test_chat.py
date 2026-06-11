import pytest

from api.chat_service import _mock_intent


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

import asyncio
import uuid
import pytest
from sqlalchemy import select, func

from api.chat_service import ChatOrchestrator
from api.llm_client import MockClient
from api.database import async_session_factory, engine, Base
from api.models import ChatConversation, ChatMessage, Product, Offer


async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


asyncio.run(_init_db())


@pytest.mark.asyncio
async def test_chat_creates_conversation():
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="What is the price of this product?")
        assert result.conversation_id is not None
        assert result.message_id is not None
        conv = await db.get(ChatConversation, result.conversation_id)
        assert conv is not None
        assert conv.title == "What is the price of this product?"


@pytest.mark.asyncio
async def test_chat_reuses_conversation():
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result1 = await orch.process(message="Hello, tell me about products")
        conv_id = result1.conversation_id
        assert conv_id is not None
        result2 = await orch.process(
            message="What about circuit breakers?",
            conversation_id=conv_id,
        )
        assert result2.conversation_id == conv_id
        count_stmt = (
            select(func.count(ChatMessage.id))
            .where(ChatMessage.conversation_id == conv_id)
        )
        count = (await db.execute(count_stmt)).scalar() or 0
        assert count == 4


@pytest.mark.asyncio
async def test_messages_persisted():
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="Show me Legrand products")
        conv_id = result.conversation_id
        assert conv_id is not None
        msgs_stmt = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conv_id)
            .order_by(ChatMessage.created_at.asc())
        )
        msgs = (await db.execute(msgs_stmt)).scalars().all()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Show me Legrand products"
        assert msgs[1].role == "assistant"
        assert len(msgs[1].content) > 0


@pytest.mark.asyncio
async def test_conversation_not_found():
    from api.chat_service import ChatOrchestrator
    llm = MockClient()
    async with async_session_factory() as db:
        conv = await db.get(ChatConversation, "00000000-0000-0000-0000-000000000000")
        assert conv is None


@pytest.mark.asyncio
async def test_list_conversations():
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        await orch.process(message="Test conversation for listing")
        count_stmt = select(func.count(ChatConversation.id))
        total = (await db.execute(count_stmt)).scalar() or 0
        assert total >= 1


@pytest.mark.asyncio
async def test_delete_conversation():
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="Conversation to delete")
        conv_id = result.conversation_id
        assert conv_id is not None
        conv = await db.get(ChatConversation, conv_id)
        assert conv is not None
        await db.delete(conv)
        await db.commit()
        conv = await db.get(ChatConversation, conv_id)
        assert conv is None


# ─────────────── conversation continuation tests ───────────────

async def _make_product_with_exact_offers(db) -> Product:
    product = Product(
        id=str(uuid.uuid4()),
        name="Legrand DX3 16A 1P C",
        brand="Legrand",
        category="circuit_breaker",
        sku="407801",
    )
    db.add(product)
    await db.commit()
    await db.refresh(product)
    offer = Offer(
        id=str(uuid.uuid4()),
        product_id=product.id,
        source="shop",
        title="Legrand DX3 16A 1P C",
        price=12.50,
        currency="EUR",
        url="https://example.com/leg",
        merchant="elec-pro",
        raw_data={
            "score": 0.92,
            "is_same_brand": True,
            "classification": "exact_match",
        },
    )
    db.add(offer)
    await db.commit()
    return product


@pytest.mark.asyncio
async def test_first_request_returns_conversation_and_message_id():
    """A request with no conversation_id must return both conversation_id and message_id."""
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        result = await orch.process(message="Show me available circuit breakers")
    assert result.conversation_id is not None
    assert result.message_id is not None


@pytest.mark.asyncio
async def test_null_conversation_id_always_creates_new_conversation():
    """Each request without conversation_id must create a distinct conversation."""
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        r1 = await orch.process(message="Hello")
        r2 = await orch.process(message="Hello again")
    assert r1.conversation_id != r2.conversation_id


@pytest.mark.asyncio
async def test_second_request_loads_conversation_context():
    """Follow-up request must be in the same conversation with all messages persisted."""
    llm = MockClient()
    async with async_session_factory() as db:
        orch = ChatOrchestrator(db=db, llm=llm)
        r1 = await orch.process(message="Tell me about Legrand breakers")
        conv_id = r1.conversation_id
        r2 = await orch.process(
            message="Are these really comparable?",
            conversation_id=conv_id,
        )
    assert r2.conversation_id == conv_id
    async with async_session_factory() as db:
        count = (await db.execute(
            select(func.count(ChatMessage.id))
            .where(ChatMessage.conversation_id == conv_id)
        )).scalar()
    assert count == 4


@pytest.mark.asyncio
async def test_product_id_reused_from_conversation_context():
    """When product_id is absent in a follow-up, it must be reused from the conversation."""
    llm = MockClient()
    async with async_session_factory() as db:
        product = await _make_product_with_exact_offers(db)
        orch = ChatOrchestrator(db=db, llm=llm)
        r1 = await orch.process(
            message=f"Tell me about {product.name}",
            product_id=str(product.id),
        )
        conv_id = r1.conversation_id
        r2 = await orch.process(
            message="What is the best offer?",
            conversation_id=conv_id,
        )
    assert r2.conversation_id == conv_id
    assert "product_id_reused_from_conversation" in r2.actions_triggered


@pytest.mark.asyncio
async def test_previous_offers_available_in_followup_response():
    """A follow-up without product_id must still return the product's offers via reused context."""
    llm = MockClient()
    async with async_session_factory() as db:
        product = await _make_product_with_exact_offers(db)
        orch = ChatOrchestrator(db=db, llm=llm)
        r1 = await orch.process(
            message=f"Prix du {product.name}",
            product_id=str(product.id),
        )
        assert len(r1.offers) > 0, "First response must have offers"
        conv_id = r1.conversation_id
        r2 = await orch.process(
            message="Which one is best?",
            conversation_id=conv_id,
        )
    assert r2.conversation_id == conv_id
    assert len(r2.offers) > 0, "Follow-up response must still include offers via reused product_id"


@pytest.mark.asyncio
async def test_followup_question_uses_previous_offers():
    """'which one is best?' after an offer search must reference the same product and its offers."""
    llm = MockClient()
    async with async_session_factory() as db:
        product = await _make_product_with_exact_offers(db)
        orch = ChatOrchestrator(db=db, llm=llm)
        r1 = await orch.process(
            message=f"Trouve les offres pour {product.name}",
            product_id=str(product.id),
        )
        conv_id = r1.conversation_id
        r2 = await orch.process(
            message="which one is best?",
            conversation_id=conv_id,
        )
    assert r2.product_id == str(product.id)
    assert r2.answer


@pytest.mark.asyncio
async def test_assistant_message_metadata_persisted():
    """Assistant messages must store product_id, offers, and sources in their metadata."""
    llm = MockClient()
    async with async_session_factory() as db:
        product = await _make_product_with_exact_offers(db)
        orch = ChatOrchestrator(db=db, llm=llm)
        r1 = await orch.process(
            message=f"Prix du {product.name}",
            product_id=str(product.id),
        )
        msg = await db.get(ChatMessage, r1.message_id)
        assert msg is not None
        meta = msg.msg_metadata or {}
        assert meta.get("product_id") == str(product.id)
        assert isinstance(meta.get("offers"), list)
        assert isinstance(meta.get("sources_used"), list)

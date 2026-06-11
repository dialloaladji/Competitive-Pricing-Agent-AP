import asyncio
import pytest
from sqlalchemy import select, func

from api.chat_service import ChatOrchestrator
from api.llm_client import MockClient
from api.database import async_session_factory, engine, Base
from api.models import ChatConversation, ChatMessage


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

import json
import logging
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import ChatConversation, ChatMessage, ChatMemorySummary

logger = logging.getLogger("api.chat.memory")


async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: str | None,
    user_id: str | None,
    first_message: str,
) -> tuple[ChatConversation, str]:
    if conversation_id:
        conv = await db.get(ChatConversation, conversation_id)
        if conv:
            return conv, "existing"
    conv = ChatConversation(
        title=first_message[:80].rstrip(",;. ") if first_message else "Chat",
        user_id=user_id,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv, "created"


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> ChatMessage:
    msg = ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        msg_metadata=metadata,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def load_recent_messages(
    db: AsyncSession,
    conversation_id: str,
    limit: int = 10,
) -> list[ChatMessage]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    msgs = list(result.scalars())
    msgs.reverse()
    return msgs


async def get_conversation_context(
    db: AsyncSession,
    conversation_id: str,
) -> dict:
    recent = await load_recent_messages(db, conversation_id, limit=10)
    summary = await load_conversation_summary(db, conversation_id)

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .where(ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
    )
    result = await db.execute(stmt)
    assistant_msgs = list(result.scalars())

    product_id = None
    product_name = None
    offers: list = []
    equivalents: list = []
    price_analysis = None

    for msg in assistant_msgs:
        meta = msg.msg_metadata or {}
        if product_id is None and meta.get("product_id"):
            product_id = meta["product_id"]
            product_name = meta.get("product_name")
        if not offers and meta.get("offers"):
            offers = meta["offers"]
        if not equivalents and meta.get("equivalents"):
            equivalents = meta["equivalents"]
        if price_analysis is None and meta.get("price_analysis"):
            price_analysis = meta["price_analysis"]
        if product_id and offers and equivalents:
            break

    return {
        "summary": summary.summary if summary else None,
        "recent_messages": recent,
        "product_id": product_id,
        "product_name": product_name,
        "offers": offers,
        "equivalents": equivalents,
        "price_analysis": price_analysis,
    }


async def load_conversation_summary(
    db: AsyncSession,
    conversation_id: str,
) -> ChatMemorySummary | None:
    stmt = (
        select(ChatMemorySummary)
        .where(ChatMemorySummary.conversation_id == conversation_id)
        .order_by(ChatMemorySummary.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def build_chat_context(
    system_prompt: str,
    conversation_summary: str | None,
    recent_messages: list[ChatMessage],
    product_context: dict,
    current_message: str,
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_summary:
        messages.append({"role": "system", "content": f"Conversation summary: {conversation_summary}"})
    if product_context:
        messages.append({
            "role": "system",
            "content": f"Verified database context: {json.dumps(product_context)}",
        })
    for m in recent_messages:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": current_message})
    return messages


async def maybe_update_conversation_summary(
    db: AsyncSession,
    conversation_id: str,
    llm,
) -> None:
    count_stmt = select(func.count(ChatMessage.id)).where(ChatMessage.conversation_id == conversation_id)
    total = (await db.execute(count_stmt)).scalar() or 0
    if total <= 20:
        return
    all_msgs_stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
    )
    all_msgs = (await db.execute(all_msgs_stmt)).scalars().all()
    if len(all_msgs) <= 10:
        return
    to_summarize = all_msgs[:-10]
    recent = all_msgs[-10:]
    text = "\n".join(f"{m.role}: {m.content[:200]}" for m in to_summarize)
    summary_prompt = f"Summarize this conversation about electrical products. Keep it brief:\n\n{text}"
    try:
        resp = await llm.chat("You summarize conversations.", summary_prompt)
        summary_text = resp.get("content", text[:500])
    except Exception:
        summary_text = text[:500]
    existing = await load_conversation_summary(db, conversation_id)
    last_id = str(recent[0].id) if recent else None
    if existing:
        existing.summary = summary_text
        existing.covered_until_message_id = last_id
        existing.updated_at = datetime.utcnow()
    else:
        db.add(ChatMemorySummary(
            conversation_id=conversation_id,
            summary=summary_text,
            covered_until_message_id=last_id,
        ))
    await db.commit()

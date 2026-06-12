import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.llm_client import get_llm_client
from api.chat_schemas import ChatRequest, ChatResponse
from api.chat_service import ChatOrchestrator
from api.models import ChatConversation, ChatMessage, ChatMemorySummary

logger = logging.getLogger("api.chat")
router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    llm = get_llm_client()
    try:
        orchestrator = ChatOrchestrator(db=db, llm=llm)
        result = await orchestrator.process(
            message=data.message,
            product_id=data.product_id,
            conversation_id=data.conversation_id,
            user_id=data.user_id,
        )
        return result
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat processing failed")
    finally:
        await llm.close()


@router.post("/chat/stream")
async def chat_stream_endpoint(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """SSE streaming endpoint. Each event is a JSON line prefixed with 'data: '."""
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    llm = get_llm_client()

    async def event_generator():
        try:
            orch = ChatOrchestrator(db=db, llm=llm)
            async for event in orch.process_stream(
                message=data.message,
                product_id=data.product_id,
                conversation_id=data.conversation_id,
                user_id=data.user_id,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        finally:
            await llm.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ChatConversation).order_by(ChatConversation.updated_at.desc()).limit(50)
    result = await db.execute(stmt)
    conversations = []
    for conv in result.scalars():
        last_msg_stmt = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conv.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
        last = (await db.execute(last_msg_stmt)).scalar_one_or_none()
        conversations.append({
            "conversation_id": str(conv.id),
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "last_message_preview": last.content[:100] if last else None,
        })
    return conversations


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(ChatConversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs_stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
    )
    msgs = (await db.execute(msgs_stmt)).scalars().all()
    summary_stmt = (
        select(ChatMemorySummary)
        .where(ChatMemorySummary.conversation_id == conversation_id)
        .order_by(ChatMemorySummary.created_at.desc())
        .limit(1)
    )
    summary = (await db.execute(summary_stmt)).scalar_one_or_none()
    return {
        "conversation_id": str(conv.id),
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content[:500],
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ],
        "summary": summary.summary if summary else None,
    }


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(ChatConversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()
    return {"message": "Conversation deleted"}

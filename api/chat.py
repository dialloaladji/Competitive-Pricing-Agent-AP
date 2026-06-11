import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.llm_client import get_llm_client
from api.chat_schemas import ChatRequest, ChatResponse
from api.chat_service import ChatOrchestrator

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
        )
        return result
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat processing failed")
    finally:
        await llm.close()

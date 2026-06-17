import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

_PLACEHOLDER_STR = frozenset({"", "string", "n/a", "unknown", "null", "none", "undefined"})

def _clean_str(val: str | None) -> str | None:
    if val is None or val.strip().lower() in _PLACEHOLDER_STR:
        return None
    return val.strip()

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_db, async_session_factory, engine, Base
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AgentLog, AnalysisStatus
from api.schemas import (
    ProductUpdate, ProductOut, OfferOut,
    AnalysisRunOut, PriceSnapshotOut, MetricsSummary,
    AnalyzeResponse, DashboardSummary,
    EquivalentRequest, EquivalentOut, AnalyzeEquivalentsResponse,
)

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","service":"api","message":"%(message)s"}')
logger = logging.getLogger("api")

redis_client: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    try:
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception:
        logger.warning("Redis unavailable — running without cache")
        redis_client = None

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ensured")

    yield

    if redis_client:
        await redis_client.close()
    await engine.dispose()


app = FastAPI(title="Competitive Pricing Agent API", version="1.0.0", lifespan=lifespan)

from api.chat import router as chat_router
app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if redis_client and request.url.path not in ["/health", "/metrics-summary"]:
        ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{ip}"
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, 60)
        if count > settings.rate_limit_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."},
                                headers={"Retry-After": "60"})
    response = await call_next(request)
    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    status = {"status": "healthy", "database": "unknown", "redis": "unknown"}
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        status["database"] = "connected"
    except Exception as e:
        status["database"] = f"error: {str(e)}"
        status["status"] = "degraded"
    if redis_client:
        try:
            await redis_client.ping()
            status["redis"] = "connected"
        except Exception:
            status["redis"] = "disconnected"
            status["status"] = "degraded"
    else:
        status["redis"] = "not configured"
    return status


# ── Metrics ─────────────────────────────────────────────────────────────

@app.get("/metrics-summary")
async def metrics_summary(db: AsyncSession = Depends(get_db)):
    total_p = (await db.execute(select(func.count(Product.id)))).scalar() or 0
    tracked_p = (await db.execute(select(func.count(Product.id)).where(Product.is_tracked == True))).scalar() or 0
    total_o = (await db.execute(select(func.count(Offer.id)))).scalar() or 0
    total_a = (await db.execute(select(func.count(AnalysisRun.id)))).scalar() or 0
    total_s = (await db.execute(select(func.count(PriceSnapshot.id)))).scalar() or 0
    today_a = (await db.execute(
        select(func.count(AnalysisRun.id)).where(AnalysisRun.created_at >= datetime.utcnow().date())
    )).scalar() or 0
    avg_lat = (await db.execute(
        select(func.avg(AnalysisRun.total_latency_ms)).where(AnalysisRun.status == "completed")
    )).scalar()
    completed = (await db.execute(
        select(AnalysisRun.candidate_count, AnalysisRun.valid_match_count)
        .where(AnalysisRun.status == "completed")
    )).all()
    match_rate = None
    no_match_rate = None
    if completed:
        total_candidates = sum(r[0] or 0 for r in completed)
        total_matches = sum(r[1] or 0 for r in completed)
        if total_candidates > 0:
            match_rate = round(total_matches / total_candidates, 3)
            no_match_count = sum(1 for r in completed if (r[1] or 0) == 0)
            no_match_rate = round(no_match_count / len(completed), 3)
    avg_conf = (await db.execute(
        select(func.avg(AnalysisRun.price_confidence)).where(AnalysisRun.status == "completed")
    )).scalar()
    return MetricsSummary(
        total_products=total_p, tracked_products=tracked_p, total_offers=total_o,
        total_analyses=total_a, total_snapshots=total_s, analyses_today=today_a,
        avg_latency_ms=round(avg_lat, 2) if avg_lat else None,
        match_rate=match_rate, no_match_rate=no_match_rate,
        avg_confidence=round(avg_conf, 2) if avg_conf else None,
    )


# ── Products ────────────────────────────────────────────────────────────

@app.get("/api/v1/products")
async def list_products(tracked: bool | None = None, page: int = 1, size: int = 20, db: AsyncSession = Depends(get_db)):
    q = select(Product).order_by(Product.created_at.desc())
    if tracked is not None:
        q = q.where(Product.is_tracked == tracked)
    q = q.offset((page - 1) * size).limit(size)
    result = await db.execute(q)
    return [ProductOut.model_validate(p) for p in result.scalars()]


@app.get("/api/v1/products/{product_id}")
async def get_product(product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    return ProductOut.model_validate(product)


@app.put("/api/v1/products/{product_id}")
async def update_product(product_id: str, data: ProductUpdate, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(product, k, v)
    await db.commit()
    await db.refresh(product)
    return ProductOut.model_validate(product)


@app.delete("/api/v1/products/{product_id}")
async def delete_product(product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    await db.delete(product)
    await db.commit()
    return {"message": "Product deleted"}


# ── Analysis ────────────────────────────────────────────────────────────

@app.post("/api/v1/products/{product_id}/analyze", status_code=202)
async def analyze_product(product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    run = AnalysisRun(product_id=product_id, status=AnalysisStatus.pending, run_metadata={"trigger": "api"})
    db.add(run)
    await db.commit()
    await db.refresh(run)

    from worker.celery_app import celery_app
    celery_app.send_task("analyze_product", args=[str(product_id), str(run.id)])
    return AnalyzeResponse(run_id=str(run.id), product_id=product_id, status="pending",
                           message="Analysis started")


@app.get("/api/v1/products/{product_id}/analysis/latest")
async def get_latest_analysis(product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    run = await db.execute(
        select(AnalysisRun)
        .where(AnalysisRun.product_id == product_id,
               AnalysisRun.status == AnalysisStatus.completed)
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )
    run = run.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "No completed analysis found for this product")
    offers = await db.execute(
        select(Offer).where(Offer.product_id == product_id).order_by(Offer.price)
    )
    offers = offers.scalars().all()
    return {
        "run_id": str(run.id),
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "total_latency_ms": run.total_latency_ms,
        "candidate_count": run.candidate_count,
        "valid_match_count": run.valid_match_count,
        "best_match_price": run.best_match_price,
        "best_match_score": run.best_match_score,
        "price_confidence": run.price_confidence,
        "recommendation": (run.final_decision or {}).get("summary") if run.final_decision else None,
        "offers": [
            {
                "title": o.title,
                "price": o.price,
                "currency": o.currency,
                "merchant": o.merchant,
                "url": o.url,
            }
            for o in offers
        ],
    }


@app.get("/api/v1/analysis/{run_id}")
async def get_analysis_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(AnalysisRun, run_id)
    if not run:
        raise HTTPException(404, "Analysis run not found")
    product_id = str(run.product_id)
    offers = await db.execute(
        select(Offer).where(Offer.product_id == product_id).order_by(Offer.price)
    )
    offers = offers.scalars().all()
    return {
        "run_id": str(run.id),
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "total_latency_ms": run.total_latency_ms,
        "candidate_count": run.candidate_count,
        "valid_match_count": run.valid_match_count,
        "best_match_price": run.best_match_price,
        "best_match_score": run.best_match_score,
        "price_confidence": run.price_confidence,
        "error_message": run.error_message,
        "recommendation": (run.final_decision or {}).get("summary") if run.final_decision else None,
        "offers": [
            {
                "title": o.title,
                "price": o.price,
                "currency": o.currency,
                "merchant": o.merchant,
                "url": o.url,
            }
            for o in offers
        ],
    }


@app.post("/api/v1/products/analyze-equivalents")
async def analyze_equivalents(data: EquivalentRequest, db: AsyncSession = Depends(get_db)):
    from api.llm_client import get_llm_client
    from api.analysis_service import run_equivalent_analysis_from_text
    from worker.scoring import _is_electrical, ELECTRICAL_BRANDS

    product_text = f"{data.description} {data.brand or ''} {data.sku or ''}"
    if not _is_electrical(product_text):
        raise HTTPException(
            status_code=400,
            detail=(
                "This API is specialized for electrical products only. "
                "Please submit an electrical product (circuit breaker, contactor, switch, "
                "cable, electrical panel, etc.) from a brand such as "
                f"{', '.join(ELECTRICAL_BRANDS[:6])}."
            ),
        )

    llm = get_llm_client()
    logger.info(f"LLM client: {type(llm).__name__}, mock_mode={settings.mock_mode}")
    try:
        result = await run_equivalent_analysis_from_text(
            query=data.description,
            db=db,
            persist=True,
            llm=llm,
            name=_clean_str(data.name),
            brand=_clean_str(data.brand),
            category=_clean_str(data.category),
            sku=_clean_str(data.sku),
            voltage_v=data.voltage_v,
            current_a=data.current_a,
            poles=data.poles,
            curve=_clean_str(data.curve),
            breaking_capacity_ka=data.breaking_capacity_ka,
            target_price=data.target_price,
            currency=data.currency or "EUR",
        )
    except Exception as e:
        import traceback
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        await llm.close()

    def _to_out(s: dict) -> EquivalentOut:
        return EquivalentOut(
            title=s.get("title", ""),
            price=s.get("price", 0),
            currency=s.get("currency", "EUR"),
            merchant=s.get("merchant"),
            brand=s.get("brand"),
            url=s.get("url", ""),
            score=s.get("score", 0),
            price_score=s.get("price_score", 0),
            relevance_score=s.get("relevance_score", 0),
            trust_score=s.get("trust_score", 0),
            spec_quality=s.get("spec_quality", 0.0),
            is_vague=s.get("is_vague", False),
            classification=s.get("classification", "unknown"),
            spec_match=s.get("spec_match", "functional_equivalent"),
            specs=s.get("specs", {}),
        )

    partial_list = result.get("partial_spec_equivalents", [])[:10]
    weak_list = result.get("weak_candidates", [])[:10]

    return AnalyzeEquivalentsResponse(
        product_id=result["product_id"],
        product_name=result["product_name"],
        run_id=result["run_id"],
        total_latency_ms=result["total_latency_ms"],
        candidate_count=result["candidate_count"],
        valid_match_count=result["valid_match_count"],
        cross_brand_count=result["cross_brand_count"],
        same_brand_count=result["same_brand_count"],
        weak_candidate_count=result["weak_candidate_count"],
        best_match_price=result["best_match_price"],
        best_match_score=result["best_match_score"],
        price_confidence=result["price_confidence"],
        recommendation=result["recommendation"],
        cross_brand_equivalents=[_to_out(s) for s in result.get("cross_brand_equivalents", [])],
        same_brand_listings=[_to_out(s) for s in result.get("same_brand_listings", [])],
        weak_candidates=[_to_out(s) for s in weak_list],
        partial_spec_equivalents=[_to_out(s) for s in partial_list],
        partial_spec_count=result["partial_spec_count"],
        brand_diversity_warning=result["brand_diversity_warning"],
        brand_diversity_stats=result["brand_diversity_stats"],
        inferred_product=result["inferred_product"],
    )
@app.get("/api/v1/products/{product_id}/offers")
async def get_offers(product_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Offer).where(Offer.product_id == product_id).order_by(Offer.price)
    )
    return [OfferOut.model_validate(o) for o in result.scalars()]


# ── Price History ───────────────────────────────────────────────────────

@app.get("/api/v1/products/{product_id}/price-history")
async def get_price_history(product_id: str, days: int = 30, db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(PriceSnapshot).where(PriceSnapshot.product_id == product_id, PriceSnapshot.snapshot_date >= since)
        .order_by(PriceSnapshot.snapshot_date)
    )
    return [PriceSnapshotOut.model_validate(s) for s in result.scalars()]


# ── Dashboard ───────────────────────────────────────────────────────────

@app.get("/api/v1/dashboard/summary")
async def dashboard_summary(db: AsyncSession = Depends(get_db)):
    total_p = (await db.execute(select(func.count(Product.id)))).scalar() or 0
    tracked_p = (await db.execute(select(func.count(Product.id)).where(Product.is_tracked == True))).scalar() or 0
    total_o = (await db.execute(select(func.count(Offer.id)))).scalar() or 0
    total_a = (await db.execute(select(func.count(AnalysisRun.id)))).scalar() or 0
    avg_conf = (await db.execute(
        select(func.avg(AnalysisRun.price_confidence)).where(AnalysisRun.status == "completed")
    )).scalar()

    best_drops = await db.execute(
        select(Product.name, AnalysisRun.best_match_price, AnalysisRun.created_at)
        .join(AnalysisRun, AnalysisRun.product_id == Product.id)
        .where(AnalysisRun.status == "completed", AnalysisRun.best_match_price.isnot(None))
        .order_by(AnalysisRun.best_match_price).limit(5)
    )

    recent = await db.execute(
        select(AnalysisRun).where(AnalysisRun.status == "completed")
        .order_by(AnalysisRun.created_at.desc()).limit(10)
    )

    return DashboardSummary(
        total_products=total_p, tracked_products=tracked_p,
        total_offers=total_o, total_analyses=total_a,
        avg_confidence=round(avg_conf, 2) if avg_conf else None,
        best_price_drops=[{"name": r[0], "price": r[1], "date": str(r[2])} for r in best_drops],
        recent_analyses=[{"id": str(r.id), "product_id": str(r.product_id),
                          "score": r.best_match_score, "confidence": r.price_confidence,
                          "date": str(r.created_at)} for r in recent.scalars()],
    )

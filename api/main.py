import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import redis.asynced as aioredis
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_db, async_session_factory, engine, Base
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AgentLog, AnalysisStatus
from api.schemas import (
    ProductCreate, ProductUpdate, ProductOut, OfferOut,
    AnalysisRunOut, PriceSnapshotOut, MetricsSummary,
    AnalyzeResponse, DashboardSummary,
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


@app.post("/api/v1/products", status_code=201)
async def create_product(data: ProductCreate, db: AsyncSession = Depends(get_db)):
    product = Product(**data.model_dump())
    db.add(product)
    await db.commit()
    await db.refresh(product)
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
    run = AnalysisRun(product_id=product_id, status=AnalysisStatus.pending, metadata={"trigger": "api"})
    db.add(run)
    await db.commit()
    await db.refresh(run)

    from worker.celery_app import celery_app
    celery_app.send_task("analyze_product", args=[str(product_id), str(run.id)])
    return AnalyzeResponse(run_id=str(run.id), product_id=product_id, status="pending",
                           message="Analysis started")


@app.get("/api/v1/products/{product_id}/analysis")
async def get_analysis_history(product_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AnalysisRun).where(AnalysisRun.product_id == product_id)
        .order_by(AnalysisRun.created_at.desc()).limit(20)
    )
    return [AnalysisRunOut.model_validate(r) for r in result.scalars()]


@app.get("/api/v1/products/{product_id}/analysis/latest")
async def get_latest_analysis(product_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AnalysisRun).where(AnalysisRun.product_id == product_id)
        .order_by(AnalysisRun.created_at.desc()).limit(1)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "No analysis found for this product")
    return AnalysisRunOut.model_validate(run)


@app.get("/api/v1/analysis/{run_id}")
async def get_analysis_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(AnalysisRun, run_id)
    if not run:
        raise HTTPException(404, "Analysis run not found")
    return AnalysisRunOut.model_validate(run)


# ── Offers ──────────────────────────────────────────────────────────────

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

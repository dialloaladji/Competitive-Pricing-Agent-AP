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


@app.get("/api/v1/products/{product_id}/analysis/{run_id}")
async def get_analysis_run(product_id: str, run_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    run = await db.get(AnalysisRun, run_id)
    if not run or str(run.product_id) != product_id:
        raise HTTPException(404, "Analysis run not found")
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
    from worker.tasks import (
        agent_serpapi_search, normalize_candidates, score_candidates, agent_summarizer,
    )
    from api.llm_client import get_llm_client
    from worker.scoring import (
        _is_electrical, ELECTRICAL_BRANDS, _infer_product_attributes,
        split_by_quality, diversify_by_brand,
    )
    from api.schemas import VALID_CLASSIFICATIONS

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

    start_total = time.time()
    inferred = _infer_product_attributes(
        description=data.description,
        brand=_clean_str(data.brand),
        category=_clean_str(data.category),
        name=_clean_str(data.name),
    )
    name = _clean_str(data.name) or inferred["name"]
    category = _clean_str(data.category) or inferred["category"]
    brand = _clean_str(data.brand) or inferred["brand"] or None
    sku = _clean_str(data.sku)
    curve = _clean_str(data.curve) or inferred["specs"].get("curve")
    inferred_specs = inferred["specs"]

    product = Product(
        name=name,
        description=data.description,
        category=category,
        brand=brand,
        sku=sku,
        target_price=data.target_price,
        currency=data.currency,
        voltage_v=data.voltage_v or inferred_specs.get("voltage_v"),
        current_a=data.current_a or inferred_specs.get("current_a"),
        poles=data.poles or inferred_specs.get("poles"),
        curve=curve,
        breaking_capacity_ka=data.breaking_capacity_ka or inferred_specs.get("breaking_capacity_ka"),
        mounting=inferred_specs.get("mounting"),
    )
    db.add(product)
    await db.commit()
    await db.refresh(product)

    run = AnalysisRun(product_id=product.id, status=AnalysisStatus.running,
                      started_at=datetime.utcnow(), run_metadata={"trigger": "api_sync"})
    db.add(run)
    await db.commit()
    await db.refresh(run)

    run_id = str(run.id)
    product_id = str(product.id)
    llm = get_llm_client()
    logger.info(f"LLM client: {type(llm).__name__}, mock_mode={settings.mock_mode}")

    try:
        queries = [data.description[:200]]
        logger.info(f"Searching SerpApi with: {queries[0][:80]}...")

        scored = []
        raw_candidates = []
        try:
            serpapi_results = await agent_serpapi_search(queries, run_id, iteration=1)
            raw_candidates = serpapi_results[:60]
        except Exception as e:
            logger.warning(f"SerpApi search failed: {e}")

        if raw_candidates:
            normalized = normalize_candidates(raw_candidates)
            logger.info(f"{len(raw_candidates)} raw → {len(normalized)} normalized")
            scored = score_candidates(product, normalized)
            logger.info(f"{len(normalized)} normalized → {len(scored)} scored")
        else:
            scored = []

        total_latency = (time.time() - start_total) * 1000
        target_currency = product.currency or "EUR"
        reliable_scored, partial_scored, weak_scored = split_by_quality(
            scored, product, target_currency=target_currency,
        )
        logger.info(f"Reliable: {len(reliable_scored)}, Partial: {len(partial_scored)}, Weak: {len(weak_scored)}")

        analysis = {}
        recommendation = None
        if len(scored) >= 3:
            llm = get_llm_client()
            summary = await agent_summarizer(llm, product, scored, run_id)
            analysis = summary.get("analysis", {})
            recommendation = analysis.get("summary")
        else:
            analysis = {
                "market_overview": f"Only {len(scored)} equivalent(s) found.",
                "recommendation": "Insufficient data for pricing recommendation. Need at least 3 equivalents (spec_quality >= 0.5, non-vague, with valid price).",
                "confidence": 0.2,
                "below_threshold": True,
            }
            recommendation = analysis["recommendation"]

        best_source = reliable_scored if reliable_scored else scored
        reliable_cross_brand = [s for s in best_source if not s.get("is_same_brand", False)]
        if reliable_cross_brand:
            best = reliable_cross_brand[0]
        else:
            best = best_source[0] if best_source else None

        run.status = AnalysisStatus.completed
        run.completed_at = datetime.utcnow()
        run.total_latency_ms = total_latency
        run.candidate_count = len(raw_candidates)
        run.valid_match_count = len(reliable_scored)
        run.best_match_price = best["price"] if best else None
        run.best_match_score = best["score"] if best else None
        run.price_confidence = analysis.get("confidence")
        run.final_decision = analysis
        run.run_metadata = {"total_scored": len(scored), "weak": len(weak_scored)}
        await db.commit()

        for s in scored:
            offer = Offer(
                product_id=product.id,
                source="analysis",
                competitor_name=s.get("title", "")[:255],
                title=s.get("title", ""),
                price=s.get("price", 0),
                currency=s.get("currency", "USD"),
                url=s.get("url", ""),
                merchant=s.get("merchant"),
            )
            db.add(offer)
        if best:
            snap = PriceSnapshot(
                product_id=product.id,
                price=best["price"],
                currency=best.get("currency", "USD"),
            )
            db.add(snap)
        await db.commit()

    except Exception as e:
        run.status = AnalysisStatus.failed
        run.error_message = str(e)
        run.completed_at = datetime.utcnow()
        await db.commit()
        import traceback
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        await llm.close()

    cross_brand_reliable = [s for s in reliable_scored if not s.get("is_same_brand", False)]
    cross_brand_list, cross_brand_overflow, diversity_stats = diversify_by_brand(
        cross_brand_reliable, max_per_brand=2, max_total=5, min_brand_count=3,
    )
    same_brand_list = [s for s in reliable_scored if s.get("is_same_brand", False)][:2]
    weak_list = weak_scored[:10]
    partial_list = partial_scored[:10]

    brand_diversity_warning = None
    if diversity_stats["needs_supplemental_search"]:
        brand_diversity_warning = (
            f"Only {diversity_stats['selected_brand_count']} distinct brand(s) found in cross-brand equivalents "
            f"(target: >= {diversity_stats.get('min_brand_count', 3)}). Consider running a supplemental brand-targeted search."
        )
        logger.warning(brand_diversity_warning)

    def _to_out(s: dict) -> EquivalentOut:
        return EquivalentOut(
            title=s["title"], price=s["price"], currency=s.get("currency", "EUR"),
            merchant=s.get("merchant"), brand=s.get("brand"), url=s.get("url", ""),
            score=s["score"], price_score=s["price_score"],
            relevance_score=s["relevance_score"], trust_score=s["trust_score"],
            spec_quality=s.get("spec_quality", 0.0),
            is_vague=s.get("is_vague", False),
            classification=s.get("classification", "unknown"),
            spec_match=s.get("spec_match", "functional_equivalent"),
            specs=s.get("specs", {}),
        )

    final_specs = {k: v for k, v in {
        "voltage_v": product.voltage_v,
        "current_a": product.current_a,
        "poles": product.poles,
        "curve": product.curve,
        "breaking_capacity_ka": product.breaking_capacity_ka,
        "mounting": product.mounting,
    }.items() if v is not None}
    inferred_product = {
        "name": product.name,
        "category": product.category,
        "brand": product.brand,
        "specs": final_specs,
    }

    return AnalyzeEquivalentsResponse(
        product_id=product_id,
        product_name=product.name,
        run_id=run_id,
        total_latency_ms=total_latency,
        candidate_count=run.candidate_count or 0,
        valid_match_count=len(reliable_scored),
        cross_brand_count=len(cross_brand_list),
        same_brand_count=len(same_brand_list),
        weak_candidate_count=len(weak_scored),
        best_match_price=best["price"] if best else None,
        best_match_score=best["score"] if best else None,
        price_confidence=analysis.get("confidence"),
        recommendation=recommendation,
        cross_brand_equivalents=[_to_out(s) for s in cross_brand_list],
        same_brand_listings=[_to_out(s) for s in same_brand_list],
        weak_candidates=[_to_out(s) for s in weak_list],
        partial_spec_equivalents=[_to_out(s) for s in partial_list],
        partial_spec_count=len(partial_list),
        brand_diversity_warning=brand_diversity_warning,
        brand_diversity_stats=diversity_stats,
        inferred_product=inferred_product,
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

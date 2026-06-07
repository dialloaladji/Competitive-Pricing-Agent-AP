import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import async_session_factory
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AnalysisStatus, AgentLog
from api.llm_client import get_llm_client, search_tavily, search_serpapi
from worker.celery_app import celery_app

logger = logging.getLogger("worker")


async def _create_agent_log(db: AsyncSession, run_id: str, agent_name: str, data: dict):
    log = AgentLog(run_id=run_id, agent_name=agent_name, **data)
    db.add(log)
    await db.commit()


async def _update_analysis(db: AsyncSession, run_id: str, **kwargs):
    run = await db.get(AnalysisRun, run_id)
    if run:
        for k, v in kwargs.items():
            setattr(run, k, v)
        await db.commit()


# ───────────────────────────── Search ─────────────────────────────

async def agent_serpapi_search(queries: list[str], run_id: str, iteration: int) -> list[dict]:
    start = time.time()
    tasks = [search_serpapi(q, settings.web_search_max_results) for q in queries]
    results_per_query = await asyncio.gather(*tasks, return_exceptions=True)
    all_results: list[dict] = []
    for r in results_per_query:
        if isinstance(r, Exception):
            logger.warning(f"SerpApi error in parallel batch: {r}")
            continue
        all_results.extend(r)

    latency = (time.time() - start) * 1000
    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0,
        "model_name": "serpapi", "prompt_version": "2.0", "json_parse_success": True,
        "log_metadata": {"query_count": len(queries), "result_count": len(all_results), "parallel": True},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "serpapi_search", log_data)
    return all_results


# ───────────────────────────── Deterministic Normalizer ─────────────────────────────

def normalize_candidates(raw_candidates: list[dict]) -> list[dict]:
    from worker.scoring import _extract_brand_from_title, _extract_specs_from_title
    seen_urls: set[str] = set()
    normalized: list[dict] = []
    for c in raw_candidates:
        url = (c.get("url") or "").strip()
        if url and url in seen_urls:
            continue
        seen_urls.add(url)

        title = (c.get("title") or "").strip()
        if not title:
            continue

        raw_price = c.get("price") or ""
        price = 0.0
        if isinstance(raw_price, (int, float)):
            price = float(raw_price)
        elif isinstance(raw_price, str):
            m = __import__("re").search(r"([\d]+(?:[.,]\d+)?)", raw_price.replace(" ", ""))
            if m:
                price = float(m.group(1).replace(",", "."))

        brand = c.get("brand") or _extract_brand_from_title(title)
        specs = _extract_specs_from_title(title)
        merchant = (c.get("merchant") or c.get("source") or "").strip()
        currency = c.get("currency") or "EUR"

        normalized.append({
            "title": title,
            "price": price,
            "currency": currency,
            "merchant": merchant,
            "brand": brand,
            "url": url,
            "source": c.get("source", "google_shopping"),
            "specs": specs,
        })
    return normalized


# ───────────────────────────── Scoring Engine (fully deterministic) ─────────────────────────────

TIER1_BRANDS = {"abb", "schneider", "schneider electric", "legrand",
                "siemens", "eaton", "hager"}


def _tier1_brand_boost(title: str | None) -> float:
    if not title:
        return 0.0
    title_lower = title.lower()
    for brand in TIER1_BRANDS:
        if brand in title_lower:
            return 0.10
    return 0.0


def _spec_match_boost(spec_match: str | None) -> float:
    SPEC_MATCH_BOOST = {
        "exact_spec_equivalent": 0.15,
        "close_spec_equivalent": 0.05,
        "functional_equivalent": 0.0,
    }
    return SPEC_MATCH_BOOST.get(spec_match or "", 0.0)


def _spec_mismatch_penalty(product: Any, candidate: dict) -> float:
    target_poles = getattr(product, "poles", None)
    cand_specs = candidate.get("specs", {}) or {}
    cand_poles = cand_specs.get("poles")
    if target_poles == 1 and cand_poles == 2:
        return -0.05
    if target_poles == 3 and cand_poles == 4:
        return -0.05
    return 0.0


MERCHANT_TRUST = {
    "rexel": 0.9, "sonepar": 0.9, "legallais": 0.9,
    "manomano": 0.85, "leroy merlin": 0.85, "domomat": 0.85,
    "123elec": 0.85, "achat electrique": 0.85, "one-elec": 0.85,
    "amazon": 0.9, "ebay": 0.6, "cdiscount": 0.8,
    "screwfix": 0.85, "bricolage direct": 0.8, "blanc habitat": 0.75,
    "brandstock": 0.7, "e-altamira": 0.8, "matériels-electriques": 0.85,
    "matyco": 0.7, "punto luce": 0.7, "heiz24": 0.7,
    "kelelek": 0.8, "habitatmat": 0.75, "brico-travo": 0.8,
    "elec 44": 0.85, "mon comptoir digital": 0.75, "eibabo": 0.8,
    "maison moderne electricite": 0.8,
}


def score_candidates(
    product: Any,
    normalized: list[dict],
) -> list[dict]:
    from worker.scoring import (
        deterministic_pre_score, spec_quality_score, _extract_brand_from_title,
    )

    scored = []
    for i, c in enumerate(normalized):
        pre = deterministic_pre_score(product, c)
        sq_score, sq_breakdown = spec_quality_score(product, c)
        is_vague = bool(sq_breakdown.get("is_vague", False))

        brand_boost = _tier1_brand_boost(c.get("title"))
        merchant_name = (c.get("merchant") or "").lower()
        trust_score = 0.5
        for key, val in MERCHANT_TRUST.items():
            if key in merchant_name:
                trust_score = val
                break

        raw_price = c.get("price", 0) or 0
        price = float(raw_price) if raw_price else 0
        target = product.target_price or price or 1
        price_ratio = price / target if target > 0 else 1.0
        price_score = max(0, 1.0 - abs(1.0 - price_ratio) * 0.5)

        det_score = pre.get("deterministic_score", 0)
        score = (0.25 * det_score + 0.35 * sq_score + 0.15 * price_score
                 + 0.05 * trust_score + 0.10 * brand_boost + 0.10)
        score = max(0.0, min(1.0, score))

        classification = pre.get("classification_hint", "functional_equivalent")
        spec_match = "functional_equivalent"
        if sq_score >= 0.5:
            if sq_score >= 0.7:
                spec_match = "exact_spec_equivalent"
                classification = "direct_competitor"
            else:
                spec_match = "close_spec_equivalent"
        elif sq_score >= 0.3:
            spec_match = "close_spec_equivalent"

        scored.append({
            "candidate_index": i,
            "title": c.get("title"),
            "price": price,
            "currency": c.get("currency", "EUR"),
            "merchant": c.get("merchant"),
            "brand": c.get("brand") or _extract_brand_from_title(c.get("title")),
            "url": c.get("url", ""),
            "price_score": round(price_score, 3),
            "relevance_score": round(det_score, 3),
            "trust_score": round(trust_score, 3),
            "score": round(score, 3),
            "classification": classification,
            "spec_match": spec_match,
            "spec_quality": round(sq_score, 3),
            "is_vague": is_vague,
            "deterministic_score": det_score,
            "is_same_brand": pre.get("is_same_brand", False),
            "specs": c.get("specs", {}),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(scored):
        s["candidate_index"] = i
    return scored


# ───────────────────────────── Optional LLM Summarizer ─────────────────────────────

async def agent_summarizer(llm: Any, product: Any, scored: list[dict], run_id: str) -> dict:
    start = time.time()
    system = (
        "You are a competitive pricing analyst. Summarize the top equivalent products "
        "found for an electrical product. Keep it brief — 3-5 sentences. "
        "Mention the price range, key competitors, and any notable difference vs the target product. "
        "Output JSON with keys: summary, price_range_min, price_range_max, confidence. "
        "IMPORTANT: Each candidate below includes a 'brand' field. Do NOT infer a different brand from the title — "
        "trust the provided 'brand' field. If brand is None or missing, say 'Unknown'."
    )
    top5 = scored[:5]
    user = (
        f"Target: {product.name} | Brand: {product.brand or 'Unknown'}\n"
        f"Specs: {getattr(product, 'current_a', None)}A {getattr(product, 'poles', None)}P "
        f"curve {getattr(product, 'curve', None)} {getattr(product, 'breaking_capacity_ka', None)}kA\n\n"
        f"Top equivalents:\n"
        + "\n".join(
            f"  Brand: {str(s.get('brand', 'Unknown')):12s} | {str(s.get('title', ''))[:40]:40s} | {str(s.get('price', '?')):>8s} EUR | "
            f"{str(s.get('merchant', '')):20s} | score={s.get('score', 0.0)}"
            for s in top5
        )
        + "\n\nSummarize the competitive landscape based on the BRAND field — not the title."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    try:
        parsed = json.loads(result["content"])
    except json.JSONDecodeError:
        parsed = {"summary": result.get("content", "")[:200], "confidence": 0.5}
    return {"analysis": parsed, "latency_ms": latency}


# ───────────────────────────── Main Orchestration Task (Celery) ─────────────────────────────

@celery_app.task(bind=True, name="analyze_product", max_retries=3)
def analyze_product(self, product_id: str, run_id: str):
    import asyncio
    asyncio.run(_run_analysis(product_id, run_id))


@celery_app.task(bind=True, name="track_all_prices", max_retries=3)
def track_all_prices(self):
    import asyncio
    asyncio.run(_track_all_prices())


@celery_app.task(bind=True, name="check_price_alerts", max_retries=3)
def check_price_alerts(self):
    import asyncio
    asyncio.run(_check_alerts())


async def _run_analysis(product_id: str, run_id: str):
    start_total = time.time()

    async with async_session_factory() as db:
        await _update_analysis(db, run_id, status=AnalysisStatus.running, started_at=datetime.utcnow())
        product = await db.get(Product, product_id)
        if not product:
            await _update_analysis(db, run_id, status=AnalysisStatus.failed, error_message="Product not found")
            return

    queries = [product.description[:200]]
    raw = await agent_serpapi_search(queries, run_id, iteration=1)
    normalized = normalize_candidates(raw)
    scored = score_candidates(product, normalized)

    from worker.scoring import split_reliable_vs_weak
    reliable, weak = split_reliable_vs_weak(scored, product, target_currency=product.currency or "EUR")

    total_latency = (time.time() - start_total) * 1000
    best = reliable[0] if reliable else (scored[0] if scored else None)

    async with async_session_factory() as db:
        await _update_analysis(db, run_id,
            status=AnalysisStatus.completed,
            completed_at=datetime.utcnow(),
            total_latency_ms=total_latency,
            candidate_count=len(raw),
            valid_match_count=len(reliable),
            best_match_price=best["price"] if best else None,
            best_match_score=best["score"] if best else None,
            run_metadata={"total_scored": len(scored), "weak": len(weak)},
        )
        for s in scored:
            offer = Offer(
                product_id=product_id,
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
                product_id=product_id,
                price=best["price"],
                currency=best.get("currency", "USD"),
            )
            db.add(snap)
        await db.commit()

    logger.info(f"Analysis complete: product={product_id} run={run_id} latency={total_latency:.0f}ms scored={len(scored)} reliable={len(reliable)}")


async def _track_all_prices():
    logger.info("Running scheduled price tracking for all tracked products")
    async with async_session_factory() as db:
        result = await db.execute(
            select(Product).where(Product.is_tracked == True).limit(50)
        )
        products = result.scalars().all()

    for product in products:
        try:
            run_id = str(uuid.uuid4())
            async with async_session_factory() as db:
                run = AnalysisRun(product_id=str(product.id), status=AnalysisStatus.pending,
                                  run_metadata={"trigger": "scheduler"})
                db.add(run)
                await db.commit()
                await db.refresh(run)
            await _run_analysis(str(product.id), str(run.id))
        except Exception as e:
            logger.error(f"Price tracking failed for {product.id}: {e}")


async def _check_alerts():
    logger.info("Checking price alerts")
    async with async_session_factory() as db:
        result = await db.execute(
            select(
                Product.id, Product.name, Product.target_price,
                PriceSnapshot.price.label("current_price"),
                PriceSnapshot.snapshot_date
            )
            .join(PriceSnapshot, PriceSnapshot.product_id == Product.id)
            .distinct(Product.id)
            .order_by(Product.id, PriceSnapshot.snapshot_date.desc())
            .limit(20)
        )
        rows = result.all()
        for row in rows:
            if row.target_price and row.current_price:
                drop_pct = (row.target_price - row.current_price) / row.target_price * 100
                if drop_pct > 15:
                    logger.info(f"ALERT: {row.name} dropped {drop_pct:.0f}% (target: {row.target_price}, now: {row.current_price})")

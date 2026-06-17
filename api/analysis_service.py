"""Reusable equivalent-analysis service.

Shared by:
  - POST /api/v1/products/analyze-equivalents  (structured REST endpoint)
  - ChatOrchestrator (free-text chat messages)

IMPORTANT: run_equivalent_analysis_from_text() returns the *full* result dict.
           Never send this dict verbatim to the LLM.
           Always call compact_analysis_context() first to build a token-safe version.
"""
import logging
import time
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AnalysisStatus
from worker.tasks import agent_serpapi_search, normalize_candidates, score_candidates, agent_summarizer
from worker.scoring import _infer_product_attributes, split_by_quality, diversify_by_brand

logger = logging.getLogger("api.analysis_service")

_COMPACT_CANDIDATE_KEYS = frozenset({
    "title", "price", "currency", "merchant", "brand", "url",
    "score", "classification", "spec_match", "specs", "is_vague", "spec_quality",
})

_CRITICAL_SPECS = ("current_a", "poles", "curve", "breaking_capacity_ka", "voltage_v")

# Thresholds (mirrored from chat_service to avoid circular import)
_EXACT_MATCH_SCORE = 0.88
_LOW_CONFIDENCE_SCORE = 0.70


def _compact_candidate(c: dict) -> dict:
    """Strip a candidate dict to only the keys the LLM needs."""
    out = {k: v for k, v in c.items() if k in _COMPACT_CANDIDATE_KEYS}
    if "title" in out:
        out["title"] = out["title"][:80]
    if "url" in out and out["url"]:
        out["url"] = out["url"][:120]
    if not out.get("specs"):
        out.pop("specs", None)
    return {k: v for k, v in out.items() if v is not None and v != "" and v != {}}


def _tech_key(c: dict, ref_specs: dict) -> tuple:
    """Return a sortable tuple — higher values mean better technical match.

    Priority: current_a > curve > poles > breaking_capacity_ka > brand present
              > spec_quality > not vague.
    All comparisons are fuzzy-tolerant (float ±0.01, string case-insensitive).
    """
    specs = c.get("specs") or {}

    def _num_eq(key: str) -> int:
        ref = ref_specs.get(key)
        cand = specs.get(key)
        if ref is None or cand is None:
            return 0
        try:
            return 1 if abs(float(ref) - float(cand)) < 0.01 else 0
        except (TypeError, ValueError):
            return 0

    def _str_eq(key: str) -> int:
        ref = ref_specs.get(key)
        cand = specs.get(key)
        if ref is None or cand is None:
            return 0
        return 1 if str(ref).upper().strip() == str(cand).upper().strip() else 0

    return (
        _num_eq("current_a"),
        _str_eq("curve"),
        _num_eq("poles"),
        _num_eq("breaking_capacity_ka"),
        1 if (c.get("brand") or "").strip() else 0,     # brand present
        float(c.get("spec_quality") or 0),
        0 if c.get("is_vague") else 1,
    )


_DIFFERENTIAL_KEYWORDS = frozenset({
    "différentiel", "differentiel", "rccb", "rcbo", "vigi", " id ",
})

_MCB_CONTEXT_KEYWORDS = (
    "disjoncteur", "circuit breaker", "mcb", "ic60", "dx3", "mcn", "s201", "5sl",
)


def compute_candidate_recommendations(full_result: dict) -> dict:
    """Deterministic candidate selection before every LLM call.

    Returns cheapest_candidate, best_score_candidate, best_technical_candidate,
    best_market_analyst_choice, and properly labelled bucket counts.

    Rules:
    - confirmed_equivalents = cross_brand + same_brand (reliable bucket)
    - valid_match_count == 0  →  has_confirmed_equivalents = False
    - best_technical ranks by spec match (current_a > curve > poles > breaking_capacity_ka)
    - Multi-factor analyst ranking: bucket > score > spec_quality > price
    - Differential-breaker penalty when product context is a plain MCB
    - Never include vague candidates in best picks
    """
    valid_match_count = full_result.get("valid_match_count", 0)
    cross_brand = list(full_result.get("cross_brand_equivalents") or [])
    same_brand = list(full_result.get("same_brand_listings") or [])
    partial = list(full_result.get("partial_spec_equivalents") or [])
    weak = list(full_result.get("weak_candidates") or [])

    confirmed = cross_brand + same_brand

    inferred = full_result.get("inferred_product") or {}
    product_name_lower = (
        inferred.get("name") or full_result.get("product_name") or ""
    ).lower()
    is_mcb_context = any(k in product_name_lower for k in _MCB_CONTEXT_KEYWORDS)
    ref_specs = inferred.get("specs") or {}

    def _is_differential(c: dict) -> bool:
        title = (c.get("title") or "").lower()
        return any(k in title for k in _DIFFERENTIAL_KEYWORDS)

    def _pool(rank: int, candidates: list) -> list:
        return [{**c, "_br": rank} for c in candidates]

    all_pool = _pool(0, confirmed) + _pool(1, partial) + _pool(2, weak)
    non_vague = [c for c in all_pool if not c.get("is_vague")]
    priced = [c for c in all_pool if c.get("price") is not None and not c.get("is_vague")]

    cheapest = min(priced, key=lambda c: float(c.get("price") or 1e9), default=None)
    best_score = max(priced, key=lambda c: float(c.get("score") or 0), default=None)

    # Technical best: spec match priority — price deliberately excluded
    best_technical = (
        max(non_vague, key=lambda c: _tech_key(c, ref_specs), default=None)
        if non_vague else None
    )

    def _analyst_key(c: dict) -> tuple:
        br = c.get("_br", 2)
        vague_p = 10 if c.get("is_vague") else 0
        diff_p = 3 if (is_mcb_context and _is_differential(c)) else 0
        return (
            br + vague_p + diff_p,
            -float(c.get("score") or 0),
            -float(c.get("spec_quality") or 0),
            float(c.get("price") or 1e9),
        )

    eligible = [c for c in all_pool if c.get("price") is not None and not c.get("is_vague")]
    best_choice = min(eligible, key=_analyst_key, default=None) if eligible else None

    def _clean(c):
        return {k: v for k, v in c.items() if k != "_br"} if c else None

    has_confirmed = valid_match_count > 0 or len(confirmed) > 0

    # Missing critical specs — specs the inferred product is lacking
    missing_critical_specs = [s for s in _CRITICAL_SPECS if not ref_specs.get(s)]

    # Derived confidence level
    best_score_val = full_result.get("best_match_score")
    if valid_match_count == 0:
        confidence_level = "low"
    elif best_score_val is None or best_score_val < _LOW_CONFIDENCE_SCORE:
        confidence_level = "low"
    elif best_score_val < _EXACT_MATCH_SCORE:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    return {
        "confirmed_equivalents": [_clean(c) for c in confirmed[:10]],
        "confirmed_equivalents_count": valid_match_count,
        "partial_candidates": [_clean(c) for c in partial[:10]],
        "weak_candidates_sample": [_clean(c) for c in weak[:5]],
        "cheapest_candidate": _clean(cheapest),
        "best_score_candidate": _clean(best_score),
        "best_technical_candidate": _clean(best_technical),
        "best_market_analyst_choice": _clean(best_choice),
        "has_confirmed_equivalents": has_confirmed,
        "result_label": "Équivalents confirmés" if has_confirmed else "Candidats à vérifier",
        "no_match_warning": (
            "Aucun équivalent confirmé n'a été trouvé. "
            "Les résultats ci-dessous sont des candidats à vérifier."
        ) if not has_confirmed else None,
        "missing_critical_specs": missing_critical_specs or None,
        "confidence_level": confidence_level,
    }


def compact_analysis_context(
    full_result: dict,
    user_message: str = "",
    max_cross_brand: int = 10,
    max_same_brand: int = 10,
    max_partial: int = 10,
    max_weak: int = 10,
) -> dict:
    """Build a compact, token-safe analysis context dict for an LLM call.

    Never send full_result directly to the LLM — always call this function first.
    The returned dict is question-aware: only fields relevant to answering
    user_message are included, and candidate lists are capped per bucket.
    Includes compute_candidate_recommendations() so the LLM never needs to do
    arithmetic on raw lists.
    """
    out: dict = {
        "product_id": full_result.get("product_id"),
        "product_name": full_result.get("product_name"),
        "run_id": full_result.get("run_id"),
        "inferred_product": full_result.get("inferred_product"),
        "candidate_count": full_result.get("candidate_count", 0),
        "valid_match_count": full_result.get("valid_match_count", 0),
        "cross_brand_count": full_result.get("cross_brand_count", 0),
        "same_brand_count": full_result.get("same_brand_count", 0),
        "partial_spec_count": full_result.get("partial_spec_count", 0),
        "weak_candidate_count": full_result.get("weak_candidate_count", 0),
        "best_match_price": full_result.get("best_match_price"),
        "best_match_score": full_result.get("best_match_score"),
        "price_confidence": full_result.get("price_confidence"),
        "recommendation": full_result.get("recommendation"),
        "brand_diversity_warning": full_result.get("brand_diversity_warning"),
        "cross_brand_equivalents": [
            _compact_candidate(c)
            for c in (full_result.get("cross_brand_equivalents") or [])[:max_cross_brand]
        ],
        "same_brand_listings": [
            _compact_candidate(c)
            for c in (full_result.get("same_brand_listings") or [])[:max_same_brand]
        ],
        "partial_spec_equivalents": [
            _compact_candidate(c)
            for c in (full_result.get("partial_spec_equivalents") or [])[:max_partial]
        ],
        "weak_candidates": [
            _compact_candidate(c)
            for c in (full_result.get("weak_candidates") or [])[:max_weak]
        ],
    }
    # Include pre-computed recommendations so LLM never needs to do arithmetic
    recs = compute_candidate_recommendations(full_result)
    out["candidate_recommendations"] = {k: v for k, v in recs.items() if v is not None}
    return {k: v for k, v in out.items() if v is not None}


async def run_equivalent_analysis_from_text(
    query: str,
    db: AsyncSession,
    persist: bool = True,
    llm=None,
    *,
    name: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    sku: str | None = None,
    voltage_v: float | None = None,
    current_a: float | None = None,
    poles: int | None = None,
    curve: str | None = None,
    breaking_capacity_ka: float | None = None,
    target_price: float | None = None,
    currency: str = "EUR",
    max_candidates: int = 60,
) -> dict:
    """Run the full equivalent analysis pipeline from a free-text query.

    Returns the complete result dict for storage and traceability.
    When persist=True, writes Product, AnalysisRun, Offer, and PriceSnapshot rows.

    IMPORTANT: Never send the returned dict verbatim to the LLM.
    Call compact_analysis_context() on it first.
    """
    from api.llm_client import get_llm_client

    start_total = time.time()
    owned_llm = llm is None
    if owned_llm:
        llm = get_llm_client()

    inferred = _infer_product_attributes(
        description=query,
        brand=brand,
        category=category,
        name=name,
    )
    p_name = name or inferred["name"]
    p_category = category or inferred["category"]
    p_brand = brand or inferred.get("brand") or None
    inferred_specs = inferred.get("specs", {})

    product = Product(
        name=p_name,
        description=query[:500],
        category=p_category,
        brand=p_brand,
        sku=sku,
        target_price=target_price,
        currency=currency,
        voltage_v=voltage_v or inferred_specs.get("voltage_v"),
        current_a=current_a or inferred_specs.get("current_a"),
        poles=poles or inferred_specs.get("poles"),
        curve=curve or inferred_specs.get("curve"),
        breaking_capacity_ka=breaking_capacity_ka or inferred_specs.get("breaking_capacity_ka"),
        mounting=inferred_specs.get("mounting"),
    )

    if persist:
        db.add(product)
        await db.commit()
        await db.refresh(product)

    run = AnalysisRun(
        product_id=product.id,
        status=AnalysisStatus.running,
        started_at=datetime.utcnow(),
        run_metadata={"trigger": "service"},
    )
    if persist:
        db.add(run)
        await db.commit()
        await db.refresh(run)

    run_id = str(run.id) if persist else "in_memory"
    product_id = str(product.id) if persist else "in_memory"

    scored: list[dict] = []
    raw_candidates: list[dict] = []
    reliable_scored: list[dict] = []
    partial_scored: list[dict] = []
    weak_scored: list[dict] = []
    analysis: dict = {}
    best: dict | None = None

    try:
        try:
            serpapi_results = await agent_serpapi_search([query[:200]], run_id, iteration=1)
            raw_candidates = serpapi_results[:max_candidates]
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")

        if raw_candidates:
            normalized = normalize_candidates(raw_candidates)
            scored = score_candidates(product, normalized)

        reliable_scored, partial_scored, weak_scored = split_by_quality(
            scored, product, target_currency=currency,
        )

        reliable_ids = {id(s) for s in reliable_scored}
        partial_ids = {id(s) for s in partial_scored}
        for s in scored:
            if id(s) in reliable_ids:
                s["quality_bucket"] = "reliable"
            elif id(s) in partial_ids:
                s["quality_bucket"] = "partial"
            else:
                s["quality_bucket"] = "weak"

        if len(scored) >= 3:
            summary = await agent_summarizer(llm, product, scored, run_id)
            analysis = summary.get("analysis", {})
        else:
            analysis = {
                "market_overview": f"Only {len(scored)} equivalent(s) found.",
                "recommendation": "Insufficient data. Need at least 3 equivalents.",
                "confidence": 0.2,
                "below_threshold": True,
            }

        best_source = reliable_scored if reliable_scored else scored
        reliable_cross_brand = [s for s in best_source if not s.get("is_same_brand", False)]
        best = reliable_cross_brand[0] if reliable_cross_brand else (best_source[0] if best_source else None)

        if persist:
            run.status = AnalysisStatus.completed
            run.completed_at = datetime.utcnow()
            run.total_latency_ms = (time.time() - start_total) * 1000
            run.candidate_count = len(raw_candidates)
            run.valid_match_count = len(reliable_scored)
            run.best_match_price = best["price"] if best else None
            run.best_match_score = best["score"] if best else None
            run.price_confidence = analysis.get("confidence")
            run.final_decision = analysis
            run.run_metadata = {
                "total_scored": len(scored),
                "weak": len(weak_scored),
                "trigger": "service",
            }
            await db.commit()

            for s in scored:
                db.add(Offer(
                    product_id=product.id,
                    source="analysis",
                    competitor_name=s.get("title", "")[:255],
                    title=s.get("title", ""),
                    price=s.get("price", 0),
                    currency=s.get("currency", "USD"),
                    url=s.get("url", ""),
                    merchant=s.get("merchant"),
                    raw_data={
                        "score": s.get("score", 0),
                        "spec_quality": s.get("spec_quality", 0),
                        "classification": s.get("classification", ""),
                        "spec_match": s.get("spec_match", ""),
                        "is_same_brand": s.get("is_same_brand", False),
                        "is_vague": s.get("is_vague", False),
                        "brand": s.get("brand"),
                        "relevance_score": s.get("relevance_score", 0),
                        "quality_bucket": s.get("quality_bucket", "weak"),
                    },
                ))
            if best:
                db.add(PriceSnapshot(
                    product_id=product.id,
                    price=best["price"],
                    currency=best.get("currency", "USD"),
                ))
            await db.commit()

    except Exception as e:
        if persist:
            run.status = AnalysisStatus.failed
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            await db.commit()
        raise
    finally:
        if owned_llm:
            await llm.close()

    total_latency = (time.time() - start_total) * 1000

    cross_brand_reliable = [s for s in reliable_scored if not s.get("is_same_brand", False)]
    cross_brand_list, _, diversity_stats = diversify_by_brand(
        cross_brand_reliable, max_per_brand=2, max_total=5, min_brand_count=3,
    )
    same_brand_list = [s for s in reliable_scored if s.get("is_same_brand", False)][:4]

    brand_diversity_warning = None
    if diversity_stats.get("needs_supplemental_search"):
        brand_diversity_warning = (
            f"Only {diversity_stats['selected_brand_count']} distinct brand(s) found "
            f"(target: >= {diversity_stats.get('min_brand_count', 3)})."
        )

    final_specs = {k: v for k, v in {
        "voltage_v": product.voltage_v,
        "current_a": product.current_a,
        "poles": product.poles,
        "curve": product.curve,
        "breaking_capacity_ka": product.breaking_capacity_ka,
        "mounting": product.mounting,
    }.items() if v is not None}

    return {
        "product_id": product_id,
        "product_name": product.name,
        "run_id": run_id,
        "total_latency_ms": total_latency,
        "candidate_count": len(raw_candidates),
        "valid_match_count": len(reliable_scored),
        "cross_brand_count": len(cross_brand_list),
        "same_brand_count": len(same_brand_list),
        "partial_spec_count": len(partial_scored),
        "weak_candidate_count": len(weak_scored),
        "best_match_price": best["price"] if best else None,
        "best_match_score": best["score"] if best else None,
        "price_confidence": analysis.get("confidence"),
        "recommendation": analysis.get("summary") or analysis.get("recommendation"),
        "brand_diversity_warning": brand_diversity_warning,
        "brand_diversity_stats": diversity_stats,
        "inferred_product": {
            "name": product.name,
            "category": product.category,
            "brand": product.brand,
            "specs": final_specs,
        },
        "cross_brand_equivalents": cross_brand_list,
        "same_brand_listings": same_brand_list,
        "partial_spec_equivalents": partial_scored[:20],
        "weak_candidates": weak_scored[:20],
    }

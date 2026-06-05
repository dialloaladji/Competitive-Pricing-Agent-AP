import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from langfuse import Langfuse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import async_session_factory
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AnalysisStatus, AgentLog
from api.llm_client import get_llm_client, search_tavily, search_serpapi, estimate_cost
from worker.celery_app import celery_app

logger = logging.getLogger("worker")

langfuse = None
if settings.langfuse_public_key and settings.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url or "https://cloud.langfuse.com",
    )


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


# ───────────────────────────── Agent 1: product_understanding_agent ─────────────────────────────

async def agent_product_understanding(llm: Any, product: Product, run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are an electrical product intelligence specialist. The API is restricted to electrical products only. "
        "Extract structured attributes from a product listing. You must ONLY extract information explicitly present "
        "in the product description. Do not infer, assume, or fabricate attributes. "
        "If a field is not present in the description, set it to null. "
        "Pay particular attention to electrical specs: voltage (V), current (A), poles (1P/2P/3P/4P), "
        "curve (B/C/D/K), breaking capacity (kA), phase (single/three), power (W/kW), mounting (DIN rail/panel), "
        "standard (IEC 60898, NF C 15-100, UL 489), and usage (residential/commercial/industrial). "
        "Output must be valid JSON with keys: name, category, brand, sku, product_type, attributes, "
        "target_audience, price_indicators, specs."
    )
    user = (
        f"Product Name: {product.name}\n"
        f"Description: {product.description}\n"
        f"Category: {product.category or 'Not specified'}\n"
        f"Brand: {product.brand or 'Not specified'}\n"
        f"SKU: {product.sku or 'Not specified'}\n"
        f"Product Type: {getattr(product, 'product_type', None) or 'Not specified'}\n"
        f"Voltage: {getattr(product, 'voltage_v', None) or 'Not specified'} V\n"
        f"Current: {getattr(product, 'current_a', None) or 'Not specified'} A\n"
        f"Poles: {getattr(product, 'poles', None) or 'Not specified'}\n"
        f"Curve: {getattr(product, 'curve', None) or 'Not specified'}\n"
        f"Breaking Capacity: {getattr(product, 'breaking_capacity_ka', None) or 'Not specified'} kA\n"
        f"Phase: {getattr(product, 'phase', None) or 'Not specified'}\n"
        f"Mounting: {getattr(product, 'mounting', None) or 'Not specified'}\n"
        f"Standard: {getattr(product, 'standard', None) or 'Not specified'}\n"
        f"Usage: {getattr(product, 'usage', None) or 'Not specified'}\n"
        f"Target Price: {product.target_price} {product.currency}\n\n"
        "Extract: name, category, brand, sku, product_type, key electrical attributes, "
        "target audience (e.g. residential installers, panel builders, industrial B2B), "
        "price_indicators, and a specs object with the electrical fields where present."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    try:
        parsed = json.loads(result["content"])
        parse_ok = True
    except json.JSONDecodeError:
        parsed = {"error": "failed to parse", "raw": result["content"]}

    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "1.0", "json_parse_success": parse_ok,
        "log_metadata": {"parsed_output": parsed} if parse_ok else {"raw": result["content"]},
    }

    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="product_understanding_agent", metadata={"run_id": run_id, "product_id": str(product.id)})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "product_understanding_agent", log_data)

    return {"parsed": parsed, "latency_ms": latency, "parse_ok": parse_ok}


# ───────────────────────────── Agent 2: query_generator_agent ─────────────────────────────

async def agent_query_generator(llm: Any, product: Product, attributes: list, run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are a competitive research query strategist specialized in ELECTRICAL PRODUCTS. "
        "Generate search queries to find competing electrical products on Google Shopping, B2B "
        "distributor sites (Rexel, Sonepar, 123elec, Legallais), and the web.\n\n"
        "Generate TWO types of queries:\n"
        "1. GENERIC queries (without brand name) — to find competitors from other electrical "
        "manufacturers. Use the product type and electrical specs (voltage, current, poles, "
        "curve, kA). E.g. 'MCB 1P 16A curve C 6kA DIN rail' instead of 'ABB S201-C16'.\n"
        "2. BRAND-SPECIFIC queries (with brand name) — to find the exact product on different "
        "platforms. E.g. 'ABB S201-C16 prix'.\n\n"
        "Known electrical brands to consider: ABB, Schneider Electric, Legrand, Siemens, Eaton, "
        "Hager, Chint, Noark, Phoenix Contact, Wago, Finder, Lovato, Mitsubishi, Bticino, Gewiss. "
        "You may also discover other regional brands (e.g. Crouzet, Klockner Moeller, Telemecanique, "
        "Merlin Gerin, Square D, ABB Stotz).\n\n"
        "Each query must be a real, specific search string a human would type. "
        "Max 6 queries (3 generic, 3 brand-specific). "
        "Output JSON: { 'queries': ['...', '...'] }"
    )
    user = (
        f"Product: {product.name}\nBrand: {product.brand or 'N/A'}\n"
        f"Category: {product.category or 'N/A'}\n"
        f"SKU: {getattr(product, 'sku', None) or 'N/A'}\n"
        f"Key Electrical Attributes: {', '.join(attributes) if attributes else 'N/A'}\n"
        f"Target Price: {product.target_price} {product.currency}\n\n"
        "Generate up to 6 queries: 3 generic (no brand, focus on type and specs) and 3 brand-specific "
        "(find same product on different platforms or distributors)."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    queries = []
    try:
        parsed = json.loads(result["content"])
        queries = parsed.get("queries", [])
        parse_ok = True
    except json.JSONDecodeError:
        queries = [product.name]

    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "1.0", "json_parse_success": parse_ok,
        "log_metadata": {"queries": queries},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="query_generator_agent", metadata={"run_id": run_id, "product_id": str(product.id)})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "query_generator_agent", log_data)
    return {"queries": queries, "latency_ms": latency}


# ───────────────────────────── Agent 3: serpapi_search ─────────────────────────────

async def agent_serpapi_search(queries: list[str], run_id: str, iteration: int) -> list[dict]:
    start = time.time()
    all_results = []
    for q in queries:
        results = await search_serpapi(q, settings.web_search_max_results)
        all_results.extend(results)

    latency = (time.time() - start) * 1000
    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0,
        "model_name": "serpapi", "prompt_version": "1.0", "json_parse_success": True,
        "log_metadata": {"query_count": len(queries), "result_count": len(all_results)},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="serpapi_search", metadata={"run_id": run_id})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "serpapi_search", log_data)
    return all_results


# ───────────────────────────── Agent 4: tavily_search ─────────────────────────────

async def agent_tavily_search(queries: list[str], run_id: str, iteration: int) -> list[dict]:
    start = time.time()
    all_results = []
    for q in queries:
        results = await search_tavily(q, settings.web_search_max_results)
        all_results.extend(results)

    latency = (time.time() - start) * 1000
    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0,
        "model_name": "tavily", "prompt_version": "1.0", "json_parse_success": True,
        "log_metadata": {"query_count": len(queries), "result_count": len(all_results)},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="tavily_search", metadata={"run_id": run_id})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "tavily_search", log_data)
    return all_results


# ───────────────────────────── Agent 5: candidate_normalizer ─────────────────────────────

async def agent_candidate_normalizer(llm: Any, product: Product, raw_candidates: list[dict], run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are a data normalization expert for ELECTRICAL PRODUCT offers. Normalize competitor data "
        "into a unified schema. Rules: Only transform fields that exist in the input. Do not generate "
        "prices, URLs, or merchant names. Deduplicate by URL (same URL = same offer). Standardize currency "
        "to ISO 4217 (EUR for European B2B offers). Extract the BRAND from the title (Schneider, Legrand, "
        "ABB, Siemens, Hager, Eaton, Chint, Noark, Phoenix Contact, etc.). Reject candidates outside the "
        "electrical domain (consumer electronics, headphones, household appliances, food, clothing, etc.).\n\n"
        "For each candidate, also extract electrical specs if present in title/description: "
        "voltage_v, current_a, poles (1/2/3/4), curve (B/C/D/K), breaking_capacity_ka, phase, mounting. "
        "Include a 'specs' dict in each normalized candidate.\n\n"
        "Output JSON array of normalized_candidates."
    )
    candidates_for_llm = raw_candidates[:30]
    user = (
        f"Target Product: {product.name} ({product.brand or 'N/A'})\n"
        f"Product Type: {getattr(product, 'product_type', None) or 'N/A'}\n\n"
        f"Raw candidates ({len(raw_candidates)} total, showing first {len(candidates_for_llm)}):\n"
        + "\n".join(f"[{i+1}] Title: {c.get('title','')} | Price: {c.get('price','')} | "
                    f"URL: {c.get('url','')} | Merchant: {c.get('merchant','N/A')}"
                    for i, c in enumerate(candidates_for_llm))
        + "\n\nNormalize each candidate. Deduplicate by URL. Standardize currency. "
        "Reject non-electrical products. Extract brand and electrical specs. "
        "Output a JSON array of normalized_candidates."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    normalized = []
    try:
        parsed = json.loads(result["content"])
        normalized = parsed if isinstance(parsed, list) else parsed.get("normalized_candidates", [])
        parse_ok = True
    except json.JSONDecodeError:
        normalized = []

    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "1.0", "json_parse_success": parse_ok,
        "log_metadata": {"raw_count": len(raw_candidates), "normalized_count": len(normalized), "candidate_count": len(normalized)},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="candidate_normalizer", metadata={"run_id": run_id, "product_id": str(product.id)})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "candidate_normalizer", log_data)
    return {"normalized": normalized, "latency_ms": latency}


# ───────────────────────────── Agent 6: llm_judge ─────────────────────────────

async def agent_llm_judge(llm: Any, product: Product, normalized: list[dict], attributes: list, run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are a competitive market analyst specialized in ELECTRICAL PRODUCTS. The API is "
        "restricted to electrical products only (circuit breakers, contactors, switches, cables, "
        "electrical panels, EV chargers, etc.).\n\n"
        "Identify credible competitive alternatives to the target electrical product.\n\n"
        "DOMAIN RULE: If a candidate is NOT an electrical product (e.g. headphones, food, clothing, "
        "consumer goods, household non-electrical items), classify as 'irrelevant' and REJECT.\n\n"
        "IMPORTANT — Prioritize candidates from DIFFERENT BRANDS. True competitive benchmarking "
        "means finding comparable products from other electrical manufacturers (ABB vs Schneider, "
        "Legrand, Siemens, Eaton, Hager, Chint, Noark, Phoenix Contact, Wago, Finder, Lovato, "
        "Mitsubishi, Bticino, Gewiss, Crouzet, Telemecanique, Merlin Gerin, Square D, etc.).\n"
        "Same-brand candidates are only relevant if they represent a different price point "
        "(cheaper_alternative, premium_alternative) or a different generation.\n\n"
        "Evaluate each candidate by asking:\n"
        "- Is it an electrical product?\n"
        "- Does it belong to the same functional category (protection, contactor, cable, etc.)?\n"
        "- Are the main electrical specs comparable (voltage, current, poles, breaking capacity)?\n"
        "- Is it suitable for the same usage (residential, commercial, industrial)?\n"
        "- Is its price in a coherent range?\n\n"
        "Examples (electrical products):\n"
        '- Target="ABB S201-C16 MCB 1P 16A curve C 6kA" → '
        'Competitor="Schneider Easy9 1P 16A C 6kA" → direct_competitor\n'
        '- Target="ABB S201-C16 MCB 1P 16A C 6kA" → '
        'Competitor="Legrand RX3 1P 16A C 6kA" → direct_competitor\n'
        '- Target="ABB S201-C16 MCB" → '
        'Competitor="Hager MCN116 1P 16A C 6kA" → direct_competitor\n'
        '- Target="ABB S201-C16 MCB" → '
        'Competitor="Siemens 5SL6106 1P 16A C 6kA" → premium_alternative (higher price)\n'
        '- Target="ABB S201-C16 MCB" → '
        'Competitor="Chint NXB-63 1P 16A C 6kA" → cheaper_alternative\n'
        '- Target="Schneider LC1D25 contactor 3P 25A" → '
        'Competitor="ABB AF16 3P 25A contactor" → direct_competitor\n'
        '- Target="ABB S201-C16 MCB" → '
        'Competitor="Schneider iC60N 1P 16A C 10kA" → premium_alternative (higher kA)\n'
        '- Target="ABB S201-C16 MCB" → '
        'Competitor="Bobine MX 12V pour ABB S200" → accessory_or_part (REJECT)\n'
        '- Target="ABB S201-C16 MCB" → '
        'Competitor="Sony WH-1000XM5 headphones" → irrelevant (REJECT, not electrical)\n'
        '- Target="Legrand 07701 1P 16A" → '
        'Competitor="Legrand 07700 1P 10A" → functional_equivalent\n\n'
        "CLASSIFY each candidate into one of:\n"
        "- same_product: exact or near-exact match (same brand, same model, possibly different seller)\n"
        "- direct_competitor: very comparable product from a DIFFERENT brand\n"
        "- functional_equivalent: same function, different brand or features\n"
        "- cheaper_alternative: relevant but cheaper (from any brand)\n"
        "- premium_alternative: relevant but more premium (from any brand)\n"
        "- previous_generation: older version of same product line\n"
        "- newer_generation: newer version of same product line\n"
        "- accessory_or_part: accessory, spare part, consumable, complementary (REJECT)\n"
        "- irrelevant: not related OR not an electrical product (REJECT)\n\n"
        "Return strictly valid JSON array:\n"
        '[{"candidate_index": 0, "classification": "direct_competitor", '
        '"confidence": 0.78, "reason": "..."}]'
    )
    user = (
        f"TARGET PRODUCT: {product.name} | Brand: {product.brand} | "
        f"Category: {product.category} | "
        f"Specs: voltage={getattr(product, 'voltage_v', None)}V, "
        f"current={getattr(product, 'current_a', None)}A, "
        f"poles={getattr(product, 'poles', None)}, "
        f"curve={getattr(product, 'curve', None)}, "
        f"kA={getattr(product, 'breaking_capacity_ka', None)} | "
        f"Attributes: {', '.join(attributes)} | "
        f"Target Price: {product.target_price} {product.currency}\n\nCANDIDATES:\n"
        + "\n".join(f"[{i}] Title: '{c.get('title','')}' | Brand: {c.get('brand','N/A')} | "
                    f"Price: {c.get('price','')} | Merchant: {c.get('merchant','')} | "
                    f"URL: {c.get('url','')}"
                    for i, c in enumerate(normalized))
        + "\n\nFor each candidate, classify as competitive equivalent, functional equivalent, or reject. "
        "Reject any non-electrical product."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    judgments = []
    try:
        parsed = json.loads(result["content"])
        raw = parsed if isinstance(parsed, list) else parsed.get("judgments", [])
        judgments = [j for j in raw if isinstance(j, dict)]
        parse_ok = True
    except json.JSONDecodeError:
        judgments = []

    VALID = {"same_product", "direct_competitor", "functional_equivalent",
             "cheaper_alternative", "premium_alternative",
             "previous_generation", "newer_generation"}
    valid_count = sum(1 for j in judgments if j.get("classification") in VALID)

    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "2.0", "json_parse_success": parse_ok,
        "log_metadata": {"judgments": judgments, "valid_match_count": valid_count},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="llm_judge", metadata={"run_id": run_id, "product_id": str(product.id)})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "llm_judge", log_data)
    return {"judgments": judgments, "latency_ms": latency}


# ───────────────────────────── Agent 7: scoring_engine (deterministic) ─────────────────────────────

VALID_CLASSIFICATIONS = frozenset({
    "same_product", "direct_competitor", "functional_equivalent",
    "cheaper_alternative", "premium_alternative",
    "previous_generation", "newer_generation",
})

CLASSIFICATION_CONFIDENCE_THRESHOLDS = {
    "same_product": 0.45,
    "direct_competitor": 0.40,
    "functional_equivalent": 0.35,
    "cheaper_alternative": 0.35,
    "premium_alternative": 0.35,
    "previous_generation": 0.35,
    "newer_generation": 0.35,
}


def agent_scoring_engine(product: Product, normalized: list[dict], judgments: list[dict],
                         deterministic_scores: list[dict] | None = None) -> list[dict]:
    scored = []
    for i, candidate in enumerate(normalized):
        judgment = next((j for j in judgments if j.get("candidate_index") == i), None)
        pre = deterministic_scores[i] if deterministic_scores and i < len(deterministic_scores) else {}

        classification = "irrelevant"
        confidence = 0.0
        reason = ""

        if judgment:
            raw_classification = judgment.get("classification", "irrelevant")
            confidence = float(judgment.get("confidence", 0))
            reason = str(judgment.get("reason", ""))

            if raw_classification in ("accessory_or_part", "irrelevant"):
                det_score = pre.get("deterministic_score", 0)
                if det_score >= 0.30 and not pre.get("is_accessory", False):
                    classification = pre.get("classification_hint", "functional_equivalent")
                    confidence = det_score
                    reason = "LLM rejected but deterministic score qualifies"
                else:
                    continue

            elif raw_classification in CLASSIFICATION_CONFIDENCE_THRESHOLDS:
                min_conf = CLASSIFICATION_CONFIDENCE_THRESHOLDS[raw_classification]
                if confidence >= min_conf:
                    classification = raw_classification
                else:
                    classification = "irrelevant"
            else:
                classification = "irrelevant"
        else:
            det_score = pre.get("deterministic_score", 0)
            if det_score >= 0.30 and not pre.get("is_accessory", False):
                classification = pre.get("classification_hint", "functional_equivalent")
                confidence = det_score
            else:
                continue

        raw_price = candidate.get("price", 0)
        try:
            price = float(raw_price) if raw_price else 0
        except (ValueError, TypeError):
            price = 0
        target = product.target_price or price

        price_ratio = price / target if target > 0 else 1.0
        price_score = max(0, 1.0 - abs(1.0 - price_ratio) * 0.5)

        trust_scores = {"amazon": 0.9, "walmart": 0.85, "bestbuy": 0.9,
                        "target": 0.85, "newegg": 0.8, "ebay": 0.6,
                        "rexel": 0.9, "sonepar": 0.9, "cdiscount": 0.8,
                        "manomano": 0.85, "leroy merlin": 0.85, "castorama": 0.85,
                        "planet-bricolage": 0.8, "bricomarche": 0.8,
                        "123elec": 0.85, "elec-shop": 0.8, "eibmarkt": 0.8,
                        "distributique": 0.8, "tubesca": 0.75, "domomat": 0.85,
                        "legallais": 0.9, "cbo": 0.85, "yed": 0.8, "fuseau": 0.8}
        trust_score = trust_scores.get(str(candidate.get("merchant", "")).lower(), 0.5)

        det_score = pre.get("deterministic_score", 0)
        score = 0.25 * det_score + 0.35 * confidence + 0.25 * price_score + 0.15 * trust_score

        is_same_brand = pre.get("is_same_brand", False)

        scored.append({
            "candidate_index": i,
            "title": candidate.get("title"),
            "price": price,
            "currency": candidate.get("currency", "EUR"),
            "merchant": candidate.get("merchant"),
            "url": candidate.get("url"),
            "price_score": round(price_score, 3),
            "relevance_score": round(confidence, 3),
            "trust_score": round(trust_score, 3),
            "score": round(score, 3),
            "classification": classification,
            "deterministic_score": det_score,
            "is_same_brand": is_same_brand,
            "specs": candidate.get("specs", {}),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(scored):
        s["candidate_index"] = i
    return scored


# ───────────────────────────── Agent 8: reflection_agent ─────────────────────────────

async def agent_reflection(llm: Any, product: Product, candidate_count: int, valid_count: int,
                           scored: list[dict], target_price: float | None, run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are a quality assurance analyst for competitive pricing data. Evaluate analysis results "
        "for quality and completeness. Check for: Price outliers (any price > 5x or < 0.2x target price "
        "→ flag), Semantic relevance (are matched products actually comparable?), Coverage (enough matches "
        "for market analysis?). Output JSON: { 'quality_score': 0.0-1.0, 'needs_reformulation': bool, "
        "'issues': [...], 'confidence': 0.0-1.0 }. Only set needs_reformulation=true if quality_score < 0.3 "
        "and zero valid matches."
    )
    user = (
        f"Product: {product.name} ({product.brand or 'N/A'})\n"
        f"Target Price: {target_price} {product.currency}\n"
        f"Candidates Found: {candidate_count}\nValid Matches: {valid_count}\n\n"
        f"Scored Competitors:\n"
        + "\n".join(f"- {s['title']} | {s['merchant']} | {s['price']} {s['currency']} | Score: {s['score']}"
                    for s in scored)
        + "\n\nEvaluate quality. Should queries be reformulated?"
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    try:
        parsed_raw = json.loads(result["content"])
        parsed = parsed_raw if isinstance(parsed_raw, dict) else {}
        parse_ok = isinstance(parsed_raw, dict)
    except json.JSONDecodeError:
        parsed = {"quality_score": 0.3, "needs_reformulation": True, "issues": ["parse error"], "confidence": 0.3}

    needs_reform = parsed.get("needs_reformulation", False)
    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "1.0", "json_parse_success": parse_ok,
        "log_metadata": {"quality_score": parsed.get("quality_score"), "needs_reformulation": needs_reform, "issues": parsed.get("issues")},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="reflection_agent", metadata={"run_id": run_id, "product_id": str(product.id)})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "reflection_agent", log_data)
    return {"quality_score": parsed.get("quality_score"), "needs_reformulation": needs_reform,
            "issues": parsed.get("issues", []), "latency_ms": latency}


# ───────────────────────────── Agent 9: query_reformulator ─────────────────────────────

async def agent_query_reformulator(llm: Any, product: Product, previous_queries: list[str],
                                   issues: list[str], attributes: list, run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are a search query optimization specialist. Given a product and the results from previous queries, "
        "generate improved queries to find better competitor data. Rules: Analyze why previous queries failed. "
        "Generate 3 new queries that address identified gaps. Vary phrasing: add synonyms, remove brand if too "
        "restrictive, add category terms. Output JSON: { 'previous_issues': [...], 'new_queries': ['...','...','...'], "
        "'strategy': '...' }"
    )
    user = (
        f"Product: {product.name}\nBrand: {product.brand or 'N/A'}\n"
        f"Category: {product.category or 'N/A'}\n"
        f"Attributes: {', '.join(attributes) if attributes else 'N/A'}\n\n"
        f"Previous Queries:\n" + "\n".join(f"- '{q}'" for q in previous_queries) + "\n"
        f"Issues: " + "; ".join(issues) + "\n\nGenerate 3 improved queries."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    new_queries = []
    try:
        parsed = json.loads(result["content"])
        new_queries = parsed.get("new_queries", [])
        parse_ok = True
    except json.JSONDecodeError:
        new_queries = [product.name + " price"]

    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "1.0", "json_parse_success": parse_ok,
        "log_metadata": {"new_queries": new_queries, "strategy": parsed.get("strategy") if parsed else ""},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="query_reformulator", metadata={"run_id": run_id, "product_id": str(product.id)})
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "query_reformulator", log_data)
    return {"queries": new_queries, "latency_ms": latency}


# ───────────────────────────── Agent 10: market_analyst_agent ─────────────────────────────

async def agent_market_analyst(llm: Any, product: Product, scored: list[dict], valid_count: int,
                               iteration_count: int, run_id: str, iteration: int) -> dict:
    start = time.time()
    system = (
        "You are a senior market intelligence analyst. Produce a competitive pricing analysis report. "
        "RULES: Every claim about a competitor's price MUST include the source URL. Do not speculate on "
        "pricing trends without data. If insufficient data (< 2 competitors), state "
        "'Insufficient data for market analysis'. Price recommendations must reference specific competitor prices. "
        "Output must be valid JSON with sections: market_overview, competitor_table, price_analysis, recommendation, confidence."
    )
    user = (
        f"Product: {product.name}\nBrand: {product.brand}\nCategory: {product.category}\n"
        f"Our Target Price: {product.target_price} {product.currency}\n"
        f"Valid Competitors Found: {valid_count}\nTotal Iterations: {iteration_count}\n\n"
        + (f"Competitor Data:\n" +
           "\n".join(f"| {s['title'][:30]} | {s['price']} | {s['merchant']} | {s['score']} | {s['url']} |"
                     for s in scored[:10])
           if scored else "No competitors found.")
        + "\n\nProduce market analysis with recommendations."
    )
    result = await llm.chat(system, user)
    latency = (time.time() - start) * 1000
    parsed = None
    parse_ok = False
    try:
        parsed = json.loads(result["content"])
        parse_ok = True
    except json.JSONDecodeError:
        parsed = {"market_overview": "Analysis failed", "competitor_table": [],
                   "price_analysis": "N/A", "recommendation": "N/A", "confidence": 0}

    log_data = {
        "iteration_number": iteration, "latency_ms": latency,
        "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
        "estimated_cost": estimate_cost(result.get("model", ""), result.get("input_tokens", 0), result.get("output_tokens", 0)),
        "model_name": result.get("model"), "prompt_version": "1.0", "json_parse_success": parse_ok,
        "log_metadata": {"analysis_summary": {k: v for k, v in parsed.items() if k != "competitor_table"}},
    }
    async with async_session_factory() as db:
        trace_id = str(uuid.uuid4())
        if langfuse:
            langfuse.trace(id=trace_id, name="market_analyst_agent", metadata={
                "run_id": run_id, "product_id": str(product.id), "final_decision": parsed
            })
        log_data["span_id"] = trace_id
        await _create_agent_log(db, run_id, "market_analyst_agent", log_data)
    return {"analysis": parsed, "latency_ms": latency}


# ───────────────────────────── Main Orchestration Task ─────────────────────────────

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
    llm = get_llm_client()

    async with async_session_factory() as db:
        await _update_analysis(db, run_id, status=AnalysisStatus.running, started_at=datetime.utcnow())
        product = await db.get(Product, product_id)
        if not product:
            await _update_analysis(db, run_id, status=AnalysisStatus.failed, error_message="Product not found")
            return

    # Agent 1: Product Understanding
    a1 = await agent_product_understanding(llm, product, run_id, iteration=1)
    attributes = a1.get("parsed", {}).get("attributes", [])

    # Agent 2: Query Generator
    a2 = await agent_query_generator(llm, product, attributes, run_id, iteration=1)
    queries = a2["queries"]

    all_raw_candidates = []
    all_normalized = []
    all_judgments = []
    scored = []
    iteration = 1
    max_iterations = 3

    while iteration <= max_iterations:
        # Agent 3 & 4: Search in parallel
        serpapi_results = await agent_serpapi_search(queries, run_id, iteration)
        tavily_results = await agent_tavily_search(queries, run_id, iteration)

        raw_candidates = serpapi_results + tavily_results
        all_raw_candidates.extend(raw_candidates)

        if not raw_candidates:
            iteration += 1
            if iteration > max_iterations:
                break
            a9 = await agent_query_reformulator(llm, product, queries, ["no results"], attributes, run_id, iteration)
            queries = a9["queries"]
            continue

        # Agent 5: Normalize
        a5 = await agent_candidate_normalizer(llm, product, raw_candidates, run_id, iteration)
        normalized = a5["normalized"]
        all_normalized.extend(normalized)

        if not normalized:
            iteration += 1
            if iteration > max_iterations:
                break
            a9 = await agent_query_reformulator(llm, product, queries, ["no valid candidates after normalization"], attributes, run_id, iteration)
            queries = a9["queries"]
            continue

        # Agent 6: Judge
        a6 = await agent_llm_judge(llm, product, normalized, attributes, run_id, iteration)
        judgments = a6["judgments"]
        all_judgments.extend(judgments)

        # Agent 7: Score (deterministic)
        scored = agent_scoring_engine(product, normalized, judgments)
        valid_count = len(scored)

        # Agent 8: Reflect
        a8 = await agent_reflection(llm, product, len(normalized), valid_count, scored,
                                     product.target_price, run_id, iteration)

        if not a8["needs_reformulation"] or iteration >= max_iterations:
            break

        # Agent 9: Reformulate
        a9 = await agent_query_reformulator(llm, product, queries, a8["issues"], attributes, run_id, iteration)
        queries = a9["queries"]
        iteration += 1

    # Agent 10: Market Analysis
    a10 = await agent_market_analyst(llm, product, scored, len(scored), iteration, run_id, iteration=iteration)

    total_latency = (time.time() - start_total) * 1000
    best = scored[0] if scored else None

    # Save results to DB
    async with async_session_factory() as db:
        await _update_analysis(db, run_id,
            status=AnalysisStatus.completed,
            completed_at=datetime.utcnow(),
            total_latency_ms=total_latency,
            candidate_count=len(all_raw_candidates) + len(all_normalized),
            valid_match_count=len(scored),
            best_match_price=best["price"] if best else None,
            best_match_score=best["score"] if best else None,
            price_confidence=a10.get("analysis", {}).get("confidence"),
            final_decision=a10.get("analysis"),
            run_metadata={"iterations": iteration, "total_queries": queries},
        )

        # Save valid offers
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

        # Save price snapshot
        if best:
            snap = PriceSnapshot(
                product_id=product_id,
                price=best["price"],
                currency=best.get("currency", "USD"),
            )
            db.add(snap)
        await db.commit()

    await llm.close()
    logger.info(f"Analysis complete: product={product_id} run={run_id} latency={total_latency:.0f}ms matches={len(scored)}")


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
        # Simple alert: find products with a price drop > 15%
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

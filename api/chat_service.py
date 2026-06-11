import json
import logging
import time
import traceback
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.chat_schemas import (
    ChatResponse, ProductBrief, PriceAnalysis, MarketAnalysis,
)
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AnalysisStatus
from api.llm_client import get_llm_client, search_serpapi
from worker.tasks import normalize_candidates, score_candidates
from worker.scoring import (
    _infer_product_attributes, _extract_brand_from_title,
    _is_electrical, ELECTRICAL_BRANDS, CATEGORY_DISPLAY_NAMES,
)

logger = logging.getLogger("api.chat")

SYSTEM_INTENT_PROMPT = """Classify the user's question about electrical products into exactly one intent.

Possible intents:
- product_lookup: user wants to find or identify a product
- product_comparison: user wants to compare multiple products
- equivalent_products_search: user wants equivalent/competitor products
- price_analysis: user wants current price analysis
- price_history_analysis: user wants price evolution over time
- stock_analysis: user wants stock/availability information
- market_analysis: user wants a full market summary
- general_question: general question not fitting above categories

Return JSON: {"intent": "product_lookup", "confidence": 0.95}
Use the most specific intent possible. When unsure, use general_question."""

SYSTEM_ANSWER_PROMPT = """You are an expert e-commerce and electrical market analyst.

Analyze the provided data about an electrical product.

ANTI-HALLUCINATION RULES (YOU MUST FOLLOW):
- Do NOT invent prices, stock status, references, EAN/GTIN, seller names, or technical specs.
- Do NOT invent historical trends or market events.
- Do NOT present hypotheses as facts.
- If information is missing, state it clearly.
- Always separate: observed facts, hypotheses, and recommendations.

Output JSON with these keys:
- answer: str — natural language expert answer (3-8 sentences)
- observed_facts: list[str]
- hypotheses: list[str]
- risks: list[str]
- recommendations: list[str]
- confidence: "high" | "medium" | "low"
- missing_information: list[str]
- sources_used: list[str]"""

SYSTEM_ANALYSIS_ANSWER_PROMPT = """You are an expert e-commerce and electrical market analyst.

The system just searched for equivalent products and found the following results.
Analyze what was found.

ANTI-HALLUCINATION RULES (YOU MUST FOLLOW):
- Do NOT invent any data not provided below.
- If no equivalents were found, say so clearly.
- If confidence is low, explain why.
- Suggest what additional information would improve the search.

Output JSON with these keys:
- answer: str — natural language analysis (3-8 sentences)
- observed_facts: list[str]
- hypotheses: list[str]
- risks: list[str]
- recommendations: list[str]
- confidence: "high" | "medium" | "low"
- missing_information: list[str]
- sources_used: list[str]"""


def _safe_analysis_answer(n: int, top: list) -> str:
    if top:
        prices = [s.get("price", 0) for s in top if isinstance(s.get("price"), (int, float))]
        if prices:
            min_p = min(prices)
            max_p = max(prices)
            return f"Found {n} product(s) online. Prices range from €{min_p:.2f} to €{max_p:.2f}. Verify compatibility before purchase."
    return f"Found {n} product(s) online. No scored results. Verify compatibility before purchase."


def _mock_intent(message: str) -> dict:
    message_lower = message.lower()
    if any(w in message_lower for w in ["historique", "price history", "price evolution", "evolution", "price trend", "trend"]):
        return {"intent": "price_history_analysis", "confidence": 0.7}
    if any(w in message_lower for w in ["compare", "comparer", "vs", "versus"]):
        return {"intent": "product_comparison", "confidence": 0.7}
    if any(w in message_lower for w in ["equivalent", "alternative", "remplacer", "similar"]):
        return {"intent": "equivalent_products_search", "confidence": 0.7}
    if any(w in message_lower for w in ["stock", "disponible", "available", "rupture"]):
        return {"intent": "stock_analysis", "confidence": 0.7}
    if any(w in message_lower for w in ["prix", "price", "cher", "coûte"]):
        return {"intent": "price_analysis", "confidence": 0.7}
    if any(w in message_lower for w in ["market", "marché", "summary", "analyse", "analyst"]):
        return {"intent": "market_analysis", "confidence": 0.7}
    if any(w in message_lower for w in ["disjoncteur", "interrupteur", "contacteur", "legrand",
                                          "schneider", "abb", "hager", "siemens", "eaton"]):
        return {"intent": "product_lookup", "confidence": 0.6}
    return {"intent": "general_question", "confidence": 0.5}


class ChatOrchestrator:

    def __init__(self, db: AsyncSession, llm: any):
        self.db = db
        self.llm = llm
        self.is_mock = getattr(llm, "model", "") == "mock"

    async def process(
        self,
        message: str,
        product_id: str | None = None,
        conversation_id: str | None = None,
    ) -> ChatResponse:
        start_time = time.time()
        actions: list[str] = []
        sources: list[str] = []

        intent, intent_conf = await self._classify_intent(message)
        sources.append("intent_classifier")
        actions.append(f"intent_classified_as_{intent}")

        product = None
        if product_id:
            product = await self._get_product_by_id(product_id)
            sources.append("database")
            actions.append("product_lookup_by_id")
            if not product:
                actions.append("product_id_not_found")

        if not product:
            candidates = await self._search_product_in_db(message)
            if candidates:
                product = candidates[0]
                sources.append("database")
                actions.append("product_found_by_search")
                if len(candidates) > 1:
                    actions.append("multiple_candidates_found")

        if product:
            resp = await self._handle_product_found(product, message, intent, sources, actions)
        else:
            resp = await self._handle_no_product(message, intent, sources, actions)

        if product:
            resp.product = ProductBrief(
                id=str(product.id),
                name=product.name,
                brand=product.brand,
                category=product.category,
                reference=product.sku,
            )

        resp.intent = intent
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Chat processed | intent={intent} | "
            f"product_found={'yes' if product else 'no'} | "
            f"latency_ms={latency_ms} | confidence={resp.confidence}"
        )
        return resp

    async def _classify_intent(self, message: str) -> tuple[str, float]:
        if self.is_mock:
            result = _mock_intent(message)
            return result["intent"], result["confidence"]
        try:
            resp = await self.llm.chat(SYSTEM_INTENT_PROMPT, f"Question: {message}")
            data = json.loads(resp["content"])
            return data.get("intent", "general_question"), data.get("confidence", 0.5)
        except Exception:
            return "general_question", 0.3

    async def _get_product_by_id(self, product_id: str) -> Product | None:
        try:
            return await self.db.get(Product, product_id)
        except Exception:
            return None

    async def _search_product_in_db(self, message: str) -> list[Product]:
        candidates: list[Product] = []
        seen_ids: set[str] = set()
        message_lower = message.lower()

        brand = _extract_brand_from_title(message)
        try:
            inferred = _infer_product_attributes(description=message)
        except Exception:
            inferred = {}

        try:
            if brand:
                stmt = select(Product).where(Product.brand.ilike(f"%{brand}%")).limit(5)
                result = await self.db.execute(stmt)
                for p in result.scalars():
                    if p.id not in seen_ids:
                        candidates.append(p)
                        seen_ids.add(p.id)

            for word in message_lower.split()[:5]:
                if len(word) >= 4:
                    stmt = select(Product).where(Product.name.ilike(f"%{word}%")).limit(3)
                    result = await self.db.execute(stmt)
                    for p in result.scalars():
                        if p.id not in seen_ids:
                            candidates.append(p)
                            seen_ids.add(p.id)
                            if len(candidates) >= 10:
                                break
                if len(candidates) >= 10:
                    break
        except Exception:
            logger.warning("DB search failed (likely loop mismatch)", exc_info=True)

        return candidates

    async def _get_offers(self, product_id: str) -> list[dict]:
        stmt = select(Offer).where(Offer.product_id == product_id).order_by(Offer.price)
        result = await self.db.execute(stmt)
        return [
            {
                "title": o.title,
                "price": o.price,
                "currency": o.currency,
                "merchant": o.merchant,
                "url": o.url,
                "in_stock": o.in_stock,
            }
            for o in result.scalars()
        ]

    async def _get_price_history(self, product_id: str) -> list[dict]:
        stmt = (
            select(PriceSnapshot)
            .where(PriceSnapshot.product_id == product_id)
            .order_by(PriceSnapshot.snapshot_date.desc())
            .limit(100)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "price": s.price,
                "currency": s.currency,
                "date": s.snapshot_date.isoformat() if s.snapshot_date else None,
            }
            for s in result.scalars()
        ]

    async def _get_equivalents(self, product_id: str) -> list[dict]:
        offers = await self._get_offers(product_id)
        return offers

    def _compute_price_analysis(self, product: Product, price_history: list[dict]) -> PriceAnalysis:
        if not price_history:
            return PriceAnalysis(has_history=False, trend="unknown")
        prices = [s["price"] for s in price_history if s["price"] is not None]
        if not prices:
            return PriceAnalysis(has_history=False, trend="unknown")
        sorted_prices = sorted(prices)
        n = len(sorted_prices)
        median = sorted_prices[n // 2] if n % 2 else (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2
        pa = PriceAnalysis(
            has_history=True,
            min_price=min(prices),
            max_price=max(prices),
            median_price=round(median, 2),
            currency=product.currency or "EUR",
        )
        if len(prices) >= 3:
            recent = prices[: min(5, len(prices))]
            oldest = prices[-min(5, len(prices)) :]
            avg_recent = sum(recent) / len(recent)
            avg_oldest = sum(oldest) / len(oldest)
            if avg_recent > avg_oldest * 1.03:
                pa.trend = "increasing"
            elif avg_recent < avg_oldest * 0.97:
                pa.trend = "decreasing"
            else:
                pa.trend = "stable"
        return pa

    async def _generate_answer(
        self,
        product: Product,
        offers: list[dict],
        price_history: list[dict],
        intent: str,
    ) -> dict:
        price_analysis = self._compute_price_analysis(product, price_history)
        if self.is_mock:
            return self._mock_answer(product, offers, price_analysis, intent)
        product_info = (
            f"Product: {product.name} | Brand: {product.brand or 'N/A'} | "
            f"Category: {product.category or 'N/A'} | SKU: {product.sku or 'N/A'}\n"
            f"Specs: current_a={product.current_a}, poles={product.poles}, "
            f"curve={product.curve}, kA={product.breaking_capacity_ka}\n"
        )
        offers_info = f"Offers ({len(offers)} found):\n"
        for o in offers[:10]:
            offers_info += f"  - €{o['price']} at {o.get('merchant', 'N/A')} (stock: {o.get('in_stock', 'N/A')})\n"
        history_info = f"Price history:\n  Has data: {price_analysis.has_history}\n"
        if price_analysis.has_history:
            history_info += (
                f"  Min: €{price_analysis.min_price}\n"
                f"  Max: €{price_analysis.max_price}\n"
                f"  Median: €{price_analysis.median_price}\n"
                f"  Trend: {price_analysis.trend}\n"
            )
        else:
            history_info += "  No historical price data available.\n"
        user = f"{product_info}\n{offers_info}\n{history_info}\nIntent: {intent}"
        try:
            resp = await self.llm.chat(SYSTEM_ANSWER_PROMPT, user)
            return json.loads(resp["content"])
        except Exception:
            return {
                "answer": f"The product {product.name} was found with {len(offers)} offer(s). "
                          f"Price range: €{min((o['price'] for o in offers), default=0):.2f} - "
                          f"€{max((o['price'] for o in offers), default=0):.2f}. "
                          f"{'Price history is available (' + price_analysis.trend + ' trend).' if price_analysis.has_history else 'No price history available.'}",
                "observed_facts": [f"Product found: {product.name}", f"{len(offers)} offers found"],
                "hypotheses": [],
                "risks": [],
                "recommendations": [
                    "Add product to watchlist for price tracking" if not price_analysis.has_history else "Monitor price trend"
                ],
                "confidence": "medium",
                "missing_information": ["price_history"] if not price_analysis.has_history else [],
                "sources_used": ["database"],
            }

    def _mock_answer(self, product, offers, price_analysis, intent):
        if not offers:
            return {
                "answer": f"Product **{product.name}** ({product.brand or 'N/A'}) found in our database. "
                          f"No current offers are tracked yet.",
                "observed_facts": [f"Product {product.name} exists in database", "No offers found"],
                "hypotheses": [],
                "risks": [],
                "recommendations": [
                    "Run an equivalent product analysis to discover market offers",
                    "Add the product to price tracking",
                ],
                "confidence": "medium",
                "missing_information": ["offers", "price_history"],
                "sources_used": ["database"],
            }
        min_price = min(o["price"] for o in offers)
        max_price = max(o["price"] for o in offers)
        cheapest = [o for o in offers if o["price"] == min_price][0]
        price_trend = price_analysis.trend if price_analysis.has_history else "unknown"
        answer = (
            f"**{product.name}** ({product.brand or 'N/A'}) — {len(offers)} offre(s) trouvée(s).\n\n"
            f"Prix : de **€{min_price:.2f}** à **€{max_price:.2f}**.\n"
            f"Meilleur prix : **€{min_price:.2f}** chez **{cheapest.get('merchant', 'inconnu')}**.\n"
        )
        if price_analysis.has_history:
            answer += f"Historique de prix disponible. Tendances : {price_trend}.\n"
        else:
            answer += "Aucun historique de prix disponible.\n"
        return {
            "answer": answer,
            "observed_facts": [
                f"Product: {product.name}",
                f"Brand: {product.brand}",
                f"Category: {product.category}",
                f"Offers found: {len(offers)}",
                f"Price range: €{min_price:.2f} - €{max_price:.2f}",
            ],
            "hypotheses": [],
            "risks": [],
            "recommendations": [
                f"Consider {cheapest.get('merchant', 'the cheapest')} for best price",
                "Add to watchlist for price alerts",
            ],
            "confidence": "high" if offers else "medium",
            "missing_information": [] if price_analysis.has_history else ["price_history"],
            "sources_used": ["database"],
        }

    async def _handle_product_found(
        self, product: Product, message: str, intent: str,
        sources: list[str], actions: list[str],
    ) -> ChatResponse:
        offers = await self._get_offers(str(product.id))
        price_history = await self._get_price_history(str(product.id))
        sources.extend(["offers", "price_history"] if price_history else ["offers"])
        actions.extend(["offers_retrieved"])
        if price_history:
            actions.append("price_history_retrieved")
        answer_data = await self._generate_answer(product, offers, price_history, intent)
        price_analysis = self._compute_price_analysis(product, price_history)
        missing = answer_data.get("missing_information", [])
        if not price_history and "price_history" not in missing:
            missing.append("price_history")
        return ChatResponse(
            answer=answer_data.get("answer", "Analysis completed."),
            intent=intent,
            offers=offers,
            price_analysis=price_analysis,
            market_analysis=MarketAnalysis(
                observed_facts=answer_data.get("observed_facts", []),
                hypotheses=answer_data.get("hypotheses", []),
                risks=answer_data.get("risks", []),
                recommendations=answer_data.get("recommendations", []),
            ),
            confidence=answer_data.get("confidence", "medium"),
            sources_used=list(set(sources)),
            actions_triggered=actions,
            missing_information=missing,
        )

    async def _handle_no_product(
        self, message: str, intent: str,
        sources: list[str], actions: list[str],
    ) -> ChatResponse:
        actions.append("product_not_found_in_db")
        if not _is_electrical(message):
            logger.info(f"Non-electrical question: {message[:80]}...")
            return ChatResponse(
                answer="I specialize in electrical products. Could you provide a product description "
                       "with a brand and technical specifications? (e.g., 'Disjoncteur Legrand 16A 6kA')",
                intent=intent,
                confidence="low",
                sources_used=sources,
                actions_triggered=actions,
                missing_information=["electrical_product_description"],
            )
        try:
            inferred = _infer_product_attributes(description=message)
        except Exception:
            inferred = {}
        answer_data = await self._trigger_equivalent_analysis(message, inferred)
        return ChatResponse(
            answer=answer_data.get("answer", "No equivalent products found."),
            intent=intent,
            market_analysis=MarketAnalysis(
                observed_facts=answer_data.get("observed_facts", []),
                hypotheses=answer_data.get("hypotheses", []),
                risks=answer_data.get("risks", []),
                recommendations=answer_data.get("recommendations", []),
            ),
            confidence=answer_data.get("confidence", "low"),
            sources_used=list(set(sources + answer_data.get("sources_used", []))),
            actions_triggered=actions + ["equivalent_analysis_triggered"],
            missing_information=answer_data.get("missing_information", []),
        )

    async def _trigger_equivalent_analysis(self, message: str, inferred: dict) -> dict:
        sources = ["serpapi"]
        try:
            results = await search_serpapi(message[:200])
        except Exception as e:
            logger.warning(f"SerpApi search failed in chat: {e}")
            return {
                "answer": "I could not find this product in our database, and the external search failed. "
                          "Please verify the product reference or add more details (brand, reference number).",
                "observed_facts": ["Product not found in database", "External search failed"],
                "hypotheses": ["Product may not be available on Google Shopping"],
                "risks": ["Cannot provide pricing without data"],
                "recommendations": ["Add product reference manually", "Try a different search query"],
                "confidence": "low",
                "missing_information": ["product_reference", "brand", "technical_specs"],
                "sources_used": sources,
            }
        if not results:
            return {
                "answer": "No equivalent products found online. Consider adding more details.",
                "observed_facts": ["No external results"],
                "hypotheses": ["Product may be niche or unavailable on Google Shopping"],
                "risks": [],
                "recommendations": ["Add brand and reference", "Try a different description"],
                "confidence": "low",
                "missing_information": ["brand", "reference", "technical_specs"],
                "sources_used": sources,
            }
        try:
            normalized = normalize_candidates(results)
            product_placeholder = type("Product", (), {
                "name": inferred.get("name", message[:100]),
                "description": message,
                "category": inferred.get("category", "unknown"),
                "brand": inferred.get("brand"),
                "target_price": None,
                "currency": "EUR",
                "current_a": inferred.get("specs", {}).get("current_a"),
                "poles": inferred.get("specs", {}).get("poles"),
            })()
            scored = score_candidates(product_placeholder, normalized)
        except Exception as e:
            logger.warning(f"Normalize/score failed in chat: {e}")
            scored = []
        if self.is_mock:
            return self._mock_analysis_result(inferred, results, scored)
        return await self._llm_analysis_result(inferred, results, scored, sources)

    def _mock_analysis_result(self, inferred: dict, results: list, scored: list) -> dict:
        n = len(results)
        top = scored[:5] if scored else results[:3]
        answer = (
            f"**{n} produit(s) trouvé(s)** sur Google Shopping.\n"
            f"Catégorie détectée : {inferred.get('category', 'inconnue')}.\n"
            f"Marque détectée : {inferred.get('brand', 'inconnue')}.\n\n"
        )
        if top:
            prices = [s.get("price", 0) for s in top if s.get("price")]
            if prices:
                answer += (
                    f"Fourchette de prix : **€{min(prices):.2f}** à **€{max(prices):.2f}**.\n"
                    f"Meilleurs résultats trouvés :\n"
                )
                for s in top[:3]:
                    answer += (
                        f"- {s.get('title', 'N/A')[:60]} — "
                        f"€{s.get('price', '?'):.2f} "
                        f"(score: {s.get('score', 0):.2f})\n"
                    )
        else:
            answer += "Aucun résultat détaillé disponible.\n"
        return {
            "answer": answer,
            "observed_facts": [f"{n} products found online", f"Category: {inferred.get('category', 'unknown')}"],
            "hypotheses": ["External search is a snapshot; prices may vary"],
            "risks": ["Check product compatibility before purchasing"],
            "recommendations": [
                "Compare specs carefully",
                "Check merchant ratings before purchasing",
            ],
            "confidence": "medium" if top else "low",
            "missing_information": ["price_history"] if not scored else [],
            "sources_used": ["serpapi"],
        }

    async def _llm_analysis_result(self, inferred: dict, results: list, scored: list, sources: list) -> dict:
        n = len(results)
        top = scored[:5] if scored else results[:3]
        product_info = (
            f"Detected category: {inferred.get('category', 'unknown')}\n"
            f"Detected brand: {inferred.get('brand', 'unknown')}\n"
            f"Detected specs: {inferred.get('specs', {})}\n"
        )
        results_info = f"Total products found: {n}\n"
        if top:
            results_info += "Top results:\n"
            for s in top:
                results_info += (
                    f"  - {s.get('title', 'N/A')[:80]} | "
                    f"€{s.get('price', 0):.2f} | score={s.get('score', 0):.2f}\n"
                )
        else:
            results_info += "No scored results available.\n"
        user = f"{product_info}\n{results_info}"
        try:
            resp = await self.llm.chat(SYSTEM_ANALYSIS_ANSWER_PROMPT, user)
            data = json.loads(resp["content"])
            data["sources_used"] = sources
            return data
        except Exception:
            return {
                "answer": _safe_analysis_answer(n, top),
                "observed_facts": [f"{n} products found", f"Category: {inferred.get('category', 'unknown')}"],
                "hypotheses": [],
                "risks": [],
                "recommendations": ["Verify specs match your needs"],
                "confidence": "medium" if top else "low",
                "missing_information": [],
                "sources_used": sources,
            }

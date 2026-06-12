import json
import logging
import time
import traceback
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.chat_memory import (
    get_or_create_conversation, save_message, load_recent_messages,
    load_conversation_summary, build_chat_context, maybe_update_conversation_summary,
    get_conversation_context,
)
from api.chat_schemas import (
    ChatResponse, ProductBrief, PriceAnalysis, MarketAnalysis,
)
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AnalysisStatus, ChatConversation, ChatMessage
from api.llm_client import get_llm_client, search_serpapi
from worker.tasks import normalize_candidates, score_candidates
from worker.scoring import (
    _infer_product_attributes, _extract_brand_from_title,
    _is_electrical, ELECTRICAL_BRANDS, CATEGORY_DISPLAY_NAMES,
)

logger = logging.getLogger("api.chat")

# Intents that warrant fetching equivalent analysis results
ANALYSIS_INTENTS = {
    "equivalent_products_search",
    "product_comparison",
    "price_analysis",
    "price_history_analysis",
    "market_analysis",
}

DEEP_FOLLOWUP_KEYWORDS = [
    # English
    "go deeper", "deeper", "more detail", "more details", "list all", "list candidates",
    "all candidates", "which one is best", "which is best", "best price", "all of them",
    "enumerate", "show all", "show me all", "weak", "partial", "dig deeper",
    "all results", "every candidate", "every result",
    # French — deepening / elaboration requests
    "plus de détail", "plus de detail", "plus d'info", "plus d'information",
    "analyse plus", "analyser plus", "approfondir", "approfondis",
    "développe", "développer", "expliquer", "explique moi",
    "un peu plus", "encore plus", "creuse", "creuser",
    "dis m'en plus", "dis moi plus", "donne moi plus", "donne-moi plus",
    "plus en détail", "plus en detail",
    "liste tous", "liste toutes", "montre moi tous", "montre tous",
    "tous les candidats", "toutes les offres",
    "meilleur rapport", "rapport qualité",
    "créer un peu plus", "creer un peu plus",
]


def _is_deep_followup(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in DEEP_FOLLOWUP_KEYWORDS)


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        text = inner.strip()
    return json.loads(text)


def _stream_extract_answer(state: dict, new_text: str) -> str:
    """
    Incrementally extract the 'answer' string value from a streaming JSON response.
    state = {"buf": str, "in_answer": bool, "done": bool, "esc": bool}
    Returns displayable text decoded from the answer value.
    """
    import re
    if state["done"]:
        return ""
    state["buf"] += new_text
    display: list[str] = []
    if not state["in_answer"]:
        m = re.search(r'"answer"\s*:\s*"', state["buf"])
        if not m:
            return ""
        state["in_answer"] = True
        remaining = state["buf"][m.end():]
        state["buf"] = ""
        new_text = remaining
    for c in new_text:
        if state["done"]:
            break
        if state["esc"]:
            if c == "n":
                display.append("\n")
            elif c == "t":
                display.append("\t")
            elif c in ('"', "\\", "/"):
                display.append(c)
            else:
                display.append(c)
            state["esc"] = False
        elif c == "\\":
            state["esc"] = True
        elif c == '"':
            state["done"] = True
        else:
            display.append(c)
    return "".join(display)

# Minimum score threshold for classifying an offer as an exact source-product match
EXACT_MATCH_SCORE_THRESHOLD = 0.88

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

OFFER VALIDATION RULES (YOU MUST FOLLOW):
- Do NOT say "Product X is available at price Y" unless the offer is a confirmed exact match for Product X (same brand AND score >= 0.88).
- If the offer belongs to another brand or is an equivalent product, say: "This appears to be an equivalent/candidate, not a confirmed exact listing for the source product."
- When equivalent_analysis data is present, ALWAYS summarize: candidate count, valid match count, best match score, best price among strong equivalents, top cross-brand equivalents, and confidence level.
- Weak candidates (low spec_quality or vague) must NOT be presented as confirmed products.
- If best_match_score < 0.88, state clearly that confidence is limited and the result may not be an exact match.

CONVERSATION MEMORY RULES:
- Conversation history contains user statements and previous assistant guesses.
- Do NOT treat them as verified facts.
- Verified facts come ONLY from the "Verified database context" section.
- Prices, stock, references must come from database context, not from memory.
- If a fact is only in conversation memory, say "according to our conversation" or "you mentioned".
- If uncertain, state clearly that the information is not verified.

DEEP ANALYSIS RULES (apply when user asks to go deeper, list candidates, compare, or re-analyse):
- NEVER repeat the same summary you already gave. If the user pushes for more, go further.
- Enumerate ALL candidate buckets present in context: reliable_candidates, partial_candidates, weak_candidates_sample.
- For EACH candidate write one line inside `answer` using the actual data values: "• [bucket] <actual product title from data, max 60 chars> — €<actual price> — score=<actual score> — <specific reason why confidence is limited>".
- Do NOT end `answer` with a colon or a heading sentence that promises a list but delivers nothing.
- Put ALL enumeration inline inside the `answer` string — not in observed_facts.
- Explain WHY a score is low: vague specs, wrong brand, missing technical data, etc.
- Give an opinionated recommendation even under uncertainty — just be explicit about confidence.
- Never refuse to discuss weak/partial candidates — label them clearly and let the user decide.
- If the user asks "which one is best", pick one and justify it, even if confidence is limited.

Output JSON with these keys:
- answer: str — natural language expert answer (as long as needed to fully enumerate candidates; minimum 5 sentences when candidates are available)
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

OFFER VALIDATION RULES (YOU MUST FOLLOW):
- Do NOT present any candidate as a confirmed exact listing for the source product unless score >= 0.88 and same brand.
- If best_match_score < 0.88, state that confidence is limited.
- Weak candidates (is_vague=true or spec_quality < 0.25) must be excluded from recommendations.
- Always state the number of candidates, valid matches, and best match score.

CONVERSATION MEMORY RULES:
- Conversation history contains user statements and previous assistant guesses.
- Do NOT treat them as verified facts.
- Verified facts come ONLY from the "Verified database context" section.
- Prices, stock, references must come from database context, not from memory.
- If uncertain, state clearly that the information is not verified.

DEEP ANALYSIS RULES (apply when user asks to go deeper, list candidates, compare, or re-analyse):
- NEVER repeat the same summary you already gave. If the user pushes for more, go further.
- Enumerate ALL candidate buckets present in context: reliable_candidates, partial_candidates, weak_candidates_sample.
- For EACH candidate write one line inside `answer` using the actual data values: "• [bucket] <actual product title from data, max 60 chars> — €<actual price> — score=<actual score> — <specific reason why confidence is limited>".
- Do NOT end `answer` with a colon or a heading sentence that promises a list but delivers nothing.
- Put ALL enumeration inline inside the `answer` string — not in observed_facts.
- Explain WHY a score is low: vague specs, wrong brand, missing technical data, etc.
- Give an opinionated recommendation even under uncertainty — just be explicit about confidence.
- Never refuse to discuss weak/partial candidates — label them clearly and let the user decide.
- If the user asks "which one is best", pick one and justify it, even if confidence is limited.

Output JSON with these keys:
- answer: str — natural language analysis (as long as needed to fully enumerate candidates; minimum 5 sentences when candidates are available)
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
    if any(w in message_lower for w in ["compare", "comparer", "comparaison", "comparison", "vs", "versus"]):
        return {"intent": "product_comparison", "confidence": 0.7}
    if any(w in message_lower for w in ["equivalent", "équivalent", "equivalents", "équivalents",
                                          "alternative", "remplacer", "similar", "trouve"]):
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
        user_id: str | None = None,
    ) -> ChatResponse:
        start_time = time.time()
        actions: list[str] = []
        sources: list[str] = []

        conv, conv_action = await get_or_create_conversation(self.db, conversation_id, user_id, message)
        conversation_id = str(conv.id)
        self._current_conversation_id = conversation_id
        await save_message(self.db, conversation_id, "user", message)
        actions.append(f"conversation_{conv_action}")
        actions.append("user_message_saved")

        conv_context = await get_conversation_context(self.db, conversation_id)
        self._conversation_summary = conv_context["summary"]
        recent = conv_context["recent_messages"]
        self._recent_messages = recent[:-1] if recent else []
        self._conversation_context = conv_context

        # Reuse product_id from previous messages when not supplied in a follow-up
        if not product_id and conv_action == "existing" and conv_context.get("product_id"):
            product_id = conv_context["product_id"]
            actions.append("product_id_reused_from_conversation")

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
            resp.product_id = str(product.id)

        assistant_metadata = {
            "product_id": str(product.id) if product else None,
            "product_name": product.name if product else None,
            "offers": resp.offers[:10] if resp.offers else [],
            "equivalents": resp.equivalents[:10] if resp.equivalents else [],
            "weak_candidates": resp.weak_candidates[:10] if resp.weak_candidates else [],
            "price_analysis": resp.price_analysis.model_dump() if resp.price_analysis else None,
            "sources_used": resp.sources_used,
            "actions_triggered": actions,
        }
        assistant_msg = await save_message(
            self.db, conversation_id, "assistant", resp.answer, metadata=assistant_metadata
        )
        await maybe_update_conversation_summary(self.db, conversation_id, self.llm)
        resp.conversation_id = conversation_id
        resp.message_id = str(assistant_msg.id)
        resp.intent = intent
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Chat processed | intent={intent} | "
            f"product_found={'yes' if product else 'no'} | "
            f"latency_ms={latency_ms} | confidence={resp.confidence}"
        )
        return resp

    async def process_stream(
        self,
        message: str,
        product_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ):
        """Async generator yielding SSE-style dicts:
          {"type": "thinking", "text": "..."}
          {"type": "token", "text": "..."}   ← streams the answer
          {"type": "done", "data": {...}}    ← full ChatResponse payload
        """
        def thinking(text: str, step: str = "") -> dict:
            return {"type": "thinking", "step": step, "text": text}

        actions: list[str] = []
        sources: list[str] = []

        conv, conv_action = await get_or_create_conversation(self.db, conversation_id, user_id, message)
        conversation_id = str(conv.id)
        self._current_conversation_id = conversation_id
        await save_message(self.db, conversation_id, "user", message)
        actions.extend([f"conversation_{conv_action}", "user_message_saved"])

        conv_context = await get_conversation_context(self.db, conversation_id)
        self._conversation_summary = conv_context["summary"]
        recent = conv_context["recent_messages"]
        self._recent_messages = recent[:-1] if recent else []
        self._conversation_context = conv_context

        if not product_id and conv_action == "existing" and conv_context.get("product_id"):
            product_id = conv_context["product_id"]
            actions.append("product_id_reused_from_conversation")

        yield thinking("Classification de l'intention…", "intent")
        intent, intent_conf = await self._classify_intent(message)
        yield thinking(f"Intention : {intent} ({intent_conf:.0%})", "intent_done")
        sources.append("intent_classifier")
        actions.append(f"intent_classified_as_{intent}")

        yield thinking("Recherche du produit…", "product")
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
        if product:
            yield thinking(f"Produit : {product.name[:60]}", "product_found")
        else:
            yield thinking("Produit non trouvé — recherche sur le marché…", "no_product")

        # ── product found path ──────────────────────────────────────────
        resp: ChatResponse
        if product:
            yield thinking("Récupération des offres…", "offers")
            offers = await self._get_offers(str(product.id))
            price_history = await self._get_price_history(str(product.id))
            sources.extend(["offers", "price_history"] if price_history else ["offers"])
            if price_history:
                actions.append("price_history_retrieved")
            yield thinking(f"{len(offers)} offre(s) trouvée(s)", "offers_done")

            eq_analysis: dict | None = None
            equivalents: list[dict] = []
            weak_candidates: list[dict] = []
            trigger_analysis = intent in ANALYSIS_INTENTS or _is_deep_followup(message)
            if trigger_analysis:
                yield thinking("Récupération de l'analyse d'équivalents…", "analysis")
                eq_analysis = await self._get_latest_equivalent_analysis(str(product.id))
                if eq_analysis:
                    yield thinking(
                        f"Analyse : {eq_analysis['candidate_count']} candidats "
                        f"(score max {eq_analysis.get('best_match_score', 0) or 0:.2f})",
                        "analysis_done",
                    )
                    sources.append("equivalent_analysis")
                    actions.append("equivalent_analysis_retrieved")
                    equivalents = (
                        eq_analysis.get("cross_brand_equivalents", [])
                        + eq_analysis.get("partial_spec_equivalents", [])
                    )
                    weak_candidates = eq_analysis.get("weak_candidates", [])
                    stored_total = len(equivalents) + len(weak_candidates)
                    if _is_deep_followup(message) and stored_total < 3:
                        yield thinking("Peu de données stockées — recherche live…", "live_search")
                        live = await self._live_eq_search(product)
                        if live:
                            sources.append("serpapi")
                            actions.append("live_augmentation_completed")
                            eq_analysis = {
                                **eq_analysis,
                                "cross_brand_equivalents": (
                                    eq_analysis.get("cross_brand_equivalents", [])
                                    + live.get("cross_brand_equivalents", [])
                                ),
                                "partial_spec_equivalents": (
                                    eq_analysis.get("partial_spec_equivalents", [])
                                    + live.get("partial_spec_equivalents", [])
                                ),
                                "weak_candidates": (
                                    eq_analysis.get("weak_candidates", [])
                                    + live.get("weak_candidates", [])
                                ),
                            }
                            equivalents = (
                                eq_analysis.get("cross_brand_equivalents", [])
                                + eq_analysis.get("partial_spec_equivalents", [])
                            )
                            weak_candidates = eq_analysis.get("weak_candidates", [])
                            yield thinking(
                                f"Live : {live['candidate_count']} candidats supplémentaires",
                                "live_done",
                            )
                else:
                    actions.append("no_db_analysis_triggering_live_search")
                    yield thinking("Pas d'analyse en cache — recherche live…", "live_search")
                    eq_analysis = await self._live_eq_search(product)
                    if eq_analysis:
                        sources.append("serpapi")
                        actions.append("live_equivalent_search_completed")
                        equivalents = (
                            eq_analysis.get("cross_brand_equivalents", [])
                            + eq_analysis.get("partial_spec_equivalents", [])
                        )
                        weak_candidates = eq_analysis.get("weak_candidates", [])
                        yield thinking(
                            f"Live : {eq_analysis['candidate_count']} candidats trouvés",
                            "live_done",
                        )
                    else:
                        actions.append("live_search_also_failed")

            yield thinking("Génération de la réponse…", "generating")
            price_analysis = self._compute_price_analysis(product, price_history)

            if self.is_mock:
                result = self._mock_answer(product, offers, price_analysis, intent, eq_analysis)
                yield {"type": "token", "text": result["answer"]}
                answer_data = result
            else:
                product_context = self._build_product_context(
                    product, offers, price_history, price_analysis, intent, eq_analysis, message
                )
                context_messages = build_chat_context(
                    SYSTEM_ANSWER_PROMPT,
                    self._conversation_summary,
                    self._recent_messages,
                    product_context,
                    self._build_current_msg(message, intent, product.name),
                )
                full_content = ""
                stream_state = {"buf": "", "in_answer": False, "done": False, "esc": False}
                try:
                    async for token in self.llm.stream_chat_messages(context_messages):
                        full_content += token
                        visible = _stream_extract_answer(stream_state, token)
                        if visible:
                            yield {"type": "token", "text": visible}
                    answer_data = _extract_json(full_content)
                except Exception as e:
                    logger.error(f"process_stream LLM failed: {e!r}", exc_info=True)
                    answer_data = self._stream_fallback(product, offers, price_analysis, eq_analysis)
                    yield {"type": "token", "text": answer_data["answer"]}

            missing = answer_data.get("missing_information", [])
            if not price_history and "price_history" not in missing:
                missing.append("price_history")
            resp = ChatResponse(
                answer=answer_data.get("answer", ""),
                intent=intent,
                offers=offers,
                equivalents=equivalents,
                weak_candidates=weak_candidates,
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
            resp.product = ProductBrief(
                id=str(product.id), name=product.name,
                brand=product.brand, category=product.category,
                reference=product.sku,
            )
            resp.product_id = str(product.id)

        # ── no product path ─────────────────────────────────────────────
        else:
            if not _is_electrical(message):
                resp = ChatResponse(
                    answer="Je suis spécialisé en produits électriques. Pouvez-vous fournir une référence avec marque et specs ? (ex: Disjoncteur Legrand 16A 6kA)",
                    intent=intent, confidence="low",
                    sources_used=sources, actions_triggered=actions,
                    missing_information=["electrical_product_description"],
                )
            else:
                yield thinking("Recherche Google Shopping…", "serp_search")
                try:
                    inferred = _infer_product_attributes(description=message)
                except Exception:
                    inferred = {}
                answer_data, scored = await self._trigger_equivalent_analysis(message, inferred)
                products_found = [
                    ProductBrief(name=s.get("title", "")[:100], brand=s.get("brand"))
                    for s in scored[:5] if s.get("title")
                ]
                yield thinking(f"{len(scored)} résultats scorés", "serp_done")
                resp = ChatResponse(
                    answer=answer_data.get("answer", ""),
                    intent=intent,
                    products_found=products_found,
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
                for token in resp.answer.split():
                    yield {"type": "token", "text": token + " "}

        # ── persist and finish ──────────────────────────────────────────
        assistant_metadata = {
            "product_id": str(product.id) if product else None,
            "product_name": product.name if product else None,
            "offers": resp.offers[:10],
            "equivalents": resp.equivalents[:10],
            "weak_candidates": resp.weak_candidates[:10],
            "price_analysis": resp.price_analysis.model_dump() if resp.price_analysis else None,
            "sources_used": resp.sources_used,
            "actions_triggered": actions,
        }
        assistant_msg = await save_message(
            self.db, conversation_id, "assistant", resp.answer, metadata=assistant_metadata
        )
        await maybe_update_conversation_summary(self.db, conversation_id, self.llm)
        resp.conversation_id = conversation_id
        resp.message_id = str(assistant_msg.id)
        resp.intent = intent
        yield {"type": "done", "data": resp.model_dump()}

    # ── helpers shared by process() and process_stream() ────────────────

    def _build_product_context(
        self,
        product: Product,
        offers: list[dict],
        price_history: list[dict],
        price_analysis: "PriceAnalysis",
        intent: str,
        eq_analysis: dict | None,
        user_message: str | None = None,
    ) -> dict:
        ctx: dict = {
            "product": product.name,
            "brand": product.brand,
            "category": product.category,
            "sku": product.sku,
            "specs": {
                "current_a": product.current_a,
                "poles": product.poles,
                "curve": product.curve,
                "breaking_capacity_ka": product.breaking_capacity_ka,
            },
            "exact_offers_count": len(offers),
            "exact_offers": [
                {"price": o["price"], "merchant": o.get("merchant"), "in_stock": o.get("in_stock")}
                for o in offers[:10]
            ],
            "price_history": {
                "has_data": price_analysis.has_history,
                "min_price": price_analysis.min_price,
                "max_price": price_analysis.max_price,
                "median_price": price_analysis.median_price,
                "trend": price_analysis.trend,
            },
            "intent": intent,
        }
        if eq_analysis:
            cross_brand = eq_analysis.get("cross_brand_equivalents", [])
            partial = eq_analysis.get("partial_spec_equivalents", [])
            weak = eq_analysis.get("weak_candidates", [])

            def _entry(e: dict, bucket: str) -> dict:
                return {
                    "bucket": bucket, "title": e["title"][:80], "price": e["price"],
                    "currency": e.get("currency", "EUR"), "brand": e.get("brand"),
                    "score": e["score"], "spec_quality": e.get("spec_quality", 0),
                    "is_vague": e.get("is_vague", False), "classification": e.get("classification", ""),
                }

            best_score = eq_analysis.get("best_match_score")
            ctx["equivalent_analysis"] = {
                "candidate_count": eq_analysis.get("candidate_count", 0),
                "valid_match_count": eq_analysis.get("valid_match_count", 0),
                "best_match_score": best_score,
                "price_confidence": eq_analysis.get("price_confidence"),
                "recommendation": eq_analysis.get("recommendation"),
                "confidence_limited": best_score is not None and best_score < EXACT_MATCH_SCORE_THRESHOLD,
                "reliable_candidates": [_entry(e, "reliable") for e in cross_brand[:10]],
                "partial_candidates": [_entry(e, "partial") for e in partial[:10]],
                "weak_candidates_sample": [
                    _entry(e, "weak")
                    for e in sorted(weak, key=lambda x: x.get("score", 0), reverse=True)[:10]
                ],
                "note": "reliable = strong spec match; partial = incomplete specs; weak = vague or low-quality",
            }
        conv_ctx = getattr(self, "_conversation_context", {})
        if not offers and conv_ctx.get("offers"):
            ctx["previous_offers_from_conversation"] = conv_ctx["offers"]
        if not (eq_analysis and eq_analysis.get("cross_brand_equivalents")) and conv_ctx.get("equivalents"):
            ctx["previous_equivalents_from_conversation"] = conv_ctx["equivalents"][:10]
        if not (eq_analysis and eq_analysis.get("weak_candidates")) and conv_ctx.get("weak_candidates"):
            ctx["previous_weak_candidates_from_conversation"] = conv_ctx["weak_candidates"][:10]
        return ctx

    def _build_current_msg(self, user_message: str | None, intent: str, product_name: str) -> str:
        if user_message:
            deep = _is_deep_followup(user_message)
            extra = (
                "\n[System instruction: enumerate every candidate from equivalent_analysis "
                "inline in the answer field using bullet points — do not summarize or skip any.]"
                if deep else ""
            )
            return f"{user_message}{extra}\n\n[System: intent={intent}, product={product_name}]"
        return f"Intent: {intent}\nProvide a detailed analysis for product: {product_name}."

    def _stream_fallback(
        self, product: Product, offers: list[dict],
        price_analysis: "PriceAnalysis", eq_analysis: dict | None,
    ) -> dict:
        min_p = min((o["price"] for o in offers), default=0)
        max_p = max((o["price"] for o in offers), default=0)
        lines = [
            f"{product.name} — {len(offers)} offre(s). Prix : €{min_p:.2f}–€{max_p:.2f}."
        ]
        if eq_analysis:
            all_cands = (
                [("reliable", e) for e in eq_analysis.get("cross_brand_equivalents", [])]
                + [("partial", e) for e in eq_analysis.get("partial_spec_equivalents", [])]
                + [("weak", e) for e in eq_analysis.get("weak_candidates", [])]
            )
            if all_cands:
                lines.append(f"\n{len(all_cands)} candidats trouvés :")
                for bucket, e in all_cands[:12]:
                    lines.append(
                        f"• [{bucket}] {e.get('title', '')[:60]} — €{e.get('price', 0):.2f} — score={e.get('score', 0):.2f}"
                    )
        return {
            "answer": "\n".join(lines),
            "observed_facts": [f"{len(offers)} offres exactes"],
            "hypotheses": [], "risks": [],
            "recommendations": ["Vérifiez les specs avant achat"],
            "confidence": "low",
            "missing_information": [],
            "sources_used": ["database"],
        }

    async def _classify_intent(self, message: str) -> tuple[str, float]:
        if self.is_mock:
            result = _mock_intent(message)
            return result["intent"], result["confidence"]
        try:
            resp = await self.llm.chat(SYSTEM_INTENT_PROMPT, f"Question: {message}")
            data = _extract_json(resp["content"])
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
        """Return only confirmed exact offers for the source product.

        An offer qualifies as an exact source-product offer when:
        - raw_data is absent (legacy offer, returned as-is for backward compat), OR
        - same brand AND score >= EXACT_MATCH_SCORE_THRESHOLD, OR
        - classification == "exact_match" AND score >= EXACT_MATCH_SCORE_THRESHOLD
        """
        stmt = select(Offer).where(Offer.product_id == product_id).order_by(Offer.price)
        result = await self.db.execute(stmt)
        exact_offers = []
        for o in result.scalars():
            raw = o.raw_data or {}
            if not raw:
                # Legacy offer without scoring data — include as-is
                exact_offers.append({
                    "title": o.title,
                    "price": o.price,
                    "currency": o.currency,
                    "merchant": o.merchant,
                    "url": o.url,
                    "in_stock": o.in_stock,
                })
                continue
            score = raw.get("score", 0)
            is_same_brand = raw.get("is_same_brand", False)
            classification = raw.get("classification", "")
            if (
                (is_same_brand and score >= EXACT_MATCH_SCORE_THRESHOLD)
                or (classification == "exact_match" and score >= EXACT_MATCH_SCORE_THRESHOLD)
            ):
                exact_offers.append({
                    "title": o.title,
                    "price": o.price,
                    "currency": o.currency,
                    "merchant": o.merchant,
                    "url": o.url,
                    "in_stock": o.in_stock,
                })
        return exact_offers

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

    async def _get_latest_equivalent_analysis(self, product_id: str) -> dict | None:
        """Retrieve the latest completed equivalent analysis for a product.

        Returns a dict with:
        - run_id, candidate_count, valid_match_count, best_match_score, price_confidence,
          recommendation, cross_brand_equivalents, partial_spec_equivalents, weak_candidates
        Returns None if no completed analysis run exists.
        """
        stmt = (
            select(AnalysisRun)
            .where(AnalysisRun.product_id == product_id)
            .where(AnalysisRun.status == AnalysisStatus.completed)
            .order_by(AnalysisRun.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        run = result.scalar_one_or_none()
        if not run:
            return None

        stmt = (
            select(Offer)
            .where(Offer.product_id == product_id)
            .where(Offer.source == "analysis")
            .order_by(Offer.price)
        )
        result = await self.db.execute(stmt)
        offers = list(result.scalars())

        cross_brand: list[dict] = []
        partial: list[dict] = []
        weak: list[dict] = []

        for o in offers:
            raw = o.raw_data or {}
            quality_bucket = raw.get("quality_bucket", "")
            is_same_brand = raw.get("is_same_brand", False)
            entry = {
                "title": o.title,
                "price": o.price,
                "currency": o.currency,
                "merchant": o.merchant,
                "url": o.url,
                "score": raw.get("score", 0),
                "spec_quality": raw.get("spec_quality", 0),
                "classification": raw.get("classification", "functional_equivalent"),
                "spec_match": raw.get("spec_match", "functional_equivalent"),
                "is_vague": raw.get("is_vague", False),
                "brand": raw.get("brand"),
                "is_same_brand": is_same_brand,
            }
            if quality_bucket == "reliable" and not is_same_brand:
                cross_brand.append(entry)
            elif quality_bucket == "partial":
                partial.append(entry)
            else:
                weak.append(entry)

        return {
            "run_id": str(run.id),
            "candidate_count": run.candidate_count or 0,
            "valid_match_count": run.valid_match_count or 0,
            "cross_brand_equivalents": cross_brand,
            "partial_spec_equivalents": partial,
            "weak_candidates": weak,
            "best_match_score": run.best_match_score,
            "price_confidence": run.price_confidence,
            "recommendation": (run.final_decision or {}).get("summary"),
        }

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
        eq_analysis: dict | None = None,
        user_message: str | None = None,
    ) -> dict:
        price_analysis = self._compute_price_analysis(product, price_history)
        if self.is_mock:
            return self._mock_answer(product, offers, price_analysis, intent, eq_analysis)
        product_context = self._build_product_context(
            product, offers, price_history, price_analysis, intent, eq_analysis, user_message
        )
        current_msg = self._build_current_msg(user_message, intent, product.name)
        context_messages = build_chat_context(
            SYSTEM_ANSWER_PROMPT,
            getattr(self, "_conversation_summary", None),
            getattr(self, "_recent_messages", []),
            product_context,
            current_msg,
        )
        try:
            resp = await self.llm.chat_messages(context_messages)
            return _extract_json(resp["content"])
        except Exception as _gen_err:
            logger.error(f"_generate_answer LLM call failed: {_gen_err!r}", exc_info=True)
            # Build a richer fallback that at least shows offers + candidates
            min_p = min((o["price"] for o in offers), default=0)
            max_p = max((o["price"] for o in offers), default=0)
            fallback_lines = [
                f"{product.name} — {len(offers)} confirmed offer(s). "
                f"Price range: €{min_p:.2f}–€{max_p:.2f}."
            ]
            if eq_analysis:
                cross = eq_analysis.get("cross_brand_equivalents", [])
                partial_list = eq_analysis.get("partial_spec_equivalents", [])
                weak_list = eq_analysis.get("weak_candidates", [])
                all_cands = (
                    [("reliable", e) for e in cross]
                    + [("partial", e) for e in partial_list]
                    + [("weak", e) for e in weak_list]
                )
                if all_cands:
                    fallback_lines.append(f"\n{len(all_cands)} market candidates found:")
                    for bucket, e in all_cands[:12]:
                        fallback_lines.append(
                            f"• [{bucket}] {e.get('title', '')[:60]} — €{e.get('price', 0):.2f} — score={e.get('score', 0):.2f}"
                        )
            return {
                "answer": "\n".join(fallback_lines),
                "observed_facts": [f"Product found: {product.name}", f"{len(offers)} exact offers found"],
                "hypotheses": [],
                "risks": [],
                "recommendations": ["Verify specs match your needs before purchasing equivalent"],
                "confidence": "low",
                "missing_information": ["price_history"] if not price_analysis.has_history else [],
                "sources_used": ["database"],
            }

    def _mock_answer(self, product, offers, price_analysis, intent, eq_analysis=None):
        cross_brand = (eq_analysis or {}).get("cross_brand_equivalents", [])
        partial = (eq_analysis or {}).get("partial_spec_equivalents", [])
        best_match_score = (eq_analysis or {}).get("best_match_score")
        candidate_count = (eq_analysis or {}).get("candidate_count", 0)
        valid_match_count = (eq_analysis or {}).get("valid_match_count", 0)
        confidence_limited = best_match_score is not None and best_match_score < EXACT_MATCH_SCORE_THRESHOLD

        if not offers and not eq_analysis:
            return {
                "answer": f"Product **{product.name}** ({product.brand or 'N/A'}) found in our database. "
                          f"No confirmed exact offers are tracked yet.",
                "observed_facts": [f"Product {product.name} exists in database", "No confirmed exact offers found"],
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

        answer_parts = [f"**{product.name}** ({product.brand or 'N/A'})"]

        if offers:
            min_price = min(o["price"] for o in offers)
            max_price = max(o["price"] for o in offers)
            cheapest = [o for o in offers if o["price"] == min_price][0]
            answer_parts.append(
                f" — {len(offers)} confirmed exact offer(s) found.\n"
                f"Price: from **€{min_price:.2f}** to **€{max_price:.2f}**.\n"
                f"Best price: **€{min_price:.2f}** at **{cheapest.get('merchant', 'unknown')}**.\n"
            )
            if price_analysis.has_history:
                answer_parts.append(f"Price history available. Trend: {price_analysis.trend}.\n")
            else:
                answer_parts.append("No price history available.\n")
        else:
            answer_parts.append(" — No confirmed exact offers for this product.\n")

        if eq_analysis:
            answer_parts.append(
                f"\n**Equivalent analysis**: {candidate_count} candidate(s) found, "
                f"{valid_match_count} valid match(es).\n"
            )
            if cross_brand:
                best_eq = cross_brand[0]
                answer_parts.append(
                    f"Best equivalent: **{best_eq['title'][:60]}** at **€{best_eq['price']:.2f}** "
                    f"(score: {best_eq['score']:.2f}).\n"
                )
                if confidence_limited:
                    answer_parts.append(
                        f"**Note**: best match score is {best_match_score:.2f} — "
                        f"confidence is limited; this may not be an exact equivalent.\n"
                    )
                if len(cross_brand) > 1:
                    other_brands = ", ".join(
                        f"{e.get('brand', 'Unknown')}" for e in cross_brand[1:3]
                    )
                    answer_parts.append(f"Other cross-brand options: {other_brands}.\n")
            elif partial:
                answer_parts.append(
                    f"Only partial-spec equivalents found ({len(partial)}). "
                    f"Confidence is limited.\n"
                )
            else:
                answer_parts.append("No strong cross-brand equivalents found.\n")

        answer = "".join(answer_parts)

        observed_facts = [
            f"Product: {product.name}",
            f"Brand: {product.brand}",
            f"Category: {product.category}",
            f"Confirmed exact offers: {len(offers)}",
        ]
        if eq_analysis:
            observed_facts += [
                f"Equivalent candidates found: {candidate_count}",
                f"Valid matches (reliable): {valid_match_count}",
                f"Cross-brand equivalents: {len(cross_brand)}",
            ]
            if confidence_limited:
                observed_facts.append(
                    f"Best match score {best_match_score:.2f} is below 0.88 — limited confidence"
                )

        return {
            "answer": answer,
            "observed_facts": observed_facts,
            "hypotheses": [],
            "risks": [],
            "recommendations": [
                "Verify specs match your needs before purchasing equivalent",
            ] if cross_brand else ["Run a fresh equivalent analysis for updated results"],
            "confidence": "low" if confidence_limited else ("high" if offers else "medium"),
            "missing_information": [] if (offers or cross_brand) else ["exact_offers", "price_history"],
            "sources_used": ["database", "equivalent_analysis"] if eq_analysis else ["database"],
        }

    async def _live_eq_search(self, product: Product) -> dict | None:
        """Live SerpAPI search for equivalents when no DB analysis run exists."""
        query = f"{product.name} {product.brand or ''}".strip()[:200]
        try:
            results = await search_serpapi(query)
        except Exception as e:
            logger.warning(f"Live SerpAPI search failed: {e}")
            return None
        if not results:
            return None
        try:
            normalized = normalize_candidates(results)
            scored = score_candidates(product, normalized)
        except Exception:
            scored = []

        cross_brand: list[dict] = []
        partial: list[dict] = []
        weak: list[dict] = []
        for s in scored:
            score = s.get("score", 0)
            is_same_brand = s.get("is_same_brand", False)
            is_vague = s.get("is_vague", False)
            spec_quality = s.get("spec_quality", 0)
            entry = {
                "title": s.get("title", ""),
                "price": s.get("price", 0),
                "currency": s.get("currency", "EUR"),
                "merchant": s.get("merchant", ""),
                "url": s.get("url", ""),
                "score": score,
                "spec_quality": spec_quality,
                "classification": s.get("classification", "functional_equivalent"),
                "spec_match": s.get("spec_match", ""),
                "is_vague": is_vague,
                "brand": s.get("brand"),
                "is_same_brand": is_same_brand,
            }
            if is_same_brand:
                continue  # confirmed exact match — already in offers
            if score >= EXACT_MATCH_SCORE_THRESHOLD and not is_vague:
                cross_brand.append(entry)
            elif spec_quality >= 0.25 and not is_vague:
                partial.append(entry)
            else:
                weak.append(entry)

        best_score = max((s.get("score", 0) for s in scored), default=None)
        valid = [s for s in scored if s.get("score", 0) >= EXACT_MATCH_SCORE_THRESHOLD]
        return {
            "run_id": "live_search",
            "candidate_count": len(results),
            "valid_match_count": len(valid),
            "cross_brand_equivalents": cross_brand,
            "partial_spec_equivalents": partial,
            "weak_candidates": weak,
            "best_match_score": best_score,
            "price_confidence": "low" if not valid else "medium",
            "recommendation": None,
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

        eq_analysis: dict | None = None
        equivalents: list[dict] = []
        weak_candidates: list[dict] = []
        trigger_analysis = intent in ANALYSIS_INTENTS or _is_deep_followup(message)
        if trigger_analysis:
            eq_analysis = await self._get_latest_equivalent_analysis(str(product.id))
            if eq_analysis:
                sources.append("equivalent_analysis")
                actions.append("equivalent_analysis_retrieved")
                equivalents = (
                    eq_analysis.get("cross_brand_equivalents", [])
                    + eq_analysis.get("partial_spec_equivalents", [])
                )
                weak_candidates = eq_analysis.get("weak_candidates", [])
                logger.info(
                    f"Equivalent analysis found | run_id={eq_analysis['run_id']} | "
                    f"cross_brand={len(eq_analysis.get('cross_brand_equivalents', []))} | "
                    f"partial={len(eq_analysis.get('partial_spec_equivalents', []))} | "
                    f"weak={len(eq_analysis.get('weak_candidates', []))}"
                )
                # If DB has sparse stored candidates and user is explicitly asking to enumerate,
                # augment with a live search to give the LLM real candidates to discuss.
                stored_total = (
                    len(eq_analysis.get("cross_brand_equivalents", []))
                    + len(eq_analysis.get("partial_spec_equivalents", []))
                    + len(eq_analysis.get("weak_candidates", []))
                )
                if _is_deep_followup(message) and stored_total < 3:
                    actions.append("augmenting_sparse_db_with_live_search")
                    live = await self._live_eq_search(product)
                    if live:
                        sources.append("serpapi")
                        actions.append("live_augmentation_completed")
                        # Merge live into eq_analysis so _generate_answer sees full picture
                        eq_analysis = {
                            **eq_analysis,
                            "cross_brand_equivalents": (
                                eq_analysis.get("cross_brand_equivalents", [])
                                + live.get("cross_brand_equivalents", [])
                            ),
                            "partial_spec_equivalents": (
                                eq_analysis.get("partial_spec_equivalents", [])
                                + live.get("partial_spec_equivalents", [])
                            ),
                            "weak_candidates": (
                                eq_analysis.get("weak_candidates", [])
                                + live.get("weak_candidates", [])
                            ),
                        }
                        equivalents = (
                            eq_analysis.get("cross_brand_equivalents", [])
                            + eq_analysis.get("partial_spec_equivalents", [])
                        )
                        weak_candidates = eq_analysis.get("weak_candidates", [])
            else:
                actions.append("no_db_analysis_triggering_live_search")
                eq_analysis = await self._live_eq_search(product)
                if eq_analysis:
                    sources.append("serpapi")
                    actions.append("live_equivalent_search_completed")
                    equivalents = (
                        eq_analysis.get("cross_brand_equivalents", [])
                        + eq_analysis.get("partial_spec_equivalents", [])
                    )
                    weak_candidates = eq_analysis.get("weak_candidates", [])
                    logger.info(
                        f"Live search complete | candidates={eq_analysis['candidate_count']} | "
                        f"cross_brand={len(eq_analysis.get('cross_brand_equivalents', []))} | "
                        f"partial={len(eq_analysis.get('partial_spec_equivalents', []))} | "
                        f"weak={len(eq_analysis.get('weak_candidates', []))}"
                    )
                else:
                    actions.append("live_search_also_failed")

        answer_data = await self._generate_answer(product, offers, price_history, intent, eq_analysis, user_message=message)
        price_analysis = self._compute_price_analysis(product, price_history)
        missing = answer_data.get("missing_information", [])
        if not price_history and "price_history" not in missing:
            missing.append("price_history")
        return ChatResponse(
            answer=answer_data.get("answer", "Analysis completed."),
            intent=intent,
            offers=offers,
            equivalents=equivalents,
            weak_candidates=weak_candidates,
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
        answer_data, scored_candidates = await self._trigger_equivalent_analysis(message, inferred)

        products_found = [
            ProductBrief(
                name=s.get("title", "")[:100],
                brand=s.get("brand"),
            )
            for s in scored_candidates[:5]
            if s.get("title")
        ]

        return ChatResponse(
            answer=answer_data.get("answer", "No equivalent products found."),
            intent=intent,
            products_found=products_found,
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

    async def _trigger_equivalent_analysis(self, message: str, inferred: dict) -> tuple[dict, list]:
        """Run an inline equivalent analysis. Returns (answer_dict, scored_candidates)."""
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
            }, []
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
            }, []
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
            return self._mock_analysis_result(inferred, results, scored), scored
        answer_data = await self._llm_analysis_result(inferred, results, scored, sources)
        return answer_data, scored

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
        current_msg = f"{product_info}\n{results_info}"
        product_context = {
            "detected_category": inferred.get("category", "unknown"),
            "detected_brand": inferred.get("brand"),
            "detected_specs": inferred.get("specs", {}),
            "results_found": n,
            "scored_results": [
                {"title": s.get("title", "")[:80], "price": s.get("price"), "score": s.get("score")}
                for s in top[:5]
            ] if top else [],
        }
        conv_ctx = getattr(self, "_conversation_context", {})
        if not top and conv_ctx.get("equivalents"):
            product_context["previous_equivalents_from_conversation"] = conv_ctx["equivalents"][:10]
        if not top and conv_ctx.get("offers"):
            product_context["previous_offers_from_conversation"] = conv_ctx["offers"]
        context_messages = build_chat_context(
            SYSTEM_ANALYSIS_ANSWER_PROMPT,
            getattr(self, "_conversation_summary", None),
            getattr(self, "_recent_messages", []),
            product_context,
            current_msg,
        )
        try:
            resp = await self.llm.chat_messages(context_messages)
            data = _extract_json(resp["content"])
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

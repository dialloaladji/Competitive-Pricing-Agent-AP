import asyncio
import json
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from api.config import settings


GROQ_PRICES = {
    "llama-3.1-8b-instant": {"input": 0.12, "output": 0.50},
    "default": {"input": 0.12, "output": 0.50},
}

MOCK_PRODUCT_UNDERSTANDING = {
    "name": "ABB S201-C16 MCB 1P 16A",
    "category": "circuit_breaker",
    "brand": "ABB",
    "sku": "S201-C16",
    "product_type": "mcb_1p_c_16a",
    "attributes": [
        "MCB", "1P (single pole)", "16A rated current", "Curve C",
        "6kA breaking capacity", "230V rated voltage", "DIN rail mounting",
        "IEC 60898", "Residential/commercial usage",
    ],
    "target_audience": "electrical installers, panel builders, distributors",
    "price_indicators": {"msrp_eur": 8.50, "trade_price_eur": 6.50, "tier": "budget-mid"},
    "specs": {
        "voltage_v": 230,
        "current_a": 16,
        "poles": 1,
        "curve": "C",
        "breaking_capacity_ka": 6.0,
        "phase": "single",
        "mounting": "din_rail",
        "standard": "IEC 60898",
        "usage": "residential",
    },
}

MOCK_QUERIES = {
    "queries": [
        "MCB 1P 16A curve C 6kA DIN rail",
        "disjoncteur modulaire 1P 16A courbe C 6kA",
        "Schneider Easy9 1P 16A curve C disjoncteur",
        "Legrand RX³ 1P 16A disjoncteur courbe C",
        "Hager MCN116 1P 16A disjoncteur modulaire",
        "Siemens 5SL6106 1P 16A MCB",
    ]
}

MOCK_CANDIDATES = [
    {"title": "Schneider Electric Easy9 1P 16A Courbe C 6kA Disjoncteur modulaire",
     "price": 8.50, "currency": "EUR", "url": "https://example.com/schneider-easy9",
     "merchant": "Rexel", "source": "serpapi", "brand": "Schneider Electric",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "Legrand RX³ 1P 16A Courbe C 6000A Disjoncteur modulaire",
     "price": 7.80, "currency": "EUR", "url": "https://example.com/legrand-rx3",
     "merchant": "Sonepar", "source": "serpapi", "brand": "Legrand",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "Hager MCN116 1P 16A Disjoncteur modulaire 6kA Courbe C",
     "price": 9.20, "currency": "EUR", "url": "https://example.com/hager-mcn",
     "merchant": "Rexel", "source": "tavily", "brand": "Hager",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "Siemens 5SL6106-6 1P 16A MCB Curve C 6kA",
     "price": 12.50, "currency": "EUR", "url": "https://example.com/siemens-5sl",
     "merchant": "Sonepar", "source": "tavily", "brand": "Siemens",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "ABB S201-C16 1P 16A Courbe C 6kA Disjoncteur",
     "price": 7.95, "currency": "EUR", "url": "https://example.com/abb-s201",
     "merchant": "Rexel", "source": "serpapi", "brand": "ABB",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "Chint NXB-63 1P 16A Courbe C 6kA Disjoncteur",
     "price": 4.20, "currency": "EUR", "url": "https://example.com/chint-nxb",
     "merchant": "123elec", "source": "tavily", "brand": "Chint",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "Noark NMB1-63 1P 16A MCB Curve C 6kA",
     "price": 3.80, "currency": "EUR", "url": "https://example.com/noark-nmb",
     "merchant": "Eibmarkt", "source": "tavily", "brand": "Noark",
     "specs": {"voltage_v": 230, "current_a": 16, "poles": 1, "curve": "C", "breaking_capacity_ka": 6.0}},
    {"title": "Bobine de déclenchement MX 12V pour disjoncteur ABB S200",
     "price": 24.99, "currency": "EUR", "url": "https://example.com/bobine-abb",
     "merchant": "123elec", "source": "tavily", "brand": "ABB",
     "specs": {"voltage_v": 12, "current_a": 0, "poles": 0}},
]

MOCK_JUDGMENT = [
    {"candidate_index": 0, "classification": "direct_competitor", "confidence": 0.92,
     "reason": "Schneider Easy9 1P 16A C 6kA is direct cross-brand competitor with identical specs"},
    {"candidate_index": 1, "classification": "direct_competitor", "confidence": 0.90,
     "reason": "Legrand RX3 1P 16A C 6kA - direct cross-brand equivalent"},
    {"candidate_index": 2, "classification": "direct_competitor", "confidence": 0.88,
     "reason": "Hager MCN116 1P 16A C 6kA - direct cross-brand competitor"},
    {"candidate_index": 3, "classification": "premium_alternative", "confidence": 0.82,
     "reason": "Siemens 5SL6106 1P 16A C 6kA - premium cross-brand equivalent at higher price"},
    {"candidate_index": 4, "classification": "same_product", "confidence": 0.95,
     "reason": "Exact ABB S201-C16 same model, same brand"},
    {"candidate_index": 5, "classification": "cheaper_alternative", "confidence": 0.78,
     "reason": "Chint NXB-63 cheaper alternative from Chinese brand, same functional specs"},
    {"candidate_index": 6, "classification": "cheaper_alternative", "confidence": 0.75,
     "reason": "Noark NMB1-63 budget alternative with same specs"},
    {"candidate_index": 7, "classification": "accessory_or_part", "confidence": 0.95,
     "reason": "Bobine de déclenchement is an accessory for ABB S200 series, not a competitor"},
]

MOCK_REFLECTION = {"quality_score": 0.9, "needs_reformulation": False, "issues": [], "confidence": 0.85}
MOCK_QUERIES_REFORMULATED = {
    "previous_issues": ["need more cross-brand candidates"],
    "new_queries": ["MCB 16A Schneider Legrand Hager prix", "disjoncteur modulaire 16A alternatif"],
    "strategy": "broaden to multiple competing brands"
}
MOCK_MARKET_ANALYSIS = {
    "market_overview": "Strong cross-brand competition for MCB 1P 16A 6kA segment with 4 direct equivalents (Schneider, Legrand, Hager, Siemens) plus 2 budget alternatives (Chint, Noark).",
    "competitor_table": [
        {"competitor": "Schneider Easy9", "price": 8.50, "score": 0.92, "merchant": "Rexel"},
        {"competitor": "Legrand RX3", "price": 7.80, "score": 0.90, "merchant": "Sonepar"},
    ],
    "price_analysis": "Cross-brand range: €3.80-€12.50. ABB S201-C16 (€7.95) is positioned mid-range. Budget alternatives 50% cheaper. Premium Siemens 57% above.",
    "recommendation": "ABB S201-C16 at €7.95 is competitive mid-range. Consider €7.50 to undercut Legrand RX3 by 4% while maintaining margin. Below €5 would signal race to bottom with Chint/Noark.",
    "confidence": 0.85,
}


def get_llm_client():
    if settings.mock_mode:
        return MockClient()
    if settings.llm_provider == "llamacpp":
        return GroqClient()
    if settings.llm_provider == "mock":
        return MockClient()
    return GroqClient()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = GROQ_PRICES.get(model, GROQ_PRICES["default"])
    return (input_tokens / 1_000_000 * price["input"]) + (output_tokens / 1_000_000 * price["output"])


class GroqClient:
    def __init__(self):
        self.base_url = settings.llama_cpp_base_url
        self.model = settings.llama_cpp_model
        self.api_key = settings.llama_cpp_api_key
        self.client = httpx.AsyncClient(timeout=60)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def chat(self, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        start = time.time()
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens or settings.llm_max_tokens,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            await asyncio.sleep(retry_after)
        response.raise_for_status()
        data = response.json()
        latency = (time.time() - start) * 1000
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return {
            "content": choice["message"]["content"],
            "latency_ms": latency,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": self.model,
        }

    async def close(self):
        await self.client.aclose()


class MockClient:
    def __init__(self):
        self.model = "mock"
        self._call_count = 0

    async def chat(self, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        self._call_count += 1
        content = self._mock_response(system, user)
        return {
            "content": json.dumps(content),
            "latency_ms": 50,
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "mock",
        }

    def _mock_response(self, system: str, user: str) -> Any:
        if "product understanding" in system.lower() or "extract structured attributes" in system.lower():
            return MOCK_PRODUCT_UNDERSTANDING
        if "generate search queries" in system.lower() or "query strategist" in system.lower():
            return MOCK_QUERIES
        if "normalize" in system.lower():
            return MOCK_CANDIDATES
        if "quality assurance" in system.lower() or "evaluate the quality" in system.lower():
            return MOCK_REFLECTION
        if "product matching" in system.lower() or "valid match" in system.lower():
            return MOCK_JUDGMENT
        if "query reformul" in system.lower() or "improved queries" in system.lower():
            return MOCK_QUERIES_REFORMULATED
        if "market intelligence" in system.lower() or "competitive market analysis" in system.lower():
            return MOCK_MARKET_ANALYSIS
        if "analyze a product" in system.lower():
            return MOCK_PRODUCT_UNDERSTANDING
        return {"result": "mock ok"}

    async def close(self):
        pass


async def search_tavily(query: str, max_results: int = 3) -> list[dict]:
    if settings.mock_mode:
        return [
            {"title": "Schneider Easy9 1P 16A - Disjoncteur modulaire 6kA",
             "url": "https://rexel.fr/schneider-easy9-16a",
             "content": "Schneider Easy9 1P 16A C 6kA MCB - Prix distributeur 8.50 EUR",
             "price": 8.50, "merchant": "Rexel"},
            {"title": "Legrand RX3 1P 16A - Disjoncteur courbe C 6kA",
             "url": "https://sonepar.fr/legrand-rx3-16a",
             "content": "Legrand RX3 1P 16A C 6kA disjoncteur modulaire - Prix 7.80 EUR",
             "price": 7.80, "merchant": "Sonepar"},
            {"title": "Hager MCN116 1P 16A - Disjoncteur modulaire 6kA",
             "url": "https://rexel.fr/hager-mcn-16a",
             "content": "Hager MCN116 1P 16A C 6kA - Prix 9.20 EUR",
             "price": 9.20, "merchant": "Rexel"},
            {"title": "ABB S201-C16 1P 16A - Disjoncteur 6kA",
             "url": "https://rexel.fr/abb-s201-c16",
             "content": "ABB S201-C16 1P 16A C 6kA - Prix 7.95 EUR",
             "price": 7.95, "merchant": "Rexel"},
            {"title": "Chint NXB-63 1P 16A - Disjoncteur 6kA",
             "url": "https://123elec.fr/chint-nxb-63-16a",
             "content": "Chint NXB-63 1P 16A C 6kA - Prix 4.20 EUR",
             "price": 4.20, "merchant": "123elec"},
        ]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": settings.tavily_api_key, "query": query, "search_depth": "advanced",
                  "max_results": max_results, "include_answer": False},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({"title": r["title"], "url": r["url"], "content": r["content"],
                            "price": None, "merchant": None})
        return results


async def search_serpapi(query: str, num: int = 5) -> list[dict]:
    if settings.mock_mode:
        return [
            {"title": "Schneider Easy9 1P 16A Courbe C 6kA Disjoncteur modulaire",
             "price": 8.50, "currency": "EUR", "url": "https://serpapi.com/show?item=1",
             "merchant": "Rexel", "source": "google_shopping", "brand": "Schneider Electric"},
            {"title": "Legrand RX3 1P 16A Courbe C 6000A Disjoncteur",
             "price": 7.80, "currency": "EUR", "url": "https://serpapi.com/show?item=2",
             "merchant": "Sonepar", "source": "google_shopping", "brand": "Legrand"},
            {"title": "Hager MCN116 1P 16A Disjoncteur modulaire 6kA",
             "price": 9.20, "currency": "EUR", "url": "https://serpapi.com/show?item=3",
             "merchant": "Rexel", "source": "google_shopping", "brand": "Hager"},
            {"title": "Siemens 5SL6106-6 1P 16A MCB Curve C 6kA",
             "price": 12.50, "currency": "EUR", "url": "https://serpapi.com/show?item=4",
             "merchant": "Sonepar", "source": "google_shopping", "brand": "Siemens"},
            {"title": "ABB S201-C16 1P 16A Courbe C 6kA Disjoncteur",
             "price": 7.95, "currency": "EUR", "url": "https://serpapi.com/show?item=5",
             "merchant": "Rexel", "source": "google_shopping", "brand": "ABB"},
        ]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={"engine": "google_shopping", "q": query, "api_key": settings.serpapi_api_key,
                    "num": num, "gl": "fr", "hl": "fr"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("shopping_results", []):
            results.append({
                "title": item.get("title", ""), "price": item.get("price", ""),
                "currency": "EUR", "url": item.get("link", ""),
                "merchant": item.get("source", ""), "source": "google_shopping",
            })
        return results

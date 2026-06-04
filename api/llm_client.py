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
    "name": "Wireless Bluetooth Headphones", "category": "Electronics",
    "brand": "SoundPro", "attributes": ["wireless", "bluetooth 5.3", "noise cancelling"],
    "target_audience": "consumers", "price_indicators": {"msrp": 79.99}
}
MOCK_QUERIES = {"queries": ["buy SoundPro headphones", "SoundPro wireless headphones price", "SoundPro vs competitors"]}
MOCK_CANDIDATES = [
    {"title": "SoundPro Wireless Headphones", "price": 69.99, "currency": "USD",
     "url": "https://example.com/1", "merchant": "Amazon", "source": "serpapi"},
    {"title": "SoundPro Bluetooth 5.3 Headset", "price": 74.99, "currency": "USD",
     "url": "https://example.com/2", "merchant": "Walmart", "source": "tavily"},
]
MOCK_JUDGMENT = [{"candidate_index": 0, "is_match": True, "confidence": 0.92, "reason": "Exact product match"},
                 {"candidate_index": 1, "is_match": True, "confidence": 0.85, "reason": "Same brand, comparable specs"}]
MOCK_REFLECTION = {"quality_score": 0.8, "needs_reformulation": False, "issues": [], "confidence": 0.85}
MOCK_QUERIES_REFORMULATED = {"previous_issues": ["queries too narrow"], "new_queries": ["wireless headphones deals", "noise cancelling headphones price comparison"], "strategy": "broaden search"}
MOCK_MARKET_ANALYSIS = {
    "market_overview": "3 competitors found for SoundPro Wireless Headphones.",
    "competitor_table": [{"competitor": "Amazon - SoundPro", "price": 69.99, "score": 0.92}],
    "price_analysis": "Best price found at $69.99, 12.5% below target of $79.99",
    "recommendation": "Price at $74.99 to stay competitive while maintaining margin.",
    "confidence": 0.85
}


def get_llm_client():
    if settings.llm_provider == "llamacpp":
        return GroqClient()
    elif settings.llm_provider == "mock" or settings.mock_mode:
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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
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
        if "product matching" in system.lower() or "valid match" in system.lower():
            return MOCK_JUDGMENT
        if "quality assurance" in system.lower() or "evaluate the quality" in system.lower():
            return MOCK_REFLECTION
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
            {"title": "SoundPro - Official Store", "url": "https://soundpro.com/product",
             "content": "SoundPro Wireless Headphones $69.99", "price": 69.99, "merchant": "SoundPro"},
            {"title": "Amazon - SoundPro Headphones", "url": "https://amazon.com/dp/123",
             "content": "SoundPro Bluetooth Headphones $74.99", "price": 74.99, "merchant": "Amazon"},
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
            {"title": "SoundPro Wireless Bluetooth Headphones", "price": 69.99, "currency": "USD",
             "url": "https://serpapi.com/show?item=1", "merchant": "Amazon", "source": "google_shopping"},
            {"title": "SoundPro Noise Cancelling Headphones", "price": 79.99, "currency": "USD",
             "url": "https://serpapi.com/show?item=2", "merchant": "Best Buy", "source": "google_shopping"},
        ]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={"engine": "google_shopping", "q": query, "api_key": settings.serpapi_api_key,
                    "num": num, "gl": "us", "hl": "en"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("shopping_results", []):
            results.append({
                "title": item.get("title", ""), "price": item.get("price", ""),
                "currency": "USD", "url": item.get("link", ""),
                "merchant": item.get("source", ""), "source": "google_shopping",
            })
        return results

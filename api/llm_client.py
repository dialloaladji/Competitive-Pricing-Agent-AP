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

MOCK_SERPAPI_RESULTS = [
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


def get_llm_client():
    if settings.mock_mode or settings.llm_provider == "mock":
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
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
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

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def chat_messages(self, messages: list[dict], max_tokens: int | None = None) -> dict[str, Any]:
        start = time.time()
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or settings.llm_max_tokens,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            await asyncio.sleep(retry_after)
            response.raise_for_status()  # raises → tenacity retries
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

    async def stream_chat_messages(self, messages: list[dict], max_tokens: int | None = None):
        """Yield raw text tokens from a streaming chat completion."""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or settings.llm_max_tokens,
                "temperature": 0.1,
                "stream": True,
                "response_format": {"type": "json_object"},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    content = data["choices"][0]["delta"].get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def close(self):
        await self.client.aclose()


class MockClient:
    def __init__(self):
        self.model = "mock"

    async def chat(self, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        return {
            "content": json.dumps({
                "summary": "Mock summary: 5 cross-brand equivalents found between €4.20 and €12.50.",
                "price_range_min": 4.20,
                "price_range_max": 12.50,
                "confidence": 0.85,
            }),
            "latency_ms": 50,
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "mock",
        }

    async def chat_messages(self, messages: list[dict], max_tokens: int | None = None) -> dict[str, Any]:
        return await self.chat("", "")

    async def stream_chat_messages(self, messages: list[dict], max_tokens: int | None = None):
        mock_result = await self.chat("", "")
        yield mock_result["content"]

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
        return MOCK_SERPAPI_RESULTS
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

# Competitive Pricing Agent — Electrical Products

An AI-powered pricing intelligence tool for electrical products. Ask a question in natural language, get a grounded market analysis with competitor prices, scored candidates, and expert recommendations.

---

## What it does

Given any electrical product description, the agent:

1. Searches Google Shopping in real time (SerpAPI)
2. Scores and ranks up to 40 candidates across competing brands
3. Returns four picks: cheapest, best score, best technical match, best overall compromise
4. Explains why confidence is high or low (missing specs, vague listings, etc.)
5. Remembers the conversation — follow-up questions reuse the stored analysis

**Example conversation:**
```
You   → "Find me the best price for an ABB S201-C16 1P 16A curve C 6kA"
Agent → Analysis: 40 candidates, 6 valid matches. Best price: €2.77 (ABB), best score: €11.15 (score 0.68)

You   → "Why is the confidence low?"
Agent → Missing specs: voltage_v. Best match score 0.68 is below the 0.88 threshold for a confirmed exact match.

You   → "Give me the top 3 technical matches"
Agent → [lists 3 candidates with specs breakdown]
```

---

## Key capabilities

| Capability | Detail |
|---|---|
| Live market search | Google Shopping via SerpAPI, 40 results per query |
| Scoring | 5-component weighted score (specs, price, brand, merchant trust) |
| Candidate picks | Cheapest / Best score / Best technical / Best analyst compromise |
| Confidence signal | Flags low confidence when best score < 0.88 or specs are missing |
| Brand diversity | Max 2 results per brand, min 3 distinct brands |
| Electrical-only | Rejects non-electrical queries before any API call |
| Conversational memory | Follow-up questions reuse stored analysis without re-searching |

**Supported brands:** ABB, Schneider, Legrand, Siemens, Eaton, Hager, Chint, Hager, Bticino, and 30+ others.

---

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/chat` | Conversational interface — ask anything in natural language |
| `POST` | `/api/v1/products/analyze-equivalents` | Direct analysis — returns raw scored candidates |
| `GET` | `/api/v1/analysis/{run_id}` | Retrieve a past analysis by ID |
| `GET` | `/api/v1/products/{id}/analysis/latest` | Latest analysis for a tracked product |
| `GET` | `/health` | Health check |

The chat endpoint is the primary interface. The analyze-equivalents endpoint is for direct API integrations.

---

## Quick start

**Prerequisites:** Python 3.12+, Docker, a SerpAPI key, a Groq API key.

```bash
# 1. Start Postgres
docker compose up -d postgres

# 2. Configure environment
cp .env.example .env   # add SERPAPI_API_KEY and LLAMA_CPP_API_KEY

# 3. Install and migrate
pip install -r requirements.txt
alembic upgrade head

# 4. Run
uvicorn api.main:app --reload --port 8001
```

Swagger UI: **http://localhost:8001/docs**

**No API keys?** Run in mock mode (returns synthetic data, no external calls):
```bash
MOCK_MODE=true uvicorn api.main:app --reload --port 8001
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL async URL |
| `SERPAPI_API_KEY` | Yes | Google Shopping search |
| `LLAMA_CPP_API_KEY` | Yes | Groq API key (LLM answers) |
| `MOCK_MODE` | No | `true` to run without API keys |
| `LLAMA_CPP_BASE_URL` | No | LLM endpoint (default: Groq) |
| `LLAMA_CPP_MODEL` | No | Model (default: llama-3.1-8b-instant) |

# Electrical Products Competitive Pricing & Market Intelligence

Real-time competitive pricing analysis for **electrical products only** — fast, deterministic, no LLM flakiness.

## Who is it for?

- **Marketing teams** benchmarking their electrical product catalog against competitors
- **Pricing managers** looking for current market price points on MCBs, RCDs, contactors, switches, cables, etc.
- **Category managers** comparing cross-brand offerings (ABB vs Schneider, Legrand, Siemens, Eaton, Hager, etc.)

## Core use case

Given ONE electrical product description, find **5 cross-brand equivalents** (max 2 per brand, min 3 distinct brands) with deterministic spec matching, scored by spec quality, and priced in EUR.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│            FastAPI Backend (sync endpoint)            │
│  POST /api/v1/products/analyze-equivalents            │
│                                                        │
│  1. Domain gate (electrical only, else HTTP 400)      │
│  2. Deterministic spec inference (regex-based)         │
│  3. SerpApi Google Shopping search                     │
│  4. Deterministic normalization (brand + specs regex)  │
│  5. Deterministic scoring (5-component weighted score) │
│  6. Brand diversification (max 2/brand, min 3 brands)  │
│  7. Optional LLM summarizer (3-5 sentence summary)     │
│                                                        │
│  Latency: ~39ms (mock) / ~5-7s (real SerpApi)         │
└──────────────────────────────────────────────────────┘
```

### Services

| Service | Tech | Port |
|---|---|---|
| `api` | FastAPI + Uvicorn | 8001 |
| `postgres` | PostgreSQL 16 | 5432 |

No Celery/Redis needed — the primary endpoint is **synchronous** and finishes in seconds.

## Pipeline (6 deterministic steps + optional LLM)

1. **Domain gate** — Rejects non-electrical products in <1ms (`_is_electrical`)
2. **Spec inference** — Extracts brand, category, current, poles, curve, kA, voltage, mounting, sensitivity, differential type from description using regex
3. **SerpApi search** — Sends description as query to Google Shopping; Tavily as fallback
4. **Normalization** — URL dedup, price parsing, brand extraction, spec extraction from each candidate title
5. **Scoring** — 5-component weighted score:

   | Component | Weight |
   |---|---|
   | Deterministic pre-score (brand match, similarity) | 0.25 |
   | Spec quality (current_a, poles, curve, kA, voltage, sensitivity, differential type) | 0.35 |
   | Price score (closer to target = better) | 0.15 |
   | Merchant trust (Rexel/Sonepar=0.9, Amazon/Leroy Merlin=0.85, Ebay=0.6) | 0.05 |
   | Tier-1 brand boost (ABB, Schneider, Legrand, Siemens, Eaton, Hager +0.10) | 0.10 |
   | Base score | 0.10 |

6. **Brand diversification** — Greedy score-based selection, max 2 per brand, max 5 total, min 3 brands
7. **Optional LLM summary** — Groq/Llama generates 3-5 sentence competitive landscape summary (only if ≥3 reliable candidates)

## Quick Start (Local)

### Prerequisites
- Python 3.12+
- Docker Desktop (for Postgres)

### 1. Start services
```bash
docker compose up -d postgres
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys (or set MOCK_MODE=true for mock data)
```

### 3. Install dependencies & run migrations
```bash
pip install -r requirements.txt
alembic upgrade head
```

### 4. Run the API
```bash
uvicorn api.main:app --reload --port 8001
```

Server starts on **http://localhost:8001/docs** (Swagger UI).

### Mock mode (no API keys needed)
```bash
MOCK_MODE=true uvicorn api.main:app --reload --port 8001
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/metrics-summary` | Aggregated metrics |
| `GET` | `/api/v1/products` | List products |
| `POST` | `/api/v1/products` | Create product |
| `GET` | `/api/v1/products/{id}` | Get product |
| `PUT` | `/api/v1/products/{id}` | Update product |
| `DELETE` | `/api/v1/products/{id}` | Delete product |
| **`POST`** | **`/api/v1/products/analyze-equivalents`** | **Main endpoint — synchronous analysis** |
| `POST` | `/api/v1/products/{id}/analyze` | Trigger Celery async analysis (legacy) |
| `GET` | `/api/v1/products/{id}/offers` | Competitor offers |
| `GET` | `/api/v1/products/{id}/price-history` | Price snapshots |
| `GET` | `/api/v1/dashboard/summary` | Dashboard aggregation |

## Request format

```json
{
  "description": "Disjoncteur Legrand 16A 6kA courbe C Rail DIN",
  "currency": "EUR"
}
```

Optional overrides (set to `"string"` or 0 to force re-inference):
```json
{
  "description": "Interrupteur différentiel 2P 40A 30mA type AC",
  "brand": "string",
  "category": "string",
  "current_a": 40,
  "poles": 2,
  "sensitivity_ma": 30,
  "differential_type": "AC"
}
```

## Response format

```json
{
  "product_id": "uuid",
  "product_name": "Disjoncteur Legrand 16A 6kA courbe C Rail DIN",
  "candidate_count": 40,
  "valid_match_count": 15,
  "cross_brand_count": 5,
  "weak_candidate_count": 0,
  "best_match_price": 8.50,
  "cross_brand_equivalents": [
    {
      "title": "Schneider Easy9 1P 16A C 6kA",
      "brand": "Schneider Electric",
      "price": 8.50,
      "score": 0.82,
      "spec_quality": 0.85,
      "spec_match": "exact_spec_equivalent",
      "specs": {
        "current_a": 16, "poles": 1, "curve": "C",
        "breaking_capacity_ka": 6.0, "mounting": "DIN rail"
      }
    }
  ],
  "partial_spec_equivalents": [
    {
      "title": "Legrand 412020 2P 40A 30mA type A",
      "brand": "Legrand",
      "price": 52.00,
      "score": 0.55,
      "spec_quality": 0.35,
      "spec_match": "functional_equivalent",
      "specs": {
        "current_a": 40, "poles": 2, "sensitivity_ma": 30,
        "differential_type": "A", "mounting": "DIN rail"
      }
    }
  ],
  "same_brand_listings": [],
  "weak_candidates": [],
  "inferred_product": {
    "name": "Disjoncteur Legrand 16A 6kA courbe C Rail DIN",
    "category": "miniature circuit breaker",
    "brand": "Legrand",
    "specs": {
      "current_a": 16, "curve": "C",
      "breaking_capacity_ka": 6.0, "mounting": "DIN rail"
    }
  },
  "recommendation": "Competitive landscape summary...",
  "price_confidence": 0.85
}
```

## Domain validation

The API is **restricted to electrical products only**. Non-electrical descriptions are rejected with **HTTP 400** before any API call.

**Accepted:** circuit breakers, RCDs, contactors, switches, cables, electrical panels, EV chargers, transformers, relays, fuses, sockets

**Rejected:** headphones, food, clothing, furniture, toys, automotive parts

## Supported electrical brands

**Tier 1 (boosted +0.10):** ABB, Schneider, Legrand, Siemens, Eaton, Hager

**Tier 2:** Chint, Noark, Phoenix Contact, Wago, Finder, Lovato, Mitsubishi, Bticino, Gewiss

**Tier 3 (regional/heritage):** Crouzet, Klockner Moeller, Telemecanique, Merlin Gerin, Square D, Carlo Gavazzi, HENSEL, Spelsberg, Rittal, Weidmuller

**Switches & wiring:** Gira, Jung, Berker, Feller, Merten, Siedle, Bals, Walther

**Cable & EV:** Nexans, Prysmian, Lapp, Helukabel, Wallbox

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | PostgreSQL async URL |
| `LLAMA_CPP_API_KEY` | — | — | Groq API key (for summarizer) |
| `LLAMA_CPP_BASE_URL` | — | `https://api.groq.com/openai/v1` | LLM endpoint |
| `LLAMA_CPP_MODEL` | — | `llama-3.1-8b-instant` | LLM model |
| `SERPAPI_API_KEY` | ✅ | — | SerpApi Google Shopping |
| `TAVILY_API_KEY` | — | — | Tavily fallback web search |
| `MOCK_MODE` | — | `false` | Run without API keys |
| `LLM_PROVIDER` | — | `groq` | LLM backend |
| `LLM_MAX_TOKENS` | — | `512` | Max LLM response tokens |

## Mock Mode

Set `MOCK_MODE=true` to run the full pipeline without any API keys. Returns 5 mock SerpApi results (Schneider €8.50, Legrand €7.80, Hager €9.20, Siemens €12.50, ABB €7.95) with mock LLM summary. Latency: ~39ms.

## Test coverage

106 unit tests covering:
- Domain detection (electrical vs non-electrical)
- Spec inference (current_a, poles, curve, kA, voltage, mounting, sensitivity_ma, differential_type, brand)
- RCD-specific extraction (30mA, type AC/A/F, 2P from "2x40A")
- Spec quality scoring with all weights and penalties
- Differential type matching penalty (-0.20 for mismatch)
- Brand diversification (max 2/brand, min 3 brands)
- Cross-brand bonus, same-brand detection
- Vague title detection
- Merchant trust scoring
- Output capping (max 5, max 2 per brand)

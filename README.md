# AI Electrical Products Competitive Pricing & Market Intelligence

Real-time competitive pricing analysis for **electrical products only**, powered by LLM agents (Groq/Llama), web search (Tavily/SerpApi), and market intelligence — deployed on Railway.

## Who is it for?

- **Marketing teams** benchmarking their electrical product catalog against competitors
- **Pricing managers** looking for current market price points on MCBs, contactors, switches, cables, etc.
- **Category managers** comparing cross-brand offerings (ABB vs Schneider, Legrand, Siemens, Eaton, Hager, etc.)
- **Marketing directors** who need a quick competitive landscape view

## Core use case

Given ONE electrical product, find 3-5 equivalent or comparable products from **other brands** and compare prices.

The API focuses on electrical product specifications: product type, brand, reference, voltage, current, poles, curve, breaking capacity, phase, power rating, mounting, standard, residential/commercial/industrial usage.

## Architecture

```
┌────────────────────────────────────────┐
│            FastAPI Backend             │
│  POST /api/v1/products/analyze-equiv  │
│  (synchronous — 10 AI agents in-line) │
└──────┬───────────────────────────┬─────┘
       │                           │
       ▼                           ▼
 ┌──────────┐               ┌──────────┐
 │PostgreSQL│               │  Redis   │
 └──────────┘               └──────────┘
       ▲                           ▲
       │                           │
 ┌─────┴─────────────┐   ┌────────┴────────┐
 │  Celery Worker    │   │  Celery Beat    │
 │  (async analysis) │   │  (scheduler)    │
 └───────────────────┘   └─────────────────┘
```

### Services (Railway)
| Service | Tech | Port |
|---|---|---|
| `api` | FastAPI + Uvicorn | 8000 |
| `worker` | Celery | — |
| `scheduler` | Celery Beat | — |
| `postgres` | PostgreSQL 16 | 5432 |
| `redis` | Redis 7 | 6379 |

### AI Agents (10-agent workflow)
1. **product_understanding_agent** — Extract electrical specs (voltage, current, poles, curve, kA, etc.)
2. **query_generator_agent** — Generate search queries for electrical brand discovery
3. **serpapi_search** — Search Google Shopping via SerpApi
4. **tavily_search** — Web search via Tavily
5. **candidate_normalizer** — Deduplicate, normalize, extract electrical specs per candidate
6. **llm_judge** — Validate electrical domain + classify as direct_competitor, premium, cheaper, etc.
7. **scoring_engine** — Score and rank valid matches (deterministic + LLM)
8. **reflection_agent** — Evaluate quality, trigger reformulation if needed
9. **query_reformulator** — Improve search queries for better results
10. **market_analyst_agent** — Produce final market analysis report

## Quick Start (Local)

### Prerequisites
- Python 3.12+
- Docker Desktop (for Postgres + Redis)

### 1. Start services
```bash
docker compose up -d postgres redis
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys (or leave defaults for mock mode)
```

### 3. Install dependencies & seed data
```bash
pip install -r requirements.txt
python scripts/seed.py
```

### 4. Run the stack
```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Worker
celery -A worker.celery_app worker --loglevel=info

# Terminal 3 — Scheduler
celery -A worker.celery_app beat --loglevel=info
```

> **Note:** The default port 8000 is often occupied by Docker Desktop. Use `--port 8001` if needed.

### Or run everything with Docker Compose
```bash
docker compose up --build
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
| `POST` | `/api/v1/products/analyze-equivalents` | Synchronous AI analysis (electrical products only) |
| `POST` | `/api/v1/products/{id}/analyze` | Trigger Celery async analysis |
| `GET` | `/api/v1/products/{id}/analysis` | Analysis history |
| `GET` | `/api/v1/products/{id}/analysis/latest` | Latest analysis |
| `GET` | `/api/v1/products/{id}/offers` | Competitor offers |
| `GET` | `/api/v1/products/{id}/price-history` | Price snapshots |
| `GET` | `/api/v1/dashboard/summary` | Dashboard aggregation |

## Request format (electrical product)

```json
{
  "name": "ABB S201-C16",
  "description": "Disjoncteur modulaire MCB 1P 16A courbe C 6kA",
  "sku": "S201-C16",
  "product_type": "mcb_1p_c_16a",
  "brand": "ABB",
  "voltage_v": 230,
  "current_a": 16,
  "poles": 1,
  "curve": "C",
  "breaking_capacity_ka": 6.0,
  "phase": "single",
  "mounting": "din_rail",
  "standard": "IEC 60898",
  "usage": "residential",
  "target_price": 8.50,
  "currency": "EUR"
}
```

## Response format

The response separates **cross-brand competitors** (the main use case) from **same-brand listings** (for price comparison only).

```json
{
  "product_id": "...",
  "product_name": "ABB S201-C16",
  "cross_brand_count": 4,
  "same_brand_count": 1,
  "cross_brand_equivalents": [
    {
      "title": "Schneider Easy9 1P 16A Courbe C 6kA",
      "brand": "Schneider Electric",
      "price": 8.50,
      "currency": "EUR",
      "score": 0.85,
      "classification": "direct_competitor",
      "specs": {
        "voltage_v": 230,
        "current_a": 16,
        "poles": 1,
        "curve": "C",
        "breaking_capacity_ka": 6.0
      }
    },
    {
      "title": "Legrand RX3 1P 16A Courbe C 6000A",
      "brand": "Legrand",
      "price": 7.80,
      "classification": "direct_competitor"
    }
  ],
  "same_brand_listings": [
    {
      "title": "ABB S201-C16 1P 16A (Rexel)",
      "price": 7.95,
      "classification": "same_product"
    }
  ]
}
```

## Domain validation

The API is **restricted to electrical products only**. Requests that don't contain an electrical product identifier (brand from the `ELECTRICAL_BRANDS` list or electrical keyword from `ELECTRICAL_KEYWORDS`) are rejected with **HTTP 400** before any API call is made.

Examples of **accepted** products: circuit breakers, contactors, switches, sockets, cables, electrical panels, EV chargers, transformers, fuses, relays, contactors.

Examples of **rejected** products: headphones, food, clothing, generic household items — any non-electrical product returns HTTP 400 immediately.

## Supported electrical brands

**Tier 1 (priority):** ABB, Schneider Electric, Legrand, Siemens, Eaton, Hager

**Tier 2 (mid-range):** Chint, Noark, Phoenix Contact, Wago, Finder, Lovato, Mitsubishi, Bticino, Gewiss

**Tier 3 (regional/heritage):** Crouzet, Klockner Moeller, Telemecanique, Merlin Gerin, Square D, ABB Stotz, Siemens Sentron, Carlo Gavazzi, HENSEL, Spelsberg, Rittal, Weidmuller

**Switches & wiring accessories:** Gira, Jung, Berker, Feller, Merten, Siedle, Bals, Walther

**Cable & EV:** Nexans, Prysmian, Lapp, Helukabel, Wallbox

## Deploy to Railway

### 1. Create Railway project
```bash
railway login
railway init
```

### 2. Add plugins
- **PostgreSQL** — Railway will inject `DATABASE_URL`
- **Redis** — Railway will inject `REDIS_URL`

### 3. Set environment variables
```bash
railway env set LLAMA_CPP_API_KEY=gsk_your_key
railway env set TAVILY_API_KEY=tvly_your_key
railway env set SERPAPI_API_KEY=your_key
railway env set LANGFUSE_PUBLIC_KEY=pk-lf-your-key
railway env set LANGFUSE_SECRET_KEY=sk-lf-your-key
```

### 4. Create services
In Railway dashboard, create 3 services all pointing to the same repo:
- **api** — `railway run` with target `api`
- **worker** — `railway run` with target `worker`
- **scheduler** — `railway run` with target `scheduler`

### 5. Run migrations
```bash
railway run alembic upgrade head
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | PostgreSQL async URL |
| `REDIS_URL` | ✅ | — | Redis URL |
| `LLAMA_CPP_API_KEY` | ✅ | — | Groq API key |
| `LLAMA_CPP_BASE_URL` | — | `https://api.groq.com/openai/v1` | LLM endpoint |
| `LLAMA_CPP_MODEL` | — | `llama-3.1-8b-instant` | LLM model |
| `TAVILY_API_KEY` | ✅ | — | Tavily web search |
| `SERPAPI_API_KEY` | ✅ | — | SerpApi Google Shopping |
| `LANGFUSE_PUBLIC_KEY` | — | — | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | — | — | Langfuse observability |
| `MOCK_MODE` | — | `false` | Run without API keys (overrides `LLM_PROVIDER`) |
| `LLM_PROVIDER` | — | `groq` | LLM backend (`groq`, `llamacpp`, `mock`) |
| `LLM_MAX_TOKENS` | — | `512` | Max LLM response tokens |

## Observability (Langfuse)

Each product analysis creates a Langfuse trace with spans for all 10 agents. Metrics tracked:
- Latency per agent and end-to-end
- Token usage and estimated cost
- JSON parse success rate
- Match success rate and reflection reformulation rate
- Price confidence scores

## Mock Mode

Set `MOCK_MODE=true` to run the full stack without any API keys. All LLM calls and external APIs return realistic electrical-product mock data (ABB, Schneider, Legrand, Hager, Siemens, Chint, Noark). Mock mode is checked **before** `LLM_PROVIDER`, so setting `MOCK_MODE=true` always uses mock clients regardless of the provider setting.

## Testing with Swagger UI

Start the API and open **http://localhost:8001/docs** (or your configured port) for interactive Swagger UI documentation. The synchronous endpoint `POST /api/v1/products/analyze-equivalents` is the primary testing entrypoint:

```json
{
  "name": "ABB S201-C16",
  "description": "Disjoncteur modulaire MCB 1P 16A courbe C 6kA",
  "sku": "S201-C16",
  "product_type": "mcb_1p_c_16a",
  "brand": "ABB",
  "voltage_v": 230,
  "current_a": 16,
  "poles": 1,
  "curve": "C",
  "breaking_capacity_ka": 6.0,
  "phase": "single",
  "mounting": "din_rail",
  "standard": "IEC 60898",
  "usage": "residential",
  "target_price": 8.50,
  "currency": "EUR"
}
```

> **Note:** Mock mode returns results instantly (~2s). Real mode requires valid API keys and may take 30–60s depending on rate limits.

## Test coverage

47 unit tests covering:
- Electrical domain detection (12 tests)
- Deterministic pre-scoring with electrical products (15 tests)
- Category similarity for electrical categories
- Cross-brand bonus with electrical brands
- Accessory detection for electrical accessories (bobine, bornier, plaque, etc.)
- Electrical keywords/brands list validation
- Fixtures (7 real electrical products: ABB, Schneider, Legrand, Hager, Siemens, EV charger, cable)

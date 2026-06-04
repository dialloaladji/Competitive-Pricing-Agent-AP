# AI Competitive Pricing & Market Intelligence Agent

Real-time competitive pricing analysis powered by LLM agents (Groq/Llama), web search (Tavily/SerpApi), and market intelligence — deployed on Railway.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│   Celery     │
│  Frontend   │     │   Backend    │     │   Worker     │
└─────────────┘     └──────┬───────┘     └──────┬───────┘
                           │                     │
                           ▼                     ▼
                     ┌──────────┐          ┌──────────┐
                     │PostgreSQL│          │  Redis   │
                     └──────────┘          └──────────┘
```

### Services (Railway)
| Service | Tech | Port |
|---|---|---|
| `api` | FastAPI + Uvicorn | 8000 |
| `frontend` | Streamlit | 8501 |
| `worker` | Celery | — |
| `scheduler` | Celery Beat | — |
| `postgres` | PostgreSQL 16 | 5432 |
| `redis` | Redis 7 | 6379 |

### AI Agents (10-agent workflow)
1. **product_understanding_agent** — Extract structured attributes from product listing
2. **query_generator_agent** — Generate search queries for competitor discovery
3. **serpapi_search** — Search Google Shopping via SerpApi
4. **tavily_search** — Web search via Tavily
5. **candidate_normalizer** — Deduplicate and normalize candidate offers
6. **llm_judge** — Validate whether candidates are true competitor matches
7. **scoring_engine** — Score and rank valid matches (deterministic)
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

# Terminal 4 — Frontend
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501** for the dashboard.

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
| `POST` | `/api/v1/products/{id}/analyze` | Trigger AI analysis |
| `GET` | `/api/v1/products/{id}/analysis` | Analysis history |
| `GET` | `/api/v1/products/{id}/analysis/latest` | Latest analysis |
| `GET` | `/api/v1/products/{id}/offers` | Competitor offers |
| `GET` | `/api/v1/products/{id}/price-history` | Price snapshots |
| `GET` | `/api/v1/dashboard/summary` | Dashboard aggregation |

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
In Railway dashboard, create 4 services all pointing to the same repo:
- **api** — `railway run` with target `api`
- **frontend** — `railway run` with target `frontend`
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
| `MOCK_MODE` | — | `false` | Run without API keys |
| `LLM_MAX_TOKENS` | — | `512` | Max LLM response tokens |

## Observability (Langfuse)

Each product analysis creates a Langfuse trace with spans for all 10 agents. Metrics tracked:
- Latency per agent and end-to-end
- Token usage and estimated cost
- JSON parse success rate
- Match success rate and reflection reformulation rate
- Price confidence scores

## Mock Mode

Set `MOCK_MODE=true` to run the full stack without any API keys. All LLM calls and external APIs return realistic mock data.

FROM python:3.12-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

FROM base AS api
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT

FROM base AS worker
CMD celery -A worker.celery_app worker --loglevel=info --concurrency=4

FROM base AS scheduler
CMD celery -A worker.celery_app beat --loglevel=info

FROM base AS frontend
CMD streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

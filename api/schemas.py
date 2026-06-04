import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    name: str
    description: str = ""
    category: str | None = None
    brand: str | None = None
    sku: str | None = None
    image_url: str | None = None
    target_price: float | None = None
    currency: str = "USD"
    is_tracked: bool = True


class ProductUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    brand: str | None = None
    sku: str | None = None
    image_url: str | None = None
    target_price: float | None = None
    currency: str | None = None
    is_tracked: bool | None = None


class ProductOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    category: str | None
    brand: str | None
    sku: str | None
    image_url: str | None
    target_price: float | None
    currency: str
    is_tracked: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OfferOut(BaseModel):
    id: uuid.UUID
    product_id: uuid.UUID
    source: str
    competitor_name: str
    title: str
    price: float
    currency: str
    url: str
    merchant: str | None
    condition: str | None
    in_stock: bool
    discovered_at: datetime

    class Config:
        from_attributes = True


class AnalysisRunOut(BaseModel):
    id: uuid.UUID
    product_id: uuid.UUID
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    total_latency_ms: float | None
    candidate_count: int | None
    valid_match_count: int | None
    best_match_price: float | None
    best_match_score: float | None
    price_confidence: float | None
    final_decision: dict | None
    error_message: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class PriceSnapshotOut(BaseModel):
    id: uuid.UUID
    product_id: uuid.UUID
    price: float
    currency: str
    snapshot_date: datetime

    class Config:
        from_attributes = True


class MetricsSummary(BaseModel):
    total_products: int
    tracked_products: int
    total_offers: int
    total_analyses: int
    total_snapshots: int
    analyses_today: int
    avg_latency_ms: float | None
    match_rate: float | None
    no_match_rate: float | None
    avg_confidence: float | None


class AnalyzeResponse(BaseModel):
    run_id: str
    product_id: str
    status: str
    message: str


class DashboardSummary(BaseModel):
    total_products: int
    tracked_products: int
    total_offers: int
    total_analyses: int
    avg_confidence: float | None
    best_price_drops: list[dict]
    recent_analyses: list[dict]

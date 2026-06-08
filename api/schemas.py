import uuid
from datetime import datetime
from pydantic import BaseModel, Field


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


class EquivalentRequest(BaseModel):
    description: str
    brand: str | None = None
    sku: str | None = None
    target_price: float | None = None
    currency: str = "EUR"
    max_iterations: int = 1
    name: str | None = None
    category: str | None = None
    voltage_v: int | None = None
    current_a: int | None = None
    poles: int | None = None
    curve: str | None = None
    breaking_capacity_ka: float | None = None
    phase: str | None = None


VALID_CLASSIFICATIONS = frozenset({
    "same_product", "direct_competitor", "functional_equivalent",
    "cheaper_alternative", "premium_alternative",
    "previous_generation", "newer_generation",
})


class EquivalentOut(BaseModel):
    title: str
    price: float
    currency: str
    merchant: str | None = None
    brand: str | None = None
    url: str
    score: float
    price_score: float
    relevance_score: float
    trust_score: float
    spec_quality: float = 0.0
    is_vague: bool = False
    classification: str = "functional_equivalent"
    spec_match: str = "functional_equivalent"
    specs: dict = Field(default_factory=dict)


class AnalyzeEquivalentsResponse(BaseModel):
    product_id: str
    product_name: str
    run_id: str
    total_latency_ms: float
    candidate_count: int
    valid_match_count: int
    cross_brand_count: int
    same_brand_count: int
    weak_candidate_count: int
    best_match_price: float | None
    best_match_score: float | None
    price_confidence: float | None
    recommendation: str | None
    cross_brand_equivalents: list[EquivalentOut]
    same_brand_listings: list[EquivalentOut]
    weak_candidates: list[EquivalentOut]
    partial_spec_equivalents: list[EquivalentOut] = []
    partial_spec_count: int = 0
    brand_diversity_warning: str | None = None
    brand_diversity_stats: dict | None = None
    inferred_product: dict | None = None

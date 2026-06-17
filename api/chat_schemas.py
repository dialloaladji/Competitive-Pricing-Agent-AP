from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    product_id: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None


class ProductBrief(BaseModel):
    id: str | None = None
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    reference: str | None = None


class PriceAnalysis(BaseModel):
    has_history: bool = False
    min_price: float | None = None
    median_price: float | None = None
    max_price: float | None = None
    currency: str | None = None
    trend: str = "unknown"
    summary: str | None = None


class MarketAnalysis(BaseModel):
    observed_facts: list[str] = []
    hypotheses: list[str] = []
    risks: list[str] = []
    recommendations: list[str] = []


class ChatResponse(BaseModel):
    answer: str
    intent: str = "general_question"
    product: ProductBrief = Field(default_factory=ProductBrief)
    products_found: list[ProductBrief] = []
    # Confirmed cross-brand equivalents (score >= threshold, not vague)
    equivalents: list[dict] = []
    # Partial-spec candidates (below confirmation threshold but not vague)
    partial_candidates: list[dict] = []
    weak_candidates: list[dict] = []
    offers: list[dict] = []
    # How many of `equivalents` are confirmed matches (valid_match_count)
    confirmed_equivalents_count: int = 0
    price_analysis: PriceAnalysis = Field(default_factory=PriceAnalysis)
    market_analysis: MarketAnalysis = Field(default_factory=MarketAnalysis)
    confidence: str = "low"
    sources_used: list[str] = []
    actions_triggered: list[str] = []
    missing_information: list[str] = []
    conversation_id: str | None = None
    message_id: str | None = None
    product_id: str | None = None

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    product_id: str | None = None
    conversation_id: str | None = None


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
    equivalents: list[dict] = []
    offers: list[dict] = []
    price_analysis: PriceAnalysis = Field(default_factory=PriceAnalysis)
    market_analysis: MarketAnalysis = Field(default_factory=MarketAnalysis)
    confidence: str = "low"
    sources_used: list[str] = []
    actions_triggered: list[str] = []
    missing_information: list[str] = []

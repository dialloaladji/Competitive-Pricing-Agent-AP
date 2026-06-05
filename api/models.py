import uuid
from datetime import datetime

from sqlalchemy import String, Float, Boolean, DateTime, Text, JSON, Integer, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from api.database import Base


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Product(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sku: Mapped[str | None] = mapped_column(String(100), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="EUR")
    is_tracked: Mapped[bool] = mapped_column(Boolean, default=True)
    product_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    voltage_v: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    poles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    curve: Mapped[str | None] = mapped_column(String(10), nullable=True)
    breaking_capacity_ka: Mapped[float | None] = mapped_column(Float, nullable=True)
    phase: Mapped[str | None] = mapped_column(String(20), nullable=True)
    power_w: Mapped[float | None] = mapped_column(Float, nullable=True)
    mounting: Mapped[str | None] = mapped_column(String(50), nullable=True)
    standard: Mapped[str | None] = mapped_column(String(50), nullable=True)
    usage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    offers = relationship("Offer", back_populates="product", cascade="all, delete-orphan")
    analysis_runs = relationship("AnalysisRun", back_populates="product", cascade="all, delete-orphan")


class Offer(Base):
    __tablename__ = "offers"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    competitor_name: Mapped[str] = mapped_column(String(255), default="")
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    merchant: Mapped[str | None] = mapped_column(String(255), nullable=True)
    condition: Mapped[str | None] = mapped_column(String(50), nullable=True)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="offers")


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    offer_id: Mapped[str | None] = mapped_column(UUID(as_uuid=True), ForeignKey("offers.id"), nullable=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    valid_match_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_match_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_decision: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="analysis_runs")
    agent_logs = relationship("AgentLog", back_populates="analysis_run", cascade="all, delete-orphan")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("analysis_runs.id"), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    iteration_number: Mapped[int] = mapped_column(Integer, default=1)
    span_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    json_parse_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    analysis_run = relationship("AnalysisRun", back_populates="agent_logs")

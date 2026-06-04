"""Seed demo data for local development."""
import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from api.database import async_session_factory
from api.models import Product, Offer, PriceSnapshot, AnalysisRun, AnalysisStatus
from datetime import datetime, timedelta
import random


DEMO_PRODUCTS = [
    {"name": "SoundPro Wireless Bluetooth Headphones", "description": "Premium wireless headphones with Bluetooth 5.3, active noise cancellation, 40h battery life.",
     "brand": "SoundPro", "category": "Electronics", "target_price": 79.99, "currency": "USD"},
    {"name": "ErgoFit Office Chair", "description": "Ergonomic mesh office chair with lumbar support, adjustable armrests, 300lb capacity.",
     "brand": "ErgoFit", "category": "Furniture", "target_price": 349.99, "currency": "USD"},
    {"name": "GreenLeaf Organic Green Tea (100 bags)", "description": "Premium Japanese organic green tea. Rich in antioxidants. 100 tea bags per box.",
     "brand": "GreenLeaf", "category": "Groceries", "target_price": 14.99, "currency": "USD"},
    {"name": "SmartView 4K Monitor 27\"", "description": "27-inch 4K UHD IPS monitor, 99% sRGB, USB-C, height adjustable.",
     "brand": "SmartView", "category": "Electronics", "target_price": 399.99, "currency": "USD"},
    {"name": "FitTrack Pro Smartwatch", "description": "Fitness smartwatch with GPS, heart rate monitor, SpO2, 14-day battery.",
     "brand": "FitTrack", "category": "Wearables", "target_price": 199.99, "currency": "USD"},
]

DEMO_OFFERS = [
    {"product_idx": 0, "source": "serpapi", "title": "SoundPro Wireless Headphones ANC", "price": 69.99, "merchant": "Amazon"},
    {"product_idx": 0, "source": "tavily", "title": "SoundPro Bluetooth 5.3 Headset", "price": 74.99, "merchant": "Walmart"},
    {"product_idx": 0, "source": "serpapi", "title": "SoundPro Noise Cancelling Headphones", "price": 84.99, "merchant": "Best Buy"},
    {"product_idx": 1, "source": "serpapi", "title": "ErgoFit Mesh Office Chair", "price": 329.99, "merchant": "Amazon"},
    {"product_idx": 1, "source": "tavily", "title": "ErgoFit Ergonomic Chair with Lumbar", "price": 349.99, "merchant": "Office Depot"},
    {"product_idx": 2, "source": "serpapi", "title": "GreenLeaf Organic Green Tea", "price": 12.99, "merchant": "Amazon"},
    {"product_idx": 2, "source": "tavily", "title": "GreenLeaf Japanese Green Tea 100 bags", "price": 15.99, "merchant": "Walmart"},
    {"product_idx": 3, "source": "serpapi", "title": "SmartView 27\" 4K UHD Monitor", "price": 379.99, "merchant": "Amazon"},
    {"product_idx": 3, "source": "tavily", "title": "SmartView 4K IPS USB-C Monitor", "price": 399.99, "merchant": "Best Buy"},
    {"product_idx": 4, "source": "serpapi", "title": "FitTrack Pro GPS Smartwatch", "price": 179.99, "merchant": "Amazon"},
]


async def seed():
    async with async_session_factory() as db:
        for p in DEMO_PRODUCTS:
            product = Product(**p)
            db.add(product)
            await db.flush()

            for o in DEMO_OFFERS:
                if o["product_idx"] == DEMO_PRODUCTS.index(p):
                    for day_ago in [0, 7, 14, 21, 28]:
                        snap = PriceSnapshot(
                            product_id=product.id,
                            price=round(o["price"] + random.uniform(-5, 5), 2),
                            currency="USD",
                            snapshot_date=datetime.utcnow() - timedelta(days=day_ago),
                        )
                        db.add(snap)

            run = AnalysisRun(
                product_id=product.id,
                status=AnalysisStatus.completed,
                started_at=datetime.utcnow() - timedelta(hours=random.randint(1, 48)),
                completed_at=datetime.utcnow(),
                total_latency_ms=random.uniform(3000, 15000),
                candidate_count=random.randint(5, 20),
                valid_match_count=random.randint(1, 5),
                best_match_price=round(p["target_price"] * random.uniform(0.7, 1.1), 2),
                best_match_score=round(random.uniform(0.5, 1.0), 3),
                price_confidence=round(random.uniform(0.6, 0.95), 2),
                final_decision={"recommendation": f"Price at ${p['target_price'] * 0.9:.2f} to beat competition",
                                "confidence": round(random.uniform(0.6, 0.95), 2)},
            )
            db.add(run)

        await db.commit()
    print("✅ Demo data seeded: 5 products, 10 offers, price history, analysis runs")


if __name__ == "__main__":
    asyncio.run(seed())

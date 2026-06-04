import pytest

from worker.scoring import deterministic_pre_score, _is_accessory, _category_similarity


class FakeProduct:
    def __init__(self, name="", description="", category="", brand="", target_price=100.0, currency="USD"):
        self.name = name
        self.description = description
        self.category = category
        self.brand = brand
        self.target_price = target_price
        self.currency = currency


class TestDeterministicPreScore:

    def test_headphone_competitor(self):
        product = FakeProduct(
            name="Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            description="Premium wireless over-ear headphones with active noise cancellation",
            category="Electronics",
            brand="Sony",
            target_price=349.99,
        )
        candidate = {
            "title": "Bose QuietComfort 45 Wireless Noise Cancelling Headphones",
            "price": 329.99,
            "url": "https://example.com/bose-qc45",
            "merchant": "Amazon",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] > 0.2, f"Expected score > 0.2, got {result['deterministic_score']}"
        assert not result["is_accessory"], "Should not be classified as accessory"

    def test_drill_competitor(self):
        product = FakeProduct(
            name="Bosch Professional GSR 18V-50 Cordless Drill",
            description="18V brushless cordless drill driver with 50Nm torque",
            category="Power Tools",
            brand="Bosch",
            target_price=199.99,
        )
        candidate = {
            "title": "Makita 18V Cordless Drill Driver 50Nm Brushless",
            "price": 179.99,
            "url": "https://example.com/makita-drill",
            "merchant": "Amazon",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] > 0.2, f"Expected score > 0.2, got {result['deterministic_score']}"
        assert not result["is_accessory"]

    def test_laptop_competitor(self):
        product = FakeProduct(
            name="Dell XPS 15 Laptop",
            description="15.6 inch OLED laptop with Intel i9 processor, 32GB RAM",
            category="Computers",
            brand="Dell",
            target_price=2499.99,
        )
        candidate = {
            "title": "MacBook Pro 16 inch M3 Pro Laptop",
            "price": 2399.99,
            "url": "https://example.com/macbook-pro",
            "merchant": "Apple",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] > 0.15
        assert not result["is_accessory"]

    def test_circuit_breaker_competitor(self):
        product = FakeProduct(
            name="Schneider Electric iC60N 16A Circuit Breaker",
            description="Miniature circuit breaker, 16A, 10kA, 1P, C curve",
            category="Electrical Equipment",
            brand="Schneider Electric",
            target_price=25.00,
        )
        candidate = {
            "title": "ABB S201 16A Miniature Circuit Breaker MCB 10kA",
            "price": 22.50,
            "url": "https://example.com/abb-mcb",
            "merchant": "Rexel",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] > 0.15
        assert not result["is_accessory"]

    def test_kettle_competitor(self):
        product = FakeProduct(
            name="Bodum Induction 1.5L Electric Kettle",
            description="1.5 liter stainless steel electric kettle compatible with induction cooktops",
            category="Kitchen Appliances",
            brand="Bodum",
            target_price=59.99,
        )
        candidate = {
            "title": "Cuisinart 1.7L Stainless Steel Electric Kettle Induction Compatible",
            "price": 49.99,
            "url": "https://example.com/cuisinart-kettle",
            "merchant": "Target",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] > 0.15
        assert not result["is_accessory"]

    def test_accessory_rejected(self):
        product = FakeProduct(
            name="Sony WH-1000XM5 Wireless Headphones",
            description="Premium wireless noise cancelling headphones",
            category="Electronics",
            brand="Sony",
            target_price=349.99,
        )
        candidate = {
            "title": "Case for Sony WH-1000XM5 Headphones - Hard Shell Protective Travel Case",
            "price": 19.99,
            "url": "https://example.com/case",
            "merchant": "Amazon",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_accessory"], "Should be detected as accessory"
        assert result["deterministic_score"] < 0.3

    def test_replacement_part_rejected(self):
        product = FakeProduct(
            name="Bosch Professional GSR 18V-50 Drill",
            description="Cordless drill",
            category="Power Tools",
            brand="Bosch",
            target_price=199.99,
        )
        candidate = {
            "title": "Replacement Battery for Bosch 18V Power Tool Battery 5.0Ah",
            "price": 49.99,
            "url": "https://example.com/battery",
            "merchant": "eBay",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_accessory"], "Spare battery should be accessory"
        assert result["deterministic_score"] < 0.3

    def test_irrelevant_category_rejected(self):
        product = FakeProduct(
            name="Sony WH-1000XM5 Headphones",
            description="Wireless noise cancelling headphones",
            category="Electronics",
            brand="Sony",
            target_price=349.99,
        )
        candidate = {
            "title": "Premium Organic Dog Food 15kg Bag",
            "price": 89.99,
            "url": "https://example.com/dog-food",
            "merchant": "PetSmart",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] < 0.2, "Unrelated product should score very low"

    def test_cable_rejected_for_headphones(self):
        product = FakeProduct(
            name="Sony WH-1000XM5 Headphones",
            description="Wireless noise cancelling headphones",
            category="Electronics",
            brand="Sony",
            target_price=349.99,
        )
        candidate = {
            "title": "USB-C Charging Cable for Sony Headphones 1.5m",
            "price": 9.99,
            "url": "https://example.com/cable",
            "merchant": "Best Buy",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_accessory"], "Cable should be classified as accessory"

    def test_empty_inputs(self):
        product = FakeProduct(name="", category="", target_price=None)
        candidate = {"title": "", "price": None}
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.0
        assert isinstance(result["is_accessory"], bool)
        assert isinstance(result["classification_hint"], str)

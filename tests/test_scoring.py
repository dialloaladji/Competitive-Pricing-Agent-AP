import json
import pytest

from worker.scoring import (
    deterministic_pre_score,
    _is_electrical,
    _is_accessory,
    _category_similarity,
    _cross_brand_bonus,
    ELECTRICAL_BRANDS,
    ELECTRICAL_KEYWORDS,
)


class FakeProduct:
    def __init__(self, name="", description="", category="", brand="",
                 sku=None, product_type=None, target_price=8.50, currency="EUR",
                 voltage_v=230, current_a=16, poles=1, curve="C",
                 breaking_capacity_ka=6.0, phase="single", mounting="din_rail",
                 standard="IEC 60898", usage="residential"):
        self.name = name
        self.description = description
        self.category = category
        self.brand = brand
        self.sku = sku
        self.product_type = product_type
        self.target_price = target_price
        self.currency = currency
        self.voltage_v = voltage_v
        self.current_a = current_a
        self.poles = poles
        self.curve = curve
        self.breaking_capacity_ka = breaking_capacity_ka
        self.phase = phase
        self.mounting = mounting
        self.standard = standard
        self.usage = usage


class TestElectricalDomainDetection:

    def test_abb_mcb_is_electrical(self):
        assert _is_electrical("ABB S201-C16 MCB 1P 16A disjoncteur") is True

    def test_schneider_breaker_is_electrical(self):
        assert _is_electrical("Schneider Electric Easy9 1P 16A circuit breaker") is True

    def test_legrand_brand_alone_is_electrical(self):
        assert _is_electrical("Legrand") is True

    def test_headphones_not_electrical(self):
        assert _is_electrical("Sony WH-1000XM5 Wireless Headphones") is False

    def test_dog_food_not_electrical(self):
        assert _is_electrical("Premium Organic Dog Food 15kg") is False

    def test_empty_string_not_electrical(self):
        assert _is_electrical("") is False

    def test_electrical_keyword_cable(self):
        assert _is_electrical("Cable H07RN-F 3G2.5") is True

    def test_electrical_keyword_contactor(self):
        assert _is_electrical("Contactor 3P 25A 230V") is True

    def test_electrical_keyword_tableau(self):
        assert _is_electrical("Tableau électrique 24 modules") is True

    def test_electrical_keyword_ev_charger(self):
        assert _is_electrical("Borne de recharge EV 7kW") is True

    def test_hager_brand_detected(self):
        assert _is_electrical("Hager MCN116 1P 16A") is True

    def test_siemens_brand_detected(self):
        assert _is_electrical("Siemens 5SL6106 MCB") is True


class TestDeterministicPreScore:

    def test_abb_s201_vs_schneider_easy9_cross_brand(self):
        product = FakeProduct(
            name="ABB S201-C16 Disjoncteur",
            description="MCB 1P 16A courbe C 6kA rail DIN",
            category="circuit_breaker",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Schneider Electric Easy9 1P 16A Courbe C 6kA Disjoncteur modulaire",
            "price": 8.50,
            "url": "https://rexel.fr/schneider-easy9",
            "merchant": "Rexel",
            "source": "serpapi",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.40, f"Expected high score, got {result['deterministic_score']}"
        assert not result["is_accessory"]
        assert not result["is_same_brand"], "Schneider != ABB should be cross-brand"
        assert result["classification_hint"] in ("direct_competitor", "functional_equivalent")

    def test_abb_s201_vs_legrand_rx3_cross_brand(self):
        product = FakeProduct(
            name="ABB S201-C16 Disjoncteur",
            description="MCB 1P 16A",
            category="circuit_breaker",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Legrand RX3 1P 16A Courbe C Disjoncteur",
            "price": 7.80,
            "url": "https://sonepar.fr/legrand-rx3",
            "merchant": "Sonepar",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.40
        assert not result["is_same_brand"]

    def test_abb_s201_vs_abb_s201_same_brand(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="MCB 1P 16A",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "ABB S201-C16 1P 16A Disjoncteur",
            "price": 7.95,
            "url": "https://example.com/abb-s201",
            "merchant": "Rexel",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_same_brand"] is True
        assert not result["is_accessory"]

    def test_schneider_contacteur_vs_abb_contacteur_cross_brand(self):
        product = FakeProduct(
            name="Schneider LC1D25 Contactor",
            description="Contactor 3P 25A 230V bobine AC",
            category="contactor",
            brand="Schneider Electric",
            target_price=85.00,
        )
        candidate = {
            "title": "ABB AF16-30-10 Contactor 3P 25A 230V",
            "price": 78.00,
            "url": "https://example.com/abb-contactor",
            "merchant": "Rexel",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.30
        assert not result["is_same_brand"]

    def test_bobine_auxiliaire_is_accessory(self):
        product = FakeProduct(
            name="ABB S201-C16 MCB",
            description="Disjoncteur modulaire",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Bobine MX 12V pour disjoncteur ABB S200",
            "price": 24.99,
            "url": "https://example.com/bobine",
            "merchant": "123elec",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_accessory"] is True
        assert result["deterministic_score"] < 0.20

    def test_bornier_is_accessory(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="Disjoncteur",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Bornier de raccordement 4mm pour tableau électrique",
            "price": 3.50,
            "url": "https://example.com/bornier",
            "merchant": "123elec",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_accessory"] is True

    def test_non_electrical_product_rejected(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="Disjoncteur",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Premium Organic Dog Food 15kg Bag",
            "price": 45.00,
            "url": "https://example.com/dogfood",
            "merchant": "PetSmart",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] < 0.10, f"Expected very low, got {result['deterministic_score']}"

    def test_usb_cable_rejected(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="Disjoncteur",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "USB-C Cable for ABB",
            "price": 9.99,
            "url": "https://example.com/cable",
            "merchant": "Amazon",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_accessory"] is True

    def test_price_mismatch_cheaper(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="MCB 1P 16A",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Chint NXB-63 1P 16A C 6kA MCB",
            "price": 4.20,
            "url": "https://example.com/chint",
            "merchant": "123elec",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.30
        assert not result["is_same_brand"]

    def test_premium_alternative_siemens(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="MCB 1P 16A C 6kA",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Siemens 5SL6106-6 1P 16A MCB Curve C 6kA",
            "price": 12.50,
            "url": "https://example.com/siemens",
            "merchant": "Sonepar",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.30
        assert not result["is_same_brand"]

    def test_previous_generation_schneider(self):
        product = FakeProduct(
            name="Schneider iC60N 1P 16A",
            description="MCB Acti9",
            brand="Schneider Electric",
            target_price=10.00,
        )
        candidate = {
            "title": "Schneider C60N 1P 16A C 6kA (older series)",
            "price": 8.50,
            "url": "https://example.com/c60",
            "merchant": "Rexel",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["is_same_brand"] is True
        assert result["deterministic_score"] >= 0.30

    def test_keine_elektrische_keywords(self):
        product = FakeProduct(
            name="Bicycle Helmet",
            description="Adult road bike helmet with ventilation",
            brand="GenericSports",
            target_price=80.0,
        )
        candidate = {
            "title": "Premium Cycling Helmet Lightweight",
            "price": 79.99,
            "url": "https://example.com/helmet",
            "merchant": "Shop",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] < 0.40
        assert result["is_same_brand"] is False
        assert result["classification_hint"] != "direct_competitor"

    def test_empty_inputs(self):
        product = FakeProduct(name="", description="", category="", brand="", target_price=None)
        candidate = {"title": "", "price": None}
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.0
        assert isinstance(result["is_accessory"], bool)
        assert isinstance(result["classification_hint"], str)

    def test_chint_chinese_brand_cross_brand(self):
        product = FakeProduct(
            name="ABB S201-C16",
            description="Disjoncteur",
            brand="ABB",
            target_price=8.50,
        )
        candidate = {
            "title": "Chint NXB-63 1P 16A 6kA Disjoncteur",
            "price": 4.20,
            "url": "https://example.com/chint",
            "merchant": "123elec",
        }
        result = deterministic_pre_score(product, candidate)
        assert result["deterministic_score"] >= 0.20
        assert not result["is_same_brand"]


class TestCategorySimilarity:

    def test_circuit_breaker_category(self):
        product_category = "circuit_breaker"
        candidate = {"title": "MCB 1P 16A disjoncteur modulaire"}
        score = _category_similarity(product_category, candidate)
        assert score >= 0.15

    def test_contactor_category(self):
        candidate = {"title": "Contactor 3P 25A 230V"}
        score = _category_similarity("contactor", candidate)
        assert score >= 0.15

    def test_irrelevant_category(self):
        candidate = {"title": "Dog Food Premium 15kg"}
        score = _category_similarity("circuit_breaker", candidate)
        assert score == 0.0


class TestCrossBrandBonus:

    def test_different_brand_returns_bonus(self):
        bonus = _cross_brand_bonus("ABB", "Schneider Easy9 1P 16A")
        assert bonus == 0.15

    def test_same_brand_returns_zero(self):
        bonus = _cross_brand_bonus("ABB", "ABB S201-C16")
        assert bonus == 0.0

    def test_unknown_brand_in_title_still_gets_bonus(self):
        bonus = _cross_brand_bonus("ABB", "CEMBRE 1P 16A")
        assert bonus == 0.15

    def test_empty_brand_no_bonus(self):
        assert _cross_brand_bonus("", "Schneider Easy9") == 0.0
        assert _cross_brand_bonus(None, "Schneider Easy9") == 0.0


class TestAccessoryDetection:

    def test_bobine_pour(self):
        assert _is_accessory("Bobine pour disjoncteur Schneider") is True

    def test_module_pour(self):
        assert _is_accessory("Module auxiliaire pour contacteur") is True

    def test_rail_din_pour(self):
        assert _is_accessory("Rail DIN pour tableau") is True

    def test_enjoliveur_pour(self):
        assert _is_accessory("Enjoliveur pour prise Legrand") is True

    def test_plaque_pour(self):
        assert _is_accessory("Plaque pour interrupteur Schneider") is True

    def test_direct_mcb_not_accessory(self):
        assert _is_accessory("MCB 1P 16A Disjoncteur") is False

    def test_direct_cable_not_accessory(self):
        assert _is_accessory("Cable H07RN-F 3G2.5") is False


class TestElectricalKeywordsAndBrands:

    def test_keywords_count(self):
        assert len(ELECTRICAL_KEYWORDS) >= 100, f"Expected 100+ keywords, got {len(ELECTRICAL_KEYWORDS)}"

    def test_brands_count(self):
        assert len(ELECTRICAL_BRANDS) >= 20, f"Expected 20+ brands, got {len(ELECTRICAL_BRANDS)}"

    def test_main_brands_in_list(self):
        expected = ["abb", "schneider", "legrand", "siemens", "eaton", "hager"]
        for brand in expected:
            assert brand in ELECTRICAL_BRANDS, f"Missing brand: {brand}"


class TestElectricalFixtures:

    def test_fixtures_load(self):
        from pathlib import Path
        fixtures_path = Path(__file__).parent / "fixtures" / "electrical_products.json"
        with open(fixtures_path) as f:
            data = json.load(f)
        assert "products" in data
        assert len(data["products"]) >= 5

    def test_fixtures_are_electrical(self):
        from pathlib import Path
        fixtures_path = Path(__file__).parent / "fixtures" / "electrical_products.json"
        with open(fixtures_path) as f:
            data = json.load(f)
        for product in data["products"]:
            text = f"{product['name']} {product.get('description', '')} {product.get('brand', '')}"
            assert _is_electrical(text), f"Not electrical: {product['name']}"

    def test_fixtures_have_specs(self):
        from pathlib import Path
        fixtures_path = Path(__file__).parent / "fixtures" / "electrical_products.json"
        with open(fixtures_path) as f:
            data = json.load(f)
        for product in data["products"]:
            assert "product_type" in product
            assert "brand" in product
            assert "expected_cross_brand_brands" in product
            assert len(product["expected_cross_brand_brands"]) >= 3

    def test_fixtures_pass_pre_score(self):
        from pathlib import Path
        fixtures_path = Path(__file__).parent / "fixtures" / "electrical_products.json"
        with open(fixtures_path) as f:
            data = json.load(f)
        for product in data["products"]:
            for expected_brand in product["expected_cross_brand_brands"]:
                assert expected_brand.lower() in [b.lower() for b in ELECTRICAL_BRANDS] or \
                       any(part in [b.lower() for b in ELECTRICAL_BRANDS]
                           for part in expected_brand.lower().split()), \
                    f"Expected brand '{expected_brand}' for {product['name']} not in ELECTRICAL_BRANDS"

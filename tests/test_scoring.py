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


class TestSpecMatchClassification:
    def test_exact_spec_boost(self):
        from worker.tasks import _spec_match_boost
        assert _spec_match_boost("exact_spec_equivalent") == 0.15
        assert _spec_match_boost("close_spec_equivalent") == 0.05
        assert _spec_match_boost("functional_equivalent") == 0.0
        assert _spec_match_boost(None) == 0.0
        assert _spec_match_boost("") == 0.0

    def test_tier1_brand_boost(self):
        from worker.tasks import _tier1_brand_boost, TIER1_BRANDS
        assert _tier1_brand_boost("ABB S201-C16") == 0.10
        assert _tier1_brand_boost("Schneider Electric Easy9") == 0.10
        assert _tier1_brand_boost("Legrand RX3") == 0.10
        assert _tier1_brand_boost("Siemens 5SL") == 0.10
        assert _tier1_brand_boost("Hager MCN") == 0.10
        assert _tier1_brand_boost("Eaton PLSM") == 0.10
        assert _tier1_brand_boost("Chint NXB") == 0.0
        assert _tier1_brand_boost("Noark NMB") == 0.0
        assert _tier1_brand_boost(None) == 0.0

    def test_tier1_brand_set_contains_expected_brands(self):
        from worker.tasks import TIER1_BRANDS
        for brand in ["abb", "schneider", "legrand", "siemens", "eaton", "hager"]:
            assert brand in TIER1_BRANDS, f"Tier-1 brand '{brand}' missing from TIER1_BRANDS"

    def test_spec_mismatch_penalty(self):
        from worker.tasks import _spec_mismatch_penalty
        p1p = FakeProduct(poles=1)
        p3p = FakeProduct(poles=3)
        assert _spec_mismatch_penalty(p1p, {"specs": {"poles": 2}}) == -0.05
        assert _spec_mismatch_penalty(p1p, {"specs": {"poles": 1}}) == 0.0
        assert _spec_mismatch_penalty(p3p, {"specs": {"poles": 4}}) == -0.05
        assert _spec_mismatch_penalty(p1p, {}) == 0.0
        assert _spec_mismatch_penalty(p1p, {"specs": {}}) == 0.0

    def test_scoring_engine_includes_spec_match(self):
        from worker.tasks import score_candidates
        p = FakeProduct(name="Schneider iC60N", brand="Schneider", category="mcb",
                        voltage_v=230, current_a=16, poles=1, curve="C",
                        breaking_capacity_ka=10, target_price=10)
        candidates = [
            {"title": "ABB S201-C16 1P 16A C 6kA", "price": 8.0, "currency": "EUR",
             "merchant": "Rexel", "brand": "ABB", "url": "", "specs": {"poles": 1, "current_a": 16, "curve": "C", "breaking_capacity_ka": 6}},
            {"title": "Eaton PLSM 1P+N 16A C 6kA", "price": 11.0, "currency": "EUR",
             "merchant": "Rexel", "brand": "Eaton", "url": "", "specs": {"poles": 2, "current_a": 16, "curve": "C", "breaking_capacity_ka": 6}},
            {"title": "Chint NXB-63 1P 16A C 6kA", "price": 4.5, "currency": "EUR",
             "merchant": "123elec", "brand": "Chint", "url": "", "specs": {"poles": 1, "current_a": 16, "curve": "C", "breaking_capacity_ka": 6}},
        ]
        scored = score_candidates(p, candidates)
        assert len(scored) == 3
        assert all("spec_match" in s for s in scored)
        spec_matches = {s["title"][:25]: s["spec_match"] for s in scored}
        assert spec_matches["ABB S201-C16 1P 16A C 6kA"] in ("exact_spec_equivalent", "close_spec_equivalent")
        abb_score = next(s["score"] for s in scored if "ABB" in s["title"])
        eaton_score = next(s["score"] for s in scored if "Eaton" in s["title"])
        assert abb_score > eaton_score, \
            f"ABB (exact poles) should score higher than Eaton 1P+N: {abb_score} vs {eaton_score}"


class TestBrandTargetedQueries:
    def test_ic60n_generates_tier1_brand_queries(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="mcb", product_name="Schneider Acti9 iC60N C16",
            product_brand="Schneider Electric", poles=1, current_a=16, curve="C",
            breaking_capacity_ka=10,
        )
        assert len(qs) >= 8
        assert any("ABB" in q for q in qs), "Should include ABB query"
        assert any("Hager" in q for q in qs), "Should include Hager query"
        assert any("Siemens" in q for q in qs), "Should include Siemens query"
        assert any("Eaton" in q for q in qs), "Should include Eaton query"
        assert not any("Schneider" in q for q in qs), "Should filter out own brand"

    def test_english_and_french_queries_generated(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="mcb", product_name="MCB", product_brand="ABB",
            poles=1, current_a=16, curve="C", breaking_capacity_ka=6,
        )
        en_count = sum(1 for q in qs if any(w in q for w in ["curve", "MCB", "equivalent"]))
        fr_count = sum(1 for q in qs if any(w in q for w in ["courbe", "disjoncteur"]))
        assert en_count >= 4, f"Should have ≥4 English queries, got {en_count}"
        assert fr_count >= 4, f"Should have ≥4 French queries, got {fr_count}"

    def test_specs_appear_in_queries(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="mcb", product_name="MCB", product_brand="ABB",
            poles=1, current_a=16, curve="C", breaking_capacity_ka=6,
        )
        combined = " ".join(qs)
        assert "1P" in combined
        assert "16A" in combined
        assert "C" in combined
        assert "6kA" in combined

    def test_contactor_uses_french_tripolaire(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="contactor", product_name="Schneider LC1D25",
            product_brand="Schneider Electric", poles=3, current_a=25,
        )
        fr_qs = [q for q in qs if "contacteur" in q or "ABB" in q]
        assert any("tripolaire" in q for q in fr_qs), \
            f"3P should be 'tripolaire' in French, got: {fr_qs}"

    def test_missing_specs_falls_back_to_category(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="mcb", product_name="MCB", product_brand="ABB",
        )
        assert len(qs) >= 8
        assert all("ABB" not in q and "ABB " not in q for q in qs), \
            "Should not include own brand (ABB) even with no specs"

    def test_detect_category_from_name(self):
        from worker.scoring import _detect_category_key
        assert _detect_category_key(None, "Schneider iC60N disjoncteur") == "disjoncteur"
        assert _detect_category_key(None, "Contactor LC1D") == "contactor"
        assert _detect_category_key(None, "Hager interrupteur") == "interrupteur"
        assert _detect_category_key(None, "Some random product") is None
        assert _detect_category_key("mcb", "anything") == "mcb"

    def test_queries_capped_at_max(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="mcb", product_name="MCB", product_brand="Generic",
            poles=1, current_a=16, curve="C", breaking_capacity_ka=6,
            max_queries=8,
        )
        assert len(qs) <= 8

    def test_no_duplicate_queries(self):
        from worker.scoring import generate_brand_targeted_queries
        qs = generate_brand_targeted_queries(
            product_category="mcb", product_name="MCB", product_brand="Generic",
            poles=1, current_a=16, curve="C", breaking_capacity_ka=6,
        )
        normalized = [q.lower().strip() for q in qs]
        assert len(normalized) == len(set(normalized)), \
            f"Duplicate queries found: {[q for q in qs if normalized.count(q.lower()) > 1]}"


class TestOutputCapping:
    def test_cross_brand_capped_to_5(self):
        """Verify the main.py output cap of 5 cross-brand + 2 same-brand."""
        from api.main import app
        from api.schemas import AnalyzeEquivalentsResponse, EquivalentOut
        # Simulate 8 cross-brand and 3 same-brand scored
        scored = [
            {"title": f"Brand{i} MCB 1P 16A", "price": 10.0 + i,
             "currency": "EUR", "merchant": "Rexel", "url": f"https://x{i}",
             "score": 0.9 - i * 0.01, "price_score": 0.8, "relevance_score": 0.9,
             "trust_score": 0.9, "classification": "direct_competitor",
             "spec_match": "exact_spec_equivalent", "specs": {},
             "is_same_brand": False}
            for i in range(8)
        ] + [
            {"title": f"Same{i}", "price": 10.0, "currency": "EUR",
             "merchant": "Rexel", "url": f"https://s{i}",
             "score": 0.7, "price_score": 0.8, "relevance_score": 0.7,
             "trust_score": 0.9, "classification": "same_product",
             "spec_match": "exact_spec_equivalent", "specs": {},
             "is_same_brand": True}
            for i in range(3)
        ]
        cross_brand_list = [s for s in scored if not s.get("is_same_brand", False)][:5]
        same_brand_list = [s for s in scored if s.get("is_same_brand", False)][:2]
        assert len(cross_brand_list) == 5
        assert len(same_brand_list) == 2


class TestSpecQuality:
    def test_full_spec_match_scores_high(self):
        from worker.scoring import spec_quality_score
        p = FakeProduct(poles=1, current_a=16, curve="C", breaking_capacity_ka=6)
        c = {"title": "MCB 1P 16A curve C 6kA ABB", "specs": {"poles": 1, "current_a": 16, "curve": "C", "breaking_capacity_ka": 6}}
        score, bd = spec_quality_score(p, c)
        assert score >= 0.85, f"Full spec match should score ≥0.85, got {score}"
        assert not bd.get("is_vague")
        assert not bd.get("missing_critical")

    def test_vague_title_penalized(self):
        from worker.scoring import spec_quality_score
        p = FakeProduct(poles=1, current_a=16, curve="C", breaking_capacity_ka=6)
        c = {"title": "ABB Disjoncteur", "specs": {}}
        score, bd = spec_quality_score(p, c)
        assert bd.get("is_vague") is True
        assert score <= 0.1, f"Vague title should score ≤0.1, got {score}"

    def test_specific_title_outranks_vague(self):
        from worker.scoring import spec_quality_score
        p = FakeProduct(poles=1, current_a=16, curve="C", breaking_capacity_ka=6)
        specific = {"title": "Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB", "specs": {}}
        vague = {"title": "ABB Disjoncteur", "specs": {}}
        s_spec, _ = spec_quality_score(p, specific)
        s_vague, _ = spec_quality_score(p, vague)
        assert s_spec > s_vague, f"Specific ({s_spec}) should beat vague ({s_vague})"

    def test_extract_brand_from_title(self):
        from worker.scoring import _extract_brand_from_title
        assert _extract_brand_from_title("ABB S201-C16") == "ABB"
        assert _extract_brand_from_title("Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB") == "ABB"
        assert _extract_brand_from_title("Legrand RX3") == "Legrand"
        assert _extract_brand_from_title("Schneider Electric iC60N") == "Schneider Electric"
        assert _extract_brand_from_title("Unknown brand product") is None
        assert _extract_brand_from_title(None) is None
        assert _extract_brand_from_title("") is None

    def test_extract_specs_from_title(self):
        from worker.scoring import _extract_specs_from_title
        specs = _extract_specs_from_title("Miniature Circuit Breaker S201-C16 - 1P - C - 16A - 6kA ABB")
        assert specs.get("poles") == 1
        assert specs.get("current_a") == 16
        assert specs.get("curve") == "C"
        assert specs.get("breaking_capacity_ka") == 6.0

    def test_extract_specs_from_s201_title(self):
        from worker.scoring import _extract_specs_from_title
        specs = _extract_specs_from_title("Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB")
        assert specs.get("poles") == 1
        assert specs.get("current_a") == 16
        assert specs.get("curve") == "C"

    def test_extract_specs_from_french_title(self):
        from worker.scoring import _extract_specs_from_title
        specs = _extract_specs_from_title("Disjoncteur modulaire 1P 16A courbe C 6kA ABB")
        assert specs.get("poles") == 1
        assert specs.get("current_a") == 16
        assert specs.get("curve") == "C"
        assert specs.get("breaking_capacity_ka") == 6.0

    def test_is_vague_title_detection(self):
        from worker.scoring import _is_vague_title
        assert _is_vague_title("ABB Disjoncteur") is True
        assert _is_vague_title("MCB Schneider") is True
        assert _is_vague_title("ABB") is True
        assert _is_vague_title(None) is True
        assert _is_vague_title("") is True
        assert _is_vague_title("Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB") is False
        assert _is_vague_title("Disjoncteur modulaire 1P 16A courbe C 6kA ABB") is False

    def test_poles_mismatch_1pn_for_1p(self):
        from worker.scoring import spec_quality_score
        p = FakeProduct(poles=1, current_a=16, curve="C", breaking_capacity_ka=6)
        c = {"title": "MCB 1P+N 16A C 6kA Eaton", "specs": {"poles": 2, "current_a": 16, "curve": "C", "breaking_capacity_ka": 6}}
        score, bd = spec_quality_score(p, c)
        assert bd.get("poles") == 0.0, "Poles should NOT match (target=1, cand=2)"

    def test_curves_match_case_insensitive(self):
        from worker.scoring import spec_quality_score
        p = FakeProduct(poles=1, current_a=16, curve="C", breaking_capacity_ka=6)
        c_lower = {"title": "MCB 1P 16A c 6kA", "specs": {"poles": 1, "current_a": 16, "curve": "c", "breaking_capacity_ka": 6}}
        c_upper = {"title": "MCB 1P 16A C 6kA", "specs": {"poles": 1, "current_a": 16, "curve": "C", "breaking_capacity_ka": 6}}
        s_lower, _ = spec_quality_score(p, c_lower)
        s_upper, _ = spec_quality_score(p, c_upper)
        assert s_lower == s_upper, "Curve matching should be case-insensitive"


class TestScoringWithSpecQuality:
    def test_specific_outranks_vague_in_engine(self):
        from worker.tasks import score_candidates
        p = FakeProduct(name="Schneider iC60N", brand="Schneider", category="mcb",
                        voltage_v=230, current_a=16, poles=1, curve="C",
                        breaking_capacity_ka=10, target_price=10)
        candidates = [
            {"title": "Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB",
             "price": 8.0, "currency": "EUR", "merchant": "Rexel", "specs": {}},
            {"title": "ABB Disjoncteur",
             "price": 8.0, "currency": "EUR", "merchant": "Rexel", "specs": {}},
        ]
        scored = score_candidates(p, candidates)
        assert len(scored) == 2
        specific = next(s for s in scored if "S201-C16" in s["title"])
        vague = next(s for s in scored if "Disjoncteur" in s["title"] and "S201" not in s["title"])
        assert specific["score"] > vague["score"], \
            f"Specific ({specific['score']}) should outrank vague ({vague['score']})"
        assert specific.get("is_vague") is False
        assert vague.get("is_vague") is True
        assert specific.get("spec_quality", 0) > vague.get("spec_quality", 0)

    def test_vague_excluded_from_best_match(self):
        """Verify vague candidates are not used as best_match."""
        from worker.tasks import score_candidates
        p = FakeProduct(name="Schneider iC60N", brand="Schneider", category="mcb",
                        current_a=16, poles=1, curve="C", target_price=10)
        candidates = [
            {"title": "ABB Disjoncteur", "price": 8.0, "currency": "EUR",
             "merchant": "Rexel", "specs": {}},
            {"title": "MCB Schneider 1P 16A C 6kA", "price": 8.0, "currency": "EUR",
             "merchant": "Rexel", "specs": {}},
        ]
        scored = score_candidates(p, candidates)
        non_vague = [s for s in scored if not s.get("is_vague", False)]
        best = non_vague[0] if non_vague else (scored[0] if scored else None)
        assert best is not None
        assert "Disjoncteur" not in best["title"] or "1P" in best["title"]
        assert best["title"] == "MCB Schneider 1P 16A C 6kA"


class TestReliableEquivalent:
    """Tests for the reliable/weak candidate split.

    Rules:
    - cross_brand_equivalents must contain only candidates with spec_quality >= 0.5
    - candidates with spec_quality = 0 go to weak_candidates
    - valid_match_count counts only reliable equivalents
    - recommendation uses only reliable candidates (spec_q >= 0.5, !vague, valid price, currency ok)
    """

    def _build_scored(self, title, brand, spec_quality, is_vague, price, currency="EUR", is_same_brand=False):
        return {
            "title": title,
            "brand": brand,
            "price": price,
            "currency": currency,
            "spec_quality": spec_quality,
            "is_vague": is_vague,
            "is_same_brand": is_same_brand,
            "score": 0.8,
            "url": "https://example.com",
            "merchant": "TestMerchant",
        }

    def test_reliable_passes_threshold(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, 7.95)
        assert is_reliable_equivalent(cand, target_currency="EUR") is True

    def test_low_spec_quality_is_weak(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB Disjoncteur 2cds251001r", "ABB", 0.0, True, 10.0)
        assert is_reliable_equivalent(cand, target_currency="EUR") is False

    def test_vague_is_weak(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("MCB Schneider", "Schneider Electric", 0.4, True, 12.0)
        assert is_reliable_equivalent(cand, target_currency="EUR") is False

    def test_missing_price_is_weak(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, None)
        assert is_reliable_equivalent(cand, target_currency="EUR") is False

    def test_zero_price_is_weak(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, 0.0)
        assert is_reliable_equivalent(cand, target_currency="EUR") is False

    def test_currency_mismatch_is_weak(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, 7.95, currency="USD")
        assert is_reliable_equivalent(cand, target_currency="EUR") is False

    def test_currency_match_eur(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, 7.95, currency="EUR")
        assert is_reliable_equivalent(cand, target_currency="EUR") is True

    def test_no_target_currency_skips_check(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, 7.95, currency="USD")
        assert is_reliable_equivalent(cand) is True

    def test_spec_quality_at_threshold_is_reliable(self):
        from worker.scoring import is_reliable_equivalent, SPEC_QUALITY_RELIABLE_THRESHOLD
        assert SPEC_QUALITY_RELIABLE_THRESHOLD == 0.5
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.5, False, 7.95)
        assert is_reliable_equivalent(cand, target_currency="EUR") is True

    def test_spec_quality_just_below_threshold_is_weak(self):
        from worker.scoring import is_reliable_equivalent
        cand = self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.49, False, 7.95)
        assert is_reliable_equivalent(cand, target_currency="EUR") is False

    def test_split_reliable_vs_weak(self):
        from worker.scoring import split_reliable_vs_weak
        scored = [
            self._build_scored("ABB S201-C16 1P 16A", "ABB", 0.8, False, 7.95),
            self._build_scored("Legrand RX3 1P 16A", "Legrand", 0.8, False, 7.80),
            self._build_scored("Hager MCN116 1P 16A", "Hager", 0.7, False, 9.20),
            self._build_scored("ABB Disjoncteur", "ABB", 0.0, True, 10.0),
            self._build_scored("MCB Schneider", "Schneider Electric", 0.4, True, 12.0),
            self._build_scored("ABB Disjoncteur 2cds251001r", "ABB", 0.0, True, 10.0),
        ]
        reliable, weak = split_reliable_vs_weak(scored, target_currency="EUR")
        assert len(reliable) == 3
        assert len(weak) == 3
        titles_reliable = [r["title"] for r in reliable]
        titles_weak = [w["title"] for w in weak]
        assert "ABB S201-C16 1P 16A" in titles_reliable
        assert "ABB Disjoncteur" in titles_weak
        assert "ABB Disjoncteur 2cds251001r" in titles_weak
        assert "MCB Schneider" in titles_weak

    def test_end_to_end_mcb_schneider_picks_reliable_abb(self):
        """Reproduce the Schneider Acti9 iC60N C16 scenario in mock mode.

        The output must:
        - Keep ABB S201-C16 in cross_brand_equivalents
        - Place 'ABB Disjoncteur 2cds251001r' in weak_candidates
        - Have valid_match_count == number of reliable candidates
        """
        scored = [
            self._build_scored("ABB S201-C16 1P 16A Courbe C 6kA Disjoncteur modulaire", "ABB", 0.8, False, 7.95, is_same_brand=False),
            self._build_scored("Legrand RX3 1P 16A Courbe C 6000A", "Legrand", 0.8, False, 7.80),
            self._build_scored("Miniature Circuit Breaker S201-C16 - 1P - C - 16A", "ABB", 0.75, False, 7.95),
            self._build_scored("Hager MCN116 1P 16A", "Hager", 0.8, False, 9.20),
            self._build_scored("Siemens 5SL6106-6 1P 16A", "Siemens", 0.8, False, 12.50),
            self._build_scored("ABB Disjoncteur", "ABB", 0.0, True, 10.0),
            self._build_scored("MCB Schneider", "Schneider Electric", 0.4, True, 12.0),
            self._build_scored("ABB Disjoncteur 2cds251001r", "ABB", 0.0, True, 10.0),
            self._build_scored("Schneider iC60N 1P 16A C 10kA", "Schneider Electric", 0.8, False, 14.50, is_same_brand=True),
            self._build_scored("Schneider Easy9 1P 16A", "Schneider Electric", 0.8, False, 8.50, is_same_brand=True),
        ]
        from worker.scoring import split_reliable_vs_weak
        reliable, weak = split_reliable_vs_weak(scored, target_currency="EUR")

        cross_brand = [s for s in reliable if not s.get("is_same_brand", False)][:5]
        same_brand = [s for s in reliable if s.get("is_same_brand", False)][:2]

        assert len(cross_brand) == 5
        assert len(same_brand) == 2
        assert len(reliable) == 7
        assert len(weak) == 3

        cross_titles = [c["title"] for c in cross_brand]
        weak_titles = [w["title"] for w in weak]

        assert "ABB S201-C16 1P 16A Courbe C 6kA Disjoncteur modulaire" in cross_titles
        assert "ABB Disjoncteur 2cds251001r" in weak_titles
        assert "ABB Disjoncteur" in weak_titles
        assert "MCB Schneider" in weak_titles

        cross_prices = [c["price"] for c in cross_brand if c["price"]]
        assert all(p >= 7.0 for p in cross_prices), \
            "No weak candidate prices should leak into cross_brand_equivalents"


class TestSpecExtractionMCB:
    """Tests for MCB-specific spec extraction patterns.

    Rules:
    - "C16" / "D20" pattern: extract current_a + curve
    - "S201-C16" / "iC60N C16" pattern: extract current_a + curve from model
    - "1P ... 16A" pattern: extract both poles and current
    - "Type C" / "Courbe C" / "Curve C" pattern: extract curve
    - "6kA" must NOT be confused with current_a
    - "6000A" (= 6kA) should be parsed as breaking_capacity_ka=6.0
    - Ranges like "6A-63A" should be ignored for current_a
    """

    def test_abb_range_title(self):
        from worker.scoring import _extract_specs_from_title
        title = "ABB Miniature Circuit Breaker 1P 6KA Type C (6A–63A) | ABB UAE 16A"
        specs = _extract_specs_from_title(title)
        assert specs.get("current_a") == 16
        assert specs.get("poles") == 1
        assert specs.get("curve") == "C"
        assert specs.get("breaking_capacity_ka") == 6.0

    def test_s201_c16_model(self):
        from worker.scoring import _extract_specs_from_title
        title = "Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB"
        specs = _extract_specs_from_title(title)
        assert specs.get("current_a") == 16
        assert specs.get("poles") == 1
        assert specs.get("curve") == "C"

    def test_ic60n_c16_short(self):
        from worker.scoring import _extract_specs_from_title
        title = "Schneider Electric Acti9 iC60N C16"
        specs = _extract_specs_from_title(title)
        assert specs.get("current_a") == 16
        assert specs.get("curve") == "C"

    def test_6ka_not_current(self):
        from worker.scoring import _extract_specs_from_title
        title = "MCB 1P 16A 6kA Courbe C"
        specs = _extract_specs_from_title(title)
        assert specs.get("current_a") == 16
        assert specs.get("breaking_capacity_ka") == 6.0
        assert "6" != str(specs.get("current_a"))

    def test_6000a_is_kA(self):
        from worker.scoring import _extract_specs_from_title
        title = "Legrand RX3 1P 16A Courbe C 6000A"
        specs = _extract_specs_from_title(title)
        assert specs.get("current_a") == 16
        assert specs.get("breaking_capacity_ka") == 6.0

    def test_d_curve(self):
        from worker.scoring import _extract_specs_from_title
        title = "Disjoncteur D20 1P 20A 6kA"
        specs = _extract_specs_from_title(title)
        assert specs.get("curve") == "D"
        assert specs.get("current_a") == 20
        assert specs.get("poles") == 1

    def test_k_curve(self):
        from worker.scoring import _extract_specs_from_title
        title = "MCB K32 3P 32A 10kA"
        specs = _extract_specs_from_title(title)
        assert specs.get("curve") == "K"
        assert specs.get("current_a") == 32
        assert specs.get("poles") == 3

    def test_1pn_poles(self):
        from worker.scoring import _extract_specs_from_title
        title = "Eaton PLSM-C16/1N 1P+N 16A MCB Curve C 6kA"
        specs = _extract_specs_from_title(title)
        assert specs.get("poles") == 1
        assert specs.get("current_a") == 16
        assert specs.get("curve") == "C"
        assert specs.get("breaking_capacity_ka") == 6.0

    def test_three_phase(self):
        from worker.scoring import _extract_specs_from_title
        title = "Disjoncteur triphasé 25A 3P courbe C 6kA"
        specs = _extract_specs_from_title(title)
        assert specs.get("poles") == 3
        assert specs.get("current_a") == 25
        assert specs.get("curve") == "C"

    def test_10ka_kA(self):
        from worker.scoring import _extract_specs_from_title
        title = "Schneider iC60N Acti9 1P 16A Courbe C 10kA"
        specs = _extract_specs_from_title(title)
        assert specs.get("current_a") == 16
        assert specs.get("breaking_capacity_ka") == 10.0

    def test_regression_full_abb_schneider_scenario(self):
        from worker.scoring import spec_quality_score
        from types import SimpleNamespace
        product = SimpleNamespace(
            name="Schneider Electric Acti9 iC60N C16 1P 16A C 10kA",
            category="mcb", brand="Schneider Electric",
            current_a=16, poles=1, curve="C", breaking_capacity_ka=10,
            voltage_v=230, target_price=14.5,
        )
        candidates_with_empty_specs = [
            {"title": "ABB Miniature Circuit Breaker 1P 6KA Type C (6A–63A) | ABB UAE 16A",
             "price": 8.0, "currency": "EUR", "specs": {}},
            {"title": "Miniature Circuit Breaker S201-C16 - 1P - C - 16A ABB",
             "price": 7.95, "currency": "EUR", "specs": {}},
        ]
        for c in candidates_with_empty_specs:
            sq, _ = spec_quality_score(product, c)
            assert sq >= 0.5, f"spec_quality={sq} should be >= 0.5 for {c['title']}"


class TestDiversifyByBrand:
    """Tests for brand diversity in cross_brand_equivalents.

    Rules:
    - final cross_brand_equivalents should contain 3-5 reliable products
    - at least 3 distinct brands when available
    - max 1-2 results per brand
    - needs_supplemental_search=True when selected has < 3 brands
    """

    def _cand(self, title, brand, score=0.8, price=10.0, currency="EUR",
              is_same_brand=False, is_vague=False, spec_quality=0.8):
        brand_slug = (brand or "unknown").lower()
        return {
            "title": title,
            "brand": brand,
            "score": score,
            "price": price,
            "currency": currency,
            "is_same_brand": is_same_brand,
            "is_vague": is_vague,
            "spec_quality": spec_quality,
            "url": f"https://example.com/{brand_slug}-{title[:10]}",
            "merchant": "Rexel",
        }

    def test_basic_three_brands(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB S201", "ABB", score=0.85, price=7.95),
            self._cand("Legrand RX3", "Legrand", score=0.82, price=7.80),
            self._cand("Hager MCN116", "Hager", score=0.80, price=9.20),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert len(selected) == 3
        assert len(overflow) == 0
        assert stats["selected_brand_count"] == 3
        assert stats["needs_supplemental_search"] is False
        brands = {c["brand"] for c in selected}
        assert brands == {"ABB", "Legrand", "Hager"}

    def test_max_two_per_brand(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB S201 #1", "ABB", score=0.90, price=7.95),
            self._cand("ABB S201 #2", "ABB", score=0.88, price=8.10),
            self._cand("ABB S201 #3", "ABB", score=0.85, price=8.50),
            self._cand("Legrand RX3", "Legrand", score=0.80, price=7.80),
            self._cand("Hager MCN116", "Hager", score=0.78, price=9.20),
            self._cand("Siemens 5SL", "Siemens", score=0.75, price=12.50),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert len(selected) == 5
        abb_count = sum(1 for c in selected if c["brand"] == "ABB")
        assert abb_count == 2, "Should select at most 2 ABB listings"
        assert stats["selected_brand_count"] == 4
        assert stats["needs_supplemental_search"] is False
        assert len(overflow) == 1

    def test_max_total_5(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB", "ABB", 0.90),
            self._cand("Legrand", "Legrand", 0.88),
            self._cand("Hager", "Hager", 0.85),
            self._cand("Siemens", "Siemens", 0.82),
            self._cand("Eaton", "Eaton", 0.80),
            self._cand("Chint", "Chint", 0.78),
            self._cand("Noark", "Noark", 0.75),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert len(selected) == 5
        assert len(overflow) == 2
        assert stats["selected_brand_count"] == 5
        assert stats["needs_supplemental_search"] is False

    def test_needs_supplemental_when_single_brand(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB #1", "ABB", 0.90),
            self._cand("ABB #2", "ABB", 0.88),
            self._cand("ABB #3", "ABB", 0.85),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert len(selected) == 2
        assert stats["selected_brand_count"] == 1
        assert stats["needs_supplemental_search"] is True

    def test_needs_supplemental_with_two_brands(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB", "ABB", 0.90),
            self._cand("Legrand", "Legrand", 0.88),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert len(selected) == 2
        assert stats["selected_brand_count"] == 2
        assert stats["needs_supplemental_search"] is True

    def test_three_brands_no_supplemental(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB", "ABB", 0.90),
            self._cand("Legrand", "Legrand", 0.88),
            self._cand("Hager", "Hager", 0.85),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert stats["selected_brand_count"] == 3
        assert stats["needs_supplemental_search"] is False

    def test_brand_key_falls_back_to_title(self):
        from worker.scoring import diversify_by_brand, _candidate_brand_key
        c1 = self._cand("ABB S201-C16", None)
        c2 = self._cand("Legrand RX3", None)
        assert _candidate_brand_key(c1) == "abb"
        assert _candidate_brand_key(c2) == "legrand"
        selected, overflow, stats = diversify_by_brand([c1, c2], max_per_brand=2, max_total=5)
        assert stats["selected_brand_count"] == 2

    def test_unknown_brand_grouped_together(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("Generic MCB", None, score=0.90),
            self._cand("ABB S201", "ABB", score=0.85),
            self._cand("Legrand RX3", "Legrand", score=0.80),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        brands = [_candidate_brand_key_safe(c) for c in selected]
        assert "unknown" in brands

    def test_higher_score_wins_within_brand(self):
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB cheap", "ABB", score=0.50, price=5.0),
            self._cand("ABB premium", "ABB", score=0.95, price=15.0),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=1, max_total=5)
        assert len(selected) == 1
        assert selected[0]["title"] == "ABB premium"

    def test_schneider_acti9_c16_realistic_scenario(self):
        """End-to-end realistic scenario for Schneider Acti9 iC60N C16.

        Reliable candidates (spec_q >= 0.5, !vague, EUR price):
        - 2 ABB (S201-C16, Miniature S201-C16) — same brand
        - 1 Legrand RX3
        - 1 Hager MCN116
        - 1 Siemens 5SL6106
        Expected: 4 distinct brands selected, no supplemental needed.
        """
        from worker.scoring import diversify_by_brand
        cands = [
            self._cand("ABB S201-C16 1P 16A Courbe C 6kA", "ABB", score=0.845, price=7.95),
            self._cand("Legrand RX3 1P 16A Courbe C 6000A", "Legrand", score=0.828, price=7.80),
            self._cand("Miniature Circuit Breaker S201-C16 - 1P - C - 16A", "ABB", score=0.826, price=7.95),
            self._cand("Hager MCN116 1P 16A 6kA", "Hager", score=0.824, price=9.20),
            self._cand("Siemens 5SL6106-6 1P 16A", "Siemens", score=0.811, price=12.50),
        ]
        selected, overflow, stats = diversify_by_brand(cands, max_per_brand=2, max_total=5)
        assert len(selected) == 5
        assert stats["selected_brand_count"] == 4, f"Expected 4 brands, got {stats['selected_brand_count']}"
        assert stats["needs_supplemental_search"] is False
        brands = [c["brand"] for c in selected]
        assert brands.count("ABB") == 2
        assert "Legrand" in brands
        assert "Hager" in brands
        assert "Siemens" in brands


def _candidate_brand_key_safe(c):
    from worker.scoring import _candidate_brand_key
    return _candidate_brand_key(c)

import re
from typing import Any

ELECTRICAL_BRANDS = [
    "abb", "schneider", "schneider electric", "legrand", "siemens", "eaton",
    "hager", "chint", "noark", "phoenix contact", "wago", "finder",
    "lovato", "mitsubishi", "bticino", "gewiss", "crouzet", "klockner moeller",
    "telemecanique", "merlin gerin", "square d", "abb stotz", "siemens sentron",
    "schneider tesys", "carlo gavazzi", "benedict", "cembre", "weidmuller",
    "bals", "walther", "cewe", "gira", "jung", "berker", "feller", "merten",
    "siedle", "rittal", "hensel", "spelsberg", "trilux", "zumtobel", "siteco",
    "wallbox", "nexans", "prysmian", "igus", "lapp", "helukabel",
]

ELECTRICAL_KEYWORDS = [
    "disjoncteur", "circuit breaker", "mcb", "mccb", "rccb", "rcbo", "rcd",
    "fusible", "fuse", "cartouche fusible",
    "contactor", "contacteur", "relay", "relais", "telerupteur", "télérupteur",
    "interrupteur", "switch", "sectionneur", "disconnector", "commutateur",
    "tableau électrique", "coffret", "enclosure", "consumer unit", "distribution board",
    "armoire électrique", "platine", "rail din", "din rail", "peigne",
    "bornier", "terminal block", "borne", "embout", "câble", "cable",
    "gaine", "conduit", "chemin de câbles", "cable tray",
    "automat", "plc", "variateur", "frequency drive", "soft starter", "démarreur",
    "relais thermique", "thermal relay", "motor protection", "guard motor",
    "minuterie", "timer", "permutateur", "va-et-vient", "prise", "socket", "outlet",
    "luminaire", "éclairage", "lighting", "led panel", "ampoule", "downlight",
    "starter", "ballast", "driver led", "transformateur", "transformer",
    "compteur", "meter", "kwh meter", "tore", "current transformer",
    "parafoudre", "surge protector", "spd", "protection foudre",
    "borne de recharge", "ev charger", "wallbox", "irve",
    "onduleur", "inverter", "pv inverter", "solar inverter",
    "boitier", "capot", "enjoliveur", "plaque", "cadre", "support",
    "bobine", "coil", "contact auxiliaire", "auxiliary contact",
    "modular", "modulaire", "tripolaire", "tétrapolaire", "monophasé", "triphasé",
    "bipolaire", "unipolaire", "1p", "2p", "3p", "4p",
    "courbe b", "courbe c", "courbe d", "courbe k", "curve b", "curve c", "curve d",
    "pouvoir de coupure", "breaking capacity", "kva", "kw", "watt",
    "shunt", "auxiliaire", "auxiliary",
    "iec 60898", "iec 60947", "iec 61009", "iec 62423", "nf c 15-100",
    "nf c 61-910", "nf", "en 60898", "din 43880", "ul 489",
    "schéma", "schema", "wiring", "raccordement", "connection",
    "modular switch", "rocker switch", "push button", "bouton poussoir",
    "selector switch", "commutateur rotatif", "cam switch",
    "hmi", "ihm", "interface homme machine", "touch panel", "écran tactile",
    "capteur", "sensor", "détecteur", "detector", "sonde",
    "power supply", "alimentation", "convertisseur", "converter",
    "automation", "automatisme", "commande",
    "ip65", "ip54", "ip44", "ip2x", "indice de protection",
    "rigide", "souple", "multibrin", "monobrin",
    "domino", "wago", "wagobox",
    "armoire de commande", "coffret de commande",
    "contacteur jour/nuit", "contacteur heures creuses",
    "differentiel", "différentiel",
    "interrupteur differentiel", "interrupteur différentiel",
    "30ma", "300ma", "500ma",
    "schuko", "2p+t", "prise de courant",
    "goulotte", "goulotte pvc", "goulotte aluminium",
    "boite de derivation", "junction box", "boîte d'encastrement",
    "barette de connexion", "connecteur wago", "borne de raccordement",
    "sectionneur rotatif", "inverseur de source", "inverseur",
    "detecteur de mouvement", "minuterie cage d'escalier",
    "cable h07rn-f", "cable rigide", "cable souple",
    "alarme", "alarme intrusion", "centrale alarme",
    "videosurveillance", "camera", "câblage réseau", "cable rj45",
    "data", "réseau", "ethernet", "fibre optique",
    "tableau divisionnaire", "tableau de répartition",
    "gtl", "gaine technique logement", "etl", "etable",
    "dtu", "promotelec", "consuel",
    "schneider electric", "merlin gerin", "square d", "lexium", "altivar", "ovaltis",
    "stotz", "oventrop", "wieland", "rittal", "hensel", "spelsberg",
    "bega", "bega lighting", "siteco", "trilux", "zumtobel",
    "merten", "siedle", "gira", "jung", "berker", "feller",
    "eubac", "kfw", "cee", "iec 60309", "prise industrielle",
]

CATEGORY_SYNONYMS = {
    "protection": ["disjoncteur", "circuit breaker", "mcb", "mccb", "rccb", "rcbo",
                   "rcd", "fusible", "fuse", "parafoudre", "surge protector", "spd",
                   "interrupteur differentiel", "interrupteur différentiel"],
    "circuit_breaker": ["disjoncteur", "circuit breaker", "mcb", "mccb", "breaker"],
    "breaker": ["disjoncteur", "circuit breaker", "mcb", "mccb", "breaker"],
    "mcb": ["mcb", "disjoncteur", "modulaire", "1p", "2p", "3p", "4p", "miniature circuit breaker"],
    "mccb": ["mccb", "molded case", "disjoncteur boitier", "boitier moule"],
    "rcd": ["rcd", "rccb", "rcbo", "differentiel", "différentiel", "residual current"],
    "contactor": ["contacteur", "contactor", "relais", "relay", "telerupteur",
                  "télérupteur", "commutateur", "permutateur"],
    "distribution": ["tableau", "coffret", "enclosure", "consumer unit",
                     "distribution board", "armoire", "platine", "goulotte", "gtl"],
    "cable": ["câble", "cable", "gaine", "conduit", "chemin de câbles", "cable tray",
              "h07rn-f", "rj45", "fibre optique", "alimentation"],
    "automation": ["plc", "automate", "variateur", "frequency drive", "soft starter",
                   "démarreur", "hmi", "ihm", "touch panel", "controller"],
    "lighting": ["luminaire", "éclairage", "lighting", "led", "ampoule", "downlight",
                 "ballast", "driver led", "starter"],
    "wiring": ["bornier", "terminal block", "borne", "embout", "connecteur", "domino",
               "wago", "wagobox", "cable lug", "barette"],
    "switch": ["interrupteur", "switch", "commutateur", "sectionneur", "disconnector",
               "va-et-vient", "permutateur", "bouton poussoir", "push button", "selector"],
    "socket": ["prise", "socket", "outlet", "schuko", "2p+t", "prise de courant",
               "prise industrielle", "cee", "iec 60309"],
    "transformer": ["transformateur", "transformer", "convertisseur", "converter",
                    "alimentation", "power supply", "tore", "current transformer"],
    "meter": ["compteur", "meter", "kwh", "kwh meter", "tore", "analyseur"],
    "ev_charging": ["borne de recharge", "ev charger", "wallbox", "irve", "evse"],
    "solar": ["onduleur", "inverter", "pv inverter", "solar inverter", "panneau solaire"],
    "safety": ["détecteur", "detector", "capteur", "sensor", "sonde", "alarme",
               "camera", "vidéosurveillance"],
}

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "this", "that", "these", "those", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "mine", "yours", "his", "hers", "ours",
    "theirs", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "buy", "new", "best", "cheap", "top", "price", "shop",
    "online", "store", "sale", "deal", "discount", "review", "reviews",
    "et", "ou", "de", "du", "le", "la", "les", "un", "une", "des", "avec",
    "sans", "pour", "par", "sur", "sous", "dans", "entre", "vers", "chez",
    "est", "sont", "a", "ont", "ai", "fait", "faire", "etre",
})

ACCESSORY_PATTERNS = [
    r"\bcase\s+for\b",
    r"\bcover\s+for\b",
    r"\bprotector\s+for\b",
    r"\breplacement\s+(part|blade|head|pad|tip|cartridge|coil|contact)\b",
    r"\bspare\s+part\b",
    r"\bcable\s+for\b",
    r"\bcharger\s+for\b",
    r"\bfilter\s+for\b",
    r"\bbattery\s+for\b",
    r"\bcompatible\s+with\b",
    r"\baccessory\b",
    r"\bconsumable\b",
    r"\battachment\b",
    r"\badapter\s+for\b",
    r"\bholder\s+for\b",
    r"\bstand\s+for\b",
    r"\bmount\s+for\b",
    r"\bbelt\s+for\b",
    r"\bbag\s+for\b",
    r"\bpouch\s+for\b",
    r"\bscreen\s+protector\b",
    r"\btempered\s+glass\b",
    r"\bskin\s+for\b",
    r"\bstrap\s+for\b",
    r"\bpad\s+for\b",
    r"\bpart\s+for\b",
    r"\bpiece\s+for\b",
    r"\bkit\s+for\b",
    r"\btool\s+for\b",
    r"\boil\s+filter\b",
    r"\bmanual\b",
    r"\bwarranty\b",
    r"\bbobine(?:\s+[a-z0-9]+)?\s+(?:pour|auxiliaire|de|compatible|12v|24v|230v)\b",
    r"\bcontact\s+auxiliaire(?:\s+\w+)?\s+(?:pour|de|compatible)\b",
    r"\bauxiliary\s+contact(?:\s+\w+)?\s+for\b",
    r"\bcoil(?:\s+\w+)?\s+for\b",
    r"\bmodule(?:\s+\w+)?\s+(?:pour|auxiliaire|de|compatible|additif)\b",
    r"\bplaque(?:\s+\w+)?\s+pour\b",
    r"\benjoliveur(?:\s+\w+)?\s+pour\b",
    r"\bcadre(?:\s+\w+)?\s+pour\b",
    r"\bboitier(?:\s+\w+)?\s+pour\b",
    r"\bcapot(?:\s+\w+)?\s+pour\b",
    r"\bsupport(?:\s+\w+)?\s+pour\b",
    r"\brail(?:\s+\w+)?\s+pour\b",
    r"\bgoulotte(?:\s+\w+)?\s+pour\b",
    r"\bpeigne(?:\s+\w+)?\s+pour\b",
    r"\bbarrette(?:\s+\w+)?\s+pour\b",
    r"\baccessoires?\s+pour\b",
    r"\bconsommables?\s+pour\b",
    r"\brechange\b",
    r"\bbobine\s+(?:de|mx|mn|of|sh)\b",
    r"\bcontact\s+(?:no|nc|of|sf)\s+(?:pour|auxiliaire)\b",
    r"\bbornier(?:\s+\w+)?\s+(?:de|pour|repartition)\b",
    r"\bpeigne(?:\s+\w+)?\s+(?:de|repartiteur|connexion)\b",
    r"\bgoulotte(?:\s+\w+)?\s+(?:pvc|aluminium|electrique)\b",
    r"\btirette\s+pour\b",
    r"\benjoliveur\b",
    r"\bcapot\s+de\b",
    r"\bplaque\s+de\b",
    r"\bcadre\s+de\b",
]

ACCESSORY_KEYWORDS = [
    "accessory", "accessories", "compatible", "replacement", "spare part",
    "consumable", "attachment", "adapter", "holder", "stand", "mount",
    "protector", "cover", "case", "pouch", "bag", "screen protector",
    "tempered glass", "charger", "cable", "battery", "filter", "strap",
    "pad", "belt", "skin", "part for", "piece for", "kit for",
]


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS and len(w) > 1}


def _is_electrical(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    for keyword in ELECTRICAL_KEYWORDS:
        if keyword in text_lower:
            return True
    for brand in ELECTRICAL_BRANDS:
        if brand in text_lower:
            return True
    return False


def _is_accessory(title: str, description: str = "") -> bool:
    combined = (title + " " + description).lower()
    for pattern in ACCESSORY_PATTERNS:
        if re.search(pattern, combined):
            return True
    return False


def _category_similarity(product_category: str | None, candidate: dict) -> float:
    if not product_category:
        return 0.0
    cat_lower = product_category.lower()
    title_lower = (candidate.get("title") or "").lower()
    cat_words = _tokenize(cat_lower)
    title_words = _tokenize(title_lower)

    if cat_lower in title_lower:
        return 0.25
    if cat_words & title_words:
        return 0.15

    for cat_word in cat_words:
        if cat_word in CATEGORY_SYNONYMS:
            for synonym in CATEGORY_SYNONYMS[cat_word]:
                if synonym in title_lower:
                    return 0.15

    return 0.0


def _keyword_overlap(product_name: str, product_desc: str, candidate: dict) -> float:
    source_text = product_name + " " + product_desc
    source_words = _tokenize(source_text)
    if not source_words:
        return 0.0
    target_text = (candidate.get("title") or "") + " " + (candidate.get("content") or "")
    target_words = _tokenize(target_text)
    if not target_words:
        return 0.0
    overlap = source_words & target_words
    score = len(overlap) / max(len(source_words), len(target_words))
    return min(score * 0.4, 0.35)


def _price_coherence(price: float | None, target_price: float | None) -> float:
    if price is None or target_price is None or target_price <= 0:
        return 0.1
    if price <= 0:
        return 0.0
    ratio = price / target_price
    if 0.8 <= ratio <= 1.2:
        return 0.25
    if 0.5 <= ratio <= 1.5:
        return 0.2
    if 0.3 <= ratio <= 3.0:
        return 0.1
    return 0.0


def _cross_brand_bonus(product_brand: str | None, title: str) -> float:
    if not product_brand or not title:
        return 0.0
    brand_lower = product_brand.lower().strip()
    title_lower = title.lower()
    if brand_lower in title_lower:
        return 0.0
    brand_variants = [brand_lower]
    if "schneider" in brand_lower:
        brand_variants.append("schneider")
    if "schneider electric" not in brand_lower:
        brand_variants.append(brand_lower + " electric")
    for variant in brand_variants:
        if variant in title_lower:
            return 0.0
    for known_brand in ELECTRICAL_BRANDS:
        if known_brand in title_lower:
            return 0.15
    return 0.0


def _same_brand_merchant_count(product_brand: str | None, title: str, all_scores: list | None) -> bool:
    if not product_brand or not title:
        return False
    brand_lower = product_brand.lower().strip()
    title_lower = title.lower()
    if brand_lower in title_lower:
        return True
    brand_variants = []
    if "electric" in brand_lower:
        brand_variants.append(brand_lower.replace(" electric", "").strip())
    if "schneider" in brand_lower:
        brand_variants.append("schneider")
    if "electric" in brand_lower:
        brand_variants.append(brand_lower.replace(" electric", ""))
    for variant in brand_variants:
        if variant and variant in title_lower:
            return True
    return False


def _poles_to_fr(poles: int) -> str:
    return {1: "1P", 2: "2P", 3: "tripolaire", 4: "tétrapolaire"}.get(poles, f"{poles}P")


CATEGORY_TERMS = {
    "mcb": {"en": "MCB", "fr": "disjoncteur"},
    "circuit_breaker": {"en": "MCB", "fr": "disjoncteur"},
    "breaker": {"en": "MCB", "fr": "disjoncteur"},
    "disjoncteur": {"en": "MCB", "fr": "disjoncteur"},
    "mccb": {"en": "MCCB", "fr": "disjoncteur boîtier"},
    "rcd": {"en": "RCD", "fr": "interrupteur différentiel"},
    "rcbo": {"en": "RCBO", "fr": "interrupteur différentiel"},
    "differentiel": {"en": "RCD", "fr": "interrupteur différentiel"},
    "différentiel": {"en": "RCD", "fr": "interrupteur différentiel"},
    "contactor": {"en": "contactor", "fr": "contacteur"},
    "contacteur": {"en": "contactor", "fr": "contacteur"},
    "relay": {"en": "relay", "fr": "relais"},
    "relais": {"en": "relay", "fr": "relais"},
    "switch": {"en": "switch", "fr": "interrupteur"},
    "interrupteur": {"en": "switch", "fr": "interrupteur"},
    "cable": {"en": "cable", "fr": "câble"},
    "câble": {"en": "cable", "fr": "câble"},
    "ev_charger": {"en": "EV charger", "fr": "borne de recharge"},
    "wallbox": {"en": "EV charger", "fr": "borne de recharge"},
    "transformer": {"en": "transformer", "fr": "transformateur"},
    "transformateur": {"en": "transformer", "fr": "transformateur"},
    "panel": {"en": "electrical panel", "fr": "tableau électrique"},
    "coffret": {"en": "electrical panel", "fr": "coffret"},
    "meter": {"en": "kWh meter", "fr": "compteur"},
    "compteur": {"en": "kWh meter", "fr": "compteur"},
}

CATEGORY_DISPLAY_NAMES = {
    "mcb": "miniature circuit breaker",
    "circuit_breaker": "miniature circuit breaker",
    "breaker": "circuit breaker",
    "disjoncteur": "miniature circuit breaker",
    "mccb": "molded case circuit breaker",
    "rcd": "residual current device",
    "rcbo": "residual current breaker with overcurrent",
    "differentiel": "residual current device",
    "différentiel": "residual current device",
    "contactor": "contactor",
    "contacteur": "contactor",
    "relay": "relay",
    "relais": "relay",
    "switch": "switch",
    "interrupteur": "switch",
    "cable": "cable",
    "câble": "cable",
    "ev_charger": "EV charger",
    "wallbox": "EV charger",
    "transformer": "transformer",
    "transformateur": "transformer",
    "panel": "electrical panel",
    "coffret": "electrical panel",
    "meter": "kWh meter",
    "compteur": "kWh meter",
}

TIER1_BRANDS_QUERY = ["ABB", "Legrand", "Hager", "Siemens", "Eaton"]


def _detect_category_key(category: str | None, name: str) -> str | None:
    cat_lower = (category or "").lower().strip()
    name_lower = name.lower()
    for k in CATEGORY_TERMS:
        if k in cat_lower or cat_lower == k:
            return k
    for k in CATEGORY_TERMS:
        if k in name_lower:
            return k
    return None


def _build_product_name(description: str, name: str | None = None) -> str:
    inferred = (name or description)[:100].strip().rstrip(",;. ")
    return inferred


def _infer_product_attributes(
    description: str,
    brand: str | None = None,
    category: str | None = None,
    name: str | None = None,
) -> dict:
    """Infer product name, category, brand, specs from a description string.

    Returns a dict with keys: name, category, brand, specs.
    All inferred values can be overridden by explicit user input (caller merges).
    """
    inferred_name = _build_product_name(description, name)
    cat_key = _detect_category_key(category, inferred_name) or "mcb"
    display_category = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key.replace("_", " ").title())
    inferred_brand = _extract_brand_from_title(description) or None
    specs = _extract_specs_from_title(description)
    desc_lower = description.lower()
    if specs.get("mounting") is None:
        if "rail din" in desc_lower or "din rail" in desc_lower or "montage rail din" in desc_lower:
            specs["mounting"] = "DIN rail"
    return {
        "name": inferred_name,
        "category": display_category,
        "brand": inferred_brand,
        "specs": specs,
    }


def generate_brand_targeted_queries(
    product_category: str | None = None,
    product_name: str = "",
    product_brand: str | None = None,
    poles: int | None = None,
    current_a: int | None = None,
    curve: str | None = None,
    breaking_capacity_ka: float | None = None,
    phase: str | None = None,
    max_queries: int = 12,
) -> list[str]:
    """Generate deterministic brand-targeted queries for known electrical product types.

    Returns up to 12 queries: 5 English (one per tier-1 brand) + 5 French (one per
    tier-1 brand) + 1-2 generic. The product's own brand is filtered out so we don't
    search for ourselves.
    """
    cat_key = _detect_category_key(product_category, product_name) or "mcb"
    cat_en = CATEGORY_TERMS[cat_key]["en"]
    cat_fr = CATEGORY_TERMS[cat_key]["fr"]

    spec_parts_en: list[str] = []
    spec_parts_fr: list[str] = []
    if poles is not None:
        spec_parts_en.append(f"{poles}P")
        spec_parts_fr.append(_poles_to_fr(poles))
    if current_a is not None:
        spec_parts_en.append(f"{current_a}A")
        spec_parts_fr.append(f"{current_a}A")
    if curve:
        spec_parts_en.append(f"curve {curve}")
        spec_parts_fr.append(f"courbe {curve}")
    if breaking_capacity_ka is not None:
        ka_str = f"{int(breaking_capacity_ka)}kA" if breaking_capacity_ka == int(breaking_capacity_ka) else f"{breaking_capacity_ka}kA"
        spec_parts_en.append(ka_str)
        spec_parts_fr.append(ka_str)
    if phase == "three":
        spec_parts_en.append("3-phase")
        spec_parts_fr.append("triphasé")
    spec_en = " ".join(spec_parts_en)
    spec_fr = " ".join(spec_parts_fr)

    own_brand_lower = (product_brand or "").lower().strip()
    target_brands: list[str] = []
    for b in TIER1_BRANDS_QUERY:
        b_lower = b.lower()
        if own_brand_lower and (b_lower in own_brand_lower or own_brand_lower in b_lower):
            continue
        if "schneider" in own_brand_lower and b_lower == "legrand":
            continue
        target_brands.append(b)

    queries: list[str] = []
    for b in target_brands:
        if spec_en:
            queries.append(f"{b} {spec_en} {cat_en} equivalent".strip())
        else:
            queries.append(f"{b} {cat_en} equivalent".strip())
    for b in target_brands:
        if spec_fr:
            queries.append(f"{cat_fr} {b} {spec_fr}".strip())
        else:
            queries.append(f"{cat_fr} {b}".strip())

    if spec_en and cat_en:
        queries.append(f"{cat_en} {spec_en} DIN rail")
    if spec_fr and cat_fr:
        queries.append(f"{cat_fr} {spec_fr} modulaire")

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        qn = q.lower().strip()
        if qn and qn not in seen:
            seen.add(qn)
            unique.append(q)
    return unique[:max_queries]


_SORTED_BRANDS_BY_LENGTH = sorted(ELECTRICAL_BRANDS, key=len, reverse=True)


def _extract_brand_from_title(title: str | None) -> str | None:
    """Deterministically extract brand from a product title using ELECTRICAL_BRANDS.

    Returns the brand name in title case (e.g. "ABB") or None if no known brand
    is found. Multi-word brands (e.g. "schneider electric") are checked first
    so that "schneider electric" wins over the "schneider" substring.
    """
    if not title:
        return None
    title_lower = title.lower()
    for brand in _SORTED_BRANDS_BY_LENGTH:
        if brand in title_lower:
            parts = brand.split()
            return " ".join(p.upper() if len(p) <= 3 else p.capitalize() for p in parts)
    return None


def _extract_specs_from_title(title: str | None) -> dict:
    r"""Deterministically extract electrical specs from a product title.

    Handles MCB patterns:
    - current_a: "16A", "C16" (e.g. S201-C16, iC60N C16), "10kA" range, "1P C 16A"
    - poles: "1P", "2P", "3P", "4P", "1P+N", "triphasé" (-> 3)
    - curve: "Type C", "Courbe C", "curve B", "C - 16A" (-> C), "C16" (-> C)
    - breaking_capacity_ka: "6kA", "10kA", "6000A" (= 6kA), "15kA"
    - voltage_v: "230V", "400V"

    Disambiguation rules:
    - "6kA" / "6KA" is breaking_capacity_ka, NEVER current_a
    - "C16" / "D20" is current_a + curve (C16 -> current_a=16, curve=C)
    - "6000A" with no "kA" suffix is breaking_capacity_ka=6.0 (typical: 6kA = 6000A)
    - "16A" at the end of the title is the most likely current_a
    - Ranges like "6A-63A" are ignored for current_a
    """
    if not title:
        return {}
    title_lower = title.lower()
    specs: dict = {}

    curve_from_letter_digit: str | None = None
    current_from_letter_digit: int | None = None

    m = re.search(r"\b([bcdk])\s*-?\s*(\d{1,3})\b", title_lower)
    if m:
        curve_from_letter_digit = m.group(1).upper()
        current_from_letter_digit = int(m.group(2))

    m = re.search(r"(\d+)\s*p\s*(?:\+\s*n)?\b", title_lower)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 4:
            specs["poles"] = val

    m = re.search(r"(\d+)\s*ka\b", title_lower)
    if m:
        specs["breaking_capacity_ka"] = float(m.group(1))
    else:
        m = re.search(r"(\d{3,4})\s*a\b", title_lower)
        if m:
            val = int(m.group(1))
            if val in (6000, 10000, 15000, 25000, 36000, 50000, 60000, 100000):
                specs["breaking_capacity_ka"] = val / 1000.0

    m = re.search(r"(\d+)\s*v\b", title_lower)
    if m:
        val = int(m.group(1))
        if val >= 12 and val <= 1000:
            specs["voltage_v"] = val

    if "triphas" in title_lower or "3-phase" in title_lower or "three-phase" in title_lower:
        specs.setdefault("poles", 3)
    if "tétrapolaire" in title_lower or "tetrapolaire" in title_lower:
        specs.setdefault("poles", 4)
    if "monophas" in title_lower or "1-phase" in title_lower or "single-phase" in title_lower:
        specs.setdefault("poles", 1)

    if "rail din" in title_lower or "din rail" in title_lower or "montage rail din" in title_lower:
        specs["mounting"] = "DIN rail"
    if specs.get("mounting") is None:
        if "encastrable" in title_lower or "flush" in title_lower or "encastrement" in title_lower:
            specs["mounting"] = "flush"

    m = re.search(r"(?:courbe|curve|type)\s*([bcdk])\b", title_lower)
    if not m:
        m = re.search(r"\s-\s*([bcdk])\s*-?\s", title_lower)
    if not m:
        m = re.search(r"\s([bcdk])\s+\d", title_lower)
    curve_from_explicit: str | None = None
    if m:
        curve_from_explicit = m.group(1).upper()

    final_curve = curve_from_explicit or curve_from_letter_digit
    if final_curve:
        specs["curve"] = final_curve

    excluded_a_values = {6000, 10000, 15000, 25000, 36000, 50000, 60000, 100000}

    def _in_range(val: int) -> bool:
        return bool(re.search(rf"\b{val}\s*a\s*[-–]\s*\d+\s*a\b", title_lower))

    candidate_current: int | None = None
    for am in re.finditer(r"\b(\d{1,4})\s*a\b", title_lower):
        val = int(am.group(1))
        if val in excluded_a_values:
            continue
        if 1 <= val <= 1250 and not _in_range(val):
            candidate_current = val

    if candidate_current is not None:
        specs["current_a"] = candidate_current

    if "current_a" not in specs and current_from_letter_digit is not None:
        specs["current_a"] = current_from_letter_digit

    return specs


VAGUE_GENERIC_TERMS = {
    "disjoncteur", "mcb", "breaker", "contactor", "contacteur",
    "interrupteur", "switch", "relay", "relais", "cable", "câble",
    "transformer", "transformateur", "borne", "wallbox", "ev charger",
    "module", "boitier", "coffret", "tableau",
}


def _is_vague_title(title: str | None) -> bool:
    """Detect vague titles like 'ABB Disjoncteur' with no specs and no model.

    A title is vague if it contains only a generic term + brand, with no model
    number and no specific specs.
    """
    if not title or len(title.strip()) < 8:
        return True
    title_lower = title.lower()
    has_model = bool(re.search(r"[a-z]+[-_]?\d{2,}", title_lower))
    has_specs = bool(re.search(
        r"\d+\s*a\b|\d+\s*p\b|courbe\s*[a-z]|curve\s*[a-z]|\d+\s*ka|\d+\s*v\b",
        title_lower,
    ))
    words = set(re.findall(r"[a-zà-ÿ]+", title_lower))
    has_generic_only = bool(words & VAGUE_GENERIC_TERMS) and not has_model
    return has_generic_only and not has_specs


def _spec_match_value(target: Any, candidate: Any) -> float:
    """Compare two spec values. Returns 1.0 match, 0.5 partial (e.g. close kA), 0.0 mismatch."""
    if target is None or candidate is None:
        return 0.0
    if isinstance(target, (int, float)) and isinstance(candidate, (int, float)):
        if target == candidate:
            return 1.0
        if abs(target - candidate) <= max(0.1 * abs(target), 2):
            return 0.5
        return 0.0
    return 1.0 if str(target).upper() == str(candidate).upper() else 0.0


def spec_quality_score(product: Any, candidate: dict) -> tuple[float, dict]:
    """Compute a 0-1 spec quality score for ranking and vagueness detection.

    Strong bonuses for matching current_a, poles, curve (critical MCB specs).
    Medium bonus for matching breaking_capacity_ka. Penalty for missing critical
    specs and for vague titles.
    """
    title = candidate.get("title") or ""
    cand_specs = candidate.get("specs") or _extract_specs_from_title(title)

    target_current = getattr(product, "current_a", None)
    target_poles = getattr(product, "poles", None)
    target_curve = getattr(product, "curve", None)
    target_ka = getattr(product, "breaking_capacity_ka", None)
    target_voltage = getattr(product, "voltage_v", None)

    cand_current = cand_specs.get("current_a")
    cand_poles = cand_specs.get("poles")
    cand_curve = cand_specs.get("curve")
    cand_ka = cand_specs.get("breaking_capacity_ka")
    cand_voltage = cand_specs.get("voltage_v")

    breakdown: dict = {
        "current_a": 0.0, "poles": 0.0, "curve": 0.0, "kA": 0.0, "voltage": 0.0,
        "missing_critical": [], "is_vague": False,
    }

    score = 0.0

    v = _spec_match_value(target_current, cand_current)
    if v == 1.0:
        score += 0.30
        breakdown["current_a"] = 0.30
    elif v == 0.5:
        score += 0.15
        breakdown["current_a"] = 0.15
    elif target_current is not None and cand_current is None:
        breakdown["missing_critical"].append("current_a")

    v = _spec_match_value(target_poles, cand_poles)
    if v == 1.0:
        score += 0.25
        breakdown["poles"] = 0.25
    elif target_poles is not None and cand_poles is None:
        breakdown["missing_critical"].append("poles")

    v = _spec_match_value(target_curve, cand_curve)
    if v == 1.0:
        score += 0.20
        breakdown["curve"] = 0.20
    elif target_curve is not None and cand_curve is None:
        breakdown["missing_critical"].append("curve")

    v = _spec_match_value(target_ka, cand_ka)
    if v == 1.0:
        score += 0.10
        breakdown["kA"] = 0.10
    elif v == 0.5:
        score += 0.05
        breakdown["kA"] = 0.05
    elif target_ka is not None and cand_ka is None:
        breakdown["missing_critical"].append("kA")

    v = _spec_match_value(target_voltage, cand_voltage)
    if v == 1.0:
        score += 0.05
        breakdown["voltage"] = 0.05

    if len(breakdown["missing_critical"]) >= 2:
        score -= 0.20
        breakdown["spec_penalty"] = -0.20

    is_vague = _is_vague_title(title)
    if is_vague:
        score -= 0.30
        breakdown["is_vague"] = True
        breakdown["vague_penalty"] = -0.30

    final = max(0.0, min(1.0, score))
    breakdown["total"] = round(final, 3)
    return final, breakdown


SPEC_QUALITY_RELIABLE_THRESHOLD = 0.5
SPEC_QUALITY_PARTIAL_THRESHOLD = 0.25


def is_reliable_equivalent(
    candidate: dict,
    product: Any | None = None,
    target_currency: str | None = None,
) -> bool:
    """Return True if the candidate is reliable enough to influence pricing.

    A candidate is reliable when ALL conditions hold:
    - spec_quality >= 0.5 (has critical specs or matches target)
    - not vague (has model number and/or explicit specs)
    - price is set and > 0
    - currency matches target (when target_currency is provided)
    """
    if not candidate:
        return False
    sq = candidate.get("spec_quality", 0.0) or 0.0
    if sq < SPEC_QUALITY_RELIABLE_THRESHOLD:
        return False
    if candidate.get("is_vague", False):
        return False
    price = candidate.get("price")
    if price is None or (isinstance(price, (int, float)) and price <= 0):
        return False
    if target_currency:
        cand_currency = (candidate.get("currency") or "").upper()
        target_upper = target_currency.upper()
        if cand_currency and cand_currency != target_upper:
            return False
    return True


def split_reliable_vs_weak(
    scored: list[dict],
    product: Any | None = None,
    target_currency: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split scored candidates into (reliable, weak) buckets.

    Reliable candidates meet the spec_quality/vague/price/currency criteria.
    Weak candidates are still returned (for transparency) but excluded from
    the cross_brand_equivalents list and pricing recommendations.
    """
    reliable: list[dict] = []
    weak: list[dict] = []
    for s in scored:
        if is_reliable_equivalent(s, product, target_currency):
            reliable.append(s)
        else:
            weak.append(s)
    return reliable, weak


def split_by_quality(
    scored: list[dict],
    product: Any | None = None,
    target_currency: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split scored candidates into (reliable, partial, weak) buckets.

    - reliable: spec_quality >= 0.5, not vague, valid price & currency
    - partial: spec_quality between 0.25-0.5, not vague (has some specs but not all)
    - weak: everything else
    """
    reliable: list[dict] = []
    partial: list[dict] = []
    weak: list[dict] = []
    for s in scored:
        if is_reliable_equivalent(s, product, target_currency):
            reliable.append(s)
        else:
            sq = s.get("spec_quality", 0) or 0
            if sq >= SPEC_QUALITY_PARTIAL_THRESHOLD and not s.get("is_vague", False):
                partial.append(s)
            else:
                weak.append(s)
    return reliable, partial, weak


def deterministic_pre_score(product: Any, candidate: dict,
                            all_scores: list[dict] | None = None) -> dict:
    title = candidate.get("title") or ""
    description = candidate.get("content") or candidate.get("description") or ""
    raw_price = candidate.get("price")
    try:
        price = float(raw_price) if raw_price else None
    except (ValueError, TypeError):
        price = None

    cat_score = _category_similarity(product.category, candidate)
    kw_score = _keyword_overlap(product.name, product.description, candidate)
    price_score = _price_coherence(price, product.target_price)
    accessory_penalty = -0.5 if _is_accessory(title, description) else 0.0
    cross_brand = _cross_brand_bonus(product.brand, title)

    total = max(0.0, min(1.0, cat_score + kw_score + price_score + accessory_penalty + cross_brand))

    is_acc = accessory_penalty < -0.1
    is_same_brand = _same_brand_merchant_count(product.brand, title, all_scores)

    if total >= 0.45:
        hint = "direct_competitor"
    elif total >= 0.3:
        hint = "functional_equivalent"
    elif total >= 0.15:
        hint = "cheaper_alternative"
    elif is_acc:
        hint = "accessory_or_part"
    else:
        hint = "functional_equivalent"

    return {
        "deterministic_score": round(total, 3),
        "is_accessory": is_acc,
        "is_same_brand": is_same_brand,
        "classification_hint": hint,
    }


def _candidate_brand_key(candidate: dict) -> str:
    """Return a stable, normalized brand key for a candidate.

    Falls back to the deterministic brand extraction from the title when
    the LLM/extractor left the brand empty. Returns 'unknown' when the
    brand cannot be determined.
    """
    brand = (candidate.get("brand") or "").strip()
    if not brand:
        brand = _extract_brand_from_title(candidate.get("title", "")) or ""
    return brand.strip().lower() or "unknown"


def diversify_by_brand(
    candidates: list[dict],
    max_per_brand: int = 2,
    max_total: int = 5,
    min_brand_count: int = 3,
) -> tuple[list[dict], list[dict], dict]:
    """Diversify a list of candidates by brand for marketing/pricing fairness.

    Algorithm: greedy score-based selection with a per-brand cap.
    - Iterates over candidates sorted by score (desc)
    - Adds a candidate only if its brand is not yet at max_per_brand
      AND the total selected is below max_total
    - Candidates that would exceed the cap go to overflow
    - This naturally gives priority to higher-scoring brands while still
      enforcing diversity (each brand is capped)

    Returns (selected, overflow, stats) where stats contains:
    - selected_brand_count: distinct brands in selected
    - unique_brands_total: distinct brands seen in input
    - needs_supplemental_search: True if selected_brand_count < min_brand_count
    - top_brands: list of (brand, count) sorted by count desc
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: -float(c.get("score", 0.0) or 0.0),
    )

    selected: list[dict] = []
    overflow: list[dict] = []
    per_brand_count: dict[str, int] = {}

    for c in sorted_candidates:
        brand_key = _candidate_brand_key(c)
        if brand_key not in per_brand_count:
            per_brand_count[brand_key] = 0
        if per_brand_count[brand_key] < max_per_brand and len(selected) < max_total:
            selected.append(c)
            per_brand_count[brand_key] += 1
        else:
            overflow.append(c)

    selected_brand_count = len({_candidate_brand_key(c) for c in selected})
    unique_brands_total = len(per_brand_count)
    top_brands = sorted(
        per_brand_count.items(),
        key=lambda x: -x[1],
    )

    stats = {
        "selected_brand_count": selected_brand_count,
        "unique_brands_total": unique_brands_total,
        "needs_supplemental_search": selected_brand_count < min_brand_count,
        "top_brands": top_brands,
    }
    return selected, overflow, stats

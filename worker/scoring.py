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

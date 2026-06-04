import re
from typing import Any

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
})

ACCESSORY_PATTERNS = [
    r"\bcase\s+for\b",
    r"\bcover\s+for\b",
    r"\bprotector\s+for\b",
    r"\breplacement\s+(part|blade|head|pad|tip|cartridge)\b",
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


def deterministic_pre_score(product: Any, candidate: dict) -> dict:
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

    total = max(0.0, min(1.0, cat_score + kw_score + price_score + accessory_penalty))

    is_acc = accessory_penalty < -0.1

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
        "classification_hint": hint,
    }

"""
tools/parse_customer_query.py
------------------------------
Student 1 – Custom Tool
Tool: parse_customer_query

Advanced NLP-driven intent extraction from raw customer queries.
Implements a multi-stage analysis pipeline:
  1. Action Intent Classification  – rule-based keyword matching
  2. Product Entity Extraction     – regex + stopword filtering
  3. Quantity Detection            – numeric pattern extraction
  4. Sentiment Analysis            – polarity scoring (positive/neutral/negative)
  5. Urgency Detection             – time-sensitive keyword flagging
  6. Multi-Item Parsing            – "and" / comma-separated product lists
  7. Contact Entity Extraction     – emails, phone numbers, order IDs
  8. Category Inference            – maps product names to likely categories
  9. Confidence Scoring            – weighted composite confidence metric

Uses only rule-based NLP so it works fully offline without any API calls.

Example
-------
    >>> from tools.parse_customer_query import parse_customer_query
    >>> parse_customer_query("I want to buy 2 Wireless Headphones Pro")
    {'action': 'buy', 'product_name': 'Wireless Headphones Pro', 'quantity': 2,
     'raw_query': '...', 'confidence': 'high', 'sentiment': 'positive', ...}
"""

import re
import math
from datetime import datetime, timezone
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ACTION INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
# ORDER MATTERS – checked top to bottom, first match wins.
_ACTION_KEYWORDS: dict[str, list[str]] = {
    "return":   ["return", "refund", "send back", "exchange", "swap", "replace"],
    "cancel":   ["cancel", "remove", "delete", "stop", "revoke", "undo"],
    "track":    ["track", "where is", "status", "shipping", "delivery",
                 "when will", "estimated", "eta", "locate"],
    "complain": ["complain", "complaint", "problem", "issue", "broken",
                 "damaged", "wrong", "defective", "faulty", "terrible",
                 "horrible", "worst", "disappointed"],
    "enquire":  ["what is", "do you have", "available", "how much", "price",
                 "cost", "details", "info", "information", "tell me about",
                 "describe", "specification", "specs"],
    "buy":      ["buy", "purchase", "order", "get", "want", "need", "add",
                 "checkout", "grab", "pick up"],
}

# ═══════════════════════════════════════════════════════════════════════════════
#  2. SENTIMENT LEXICON
# ═══════════════════════════════════════════════════════════════════════════════
_POSITIVE_WORDS = {
    "love", "great", "awesome", "excellent", "amazing", "fantastic", "happy",
    "perfect", "wonderful", "pleased", "thanks", "thank", "good", "best",
    "excited", "glad", "recommend", "favourite", "favorite", "impressed",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "worst", "hate", "angry", "frustrated",
    "disappointed", "broken", "damaged", "defective", "faulty", "poor",
    "awful", "useless", "rubbish", "pathetic", "unacceptable", "annoyed",
    "furious", "upset", "wrong", "never", "complaint", "complain",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  3. URGENCY KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════════
_URGENCY_KEYWORDS = [
    "asap", "urgent", "urgently", "rush", "immediately", "right now",
    "right away", "as soon as possible", "hurry", "emergency", "critical",
    "fast", "quickly", "today", "tonight", "express", "priority",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  4. PRODUCT CATEGORY HINTS
# ═══════════════════════════════════════════════════════════════════════════════
_CATEGORY_HINTS: dict[str, list[str]] = {
    "Electronics":  ["headphone", "keyboard", "mouse", "webcam", "monitor",
                     "speaker", "earphone", "laptop", "tablet", "phone",
                     "camera", "microphone", "charger"],
    "Accessories":  ["cable", "stand", "hub", "adapter", "dongle", "case",
                     "sleeve", "holder", "dock", "hdmi", "usb", "mount"],
    "Home Office":  ["lamp", "desk", "chair", "organizer", "shelf",
                     "whiteboard", "clock", "fan", "light"],
}

# Stopwords for product name cleaning
_FILLER_PATTERN = re.compile(
    r"\b(i|me|my|please|a|an|the|some|for|to|would|like|want|need|can|could|"
    r"just|really|also|do|does|it|is|are|was|were|be|being|been|have|has|had|"
    r"with|this|that|these|those|very|so|much|many|more|and|or|but|if|in|on|at|"
    r"of|from|by|about)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def parse_customer_query(raw_query: str) -> dict[str, Any]:
    """
    Parse a raw customer query through a multi-stage NLP pipeline.

    Pipeline Stages
    ---------------
    1. **Action Classification** — Match intent keywords to determine the
       customer's primary action (buy / return / track / cancel / complain /
       enquire / unknown).
    2. **Entity Extraction** — Pull out contact details (email, phone),
       order references, and other structured entities.
    3. **Product Extraction** — Clean the query to isolate the product name,
       stripping action verbs, quantities, and filler words.
    4. **Quantity Detection** — Find positive integers; cap at 999 to prevent
       unreasonable orders.
    5. **Multi-Item Detection** — Identify queries requesting more than one
       distinct product (e.g. "Buy 2 headphones and 3 cables").
    6. **Sentiment Analysis** — Score polarity using a lexicon-based approach
       and classify as positive / neutral / negative.
    7. **Urgency Detection** — Flag time-sensitive requests.
    8. **Category Inference** — Guess the likely product category based on
       keyword matching against known categories.
    9. **Confidence Scoring** — Compute a weighted composite score from
       action certainty, product clarity, and quantity presence.

    Parameters
    ----------
    raw_query : str
        Free-text customer input, e.g. "I want to buy 3 Mechanical Keyboards".

    Returns
    -------
    dict[str, Any]
        Comprehensive intent payload with the following keys:

        - action           (str)   : Primary intent classification
        - product_name     (str)   : Extracted product name
        - quantity         (int)   : Units requested (default 1)
        - raw_query        (str)   : Original input for audit trail
        - confidence       (str)   : 'high' / 'medium' / 'low'
        - confidence_score (float) : Numeric confidence (0.0 – 1.0)
        - sentiment        (str)   : 'positive' / 'neutral' / 'negative'
        - sentiment_score  (float) : Polarity score (-1.0 to +1.0)
        - is_urgent        (bool)  : Whether urgency was detected
        - urgency_keywords (list)  : Matched urgency terms
        - detected_entities(dict)  : Extracted emails, phones, order IDs
        - multi_item       (bool)  : Whether multiple products were detected
        - items            (list)  : Parsed items when multi_item is True
        - inferred_category(str)   : Best-guess product category
        - word_count       (int)   : Query length metric
        - processed_at     (str)   : ISO-8601 timestamp

    Raises
    ------
    ValueError
        If raw_query is empty or not a string.
    """
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError("raw_query must be a non-empty string.")

    query_lower = raw_query.lower().strip()
    words = query_lower.split()

    # ── Stage 1: Action Intent Classification ─────────────────────────────
    detected_action: str = "unknown"
    matched_keyword: str = ""
    for action, keywords in _ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                detected_action = action
                matched_keyword = kw
                break
        if detected_action != "unknown":
            break

    # ── Stage 2: Entity Extraction ────────────────────────────────────────
    entities: dict[str, Any] = {
        "emails":    re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_query),
        "phones":    re.findall(r"\+?[\d\s\-()]{7,15}", raw_query),
        "order_ids": re.findall(r"ORD-\d{8}-\d{4}", raw_query, re.IGNORECASE),
    }
    # Clean up phone matches (filter out short false positives)
    entities["phones"] = [p.strip() for p in entities["phones"] if len(re.sub(r"\D", "", p)) >= 7]

    # ── Stage 3: Quantity Detection ───────────────────────────────────────
    qty_match = re.search(r"\b([1-9]\d*)\b", raw_query)
    quantity: int = int(qty_match.group(1)) if qty_match else 1
    quantity = min(quantity, 999)  # cap unreasonable orders

    # ── Stage 4: Product Name Extraction ──────────────────────────────────
    clean = raw_query
    # Remove action keywords
    for kws in _ACTION_KEYWORDS.values():
        for kw in kws:
            clean = re.sub(rf"\b{re.escape(kw)}\b", "", clean, flags=re.IGNORECASE)
    # Remove matched quantity
    if qty_match:
        clean = clean.replace(qty_match.group(0), "", 1)
    # Remove filler / stopwords
    clean = _FILLER_PATTERN.sub("", clean)
    # Remove extracted entities
    for email in entities["emails"]:
        clean = clean.replace(email, "")
    for phone in entities["phones"]:
        clean = clean.replace(phone, "")
    for oid in entities["order_ids"]:
        clean = clean.replace(oid, "")

    product_name: str = " ".join(clean.split()).strip(" .,?!;:'\"")
    if not product_name:
        product_name = "unspecified"

    # ── Stage 5: Multi-Item Detection ─────────────────────────────────────
    multi_items: list[dict] = []
    # Check for "and" or comma-separated products with quantities
    split_pattern = re.split(r"\band\b|,", raw_query, flags=re.IGNORECASE)
    if len(split_pattern) > 1:
        for chunk in split_pattern:
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk_qty_m = re.search(r"\b([1-9]\d*)\b", chunk)
            chunk_qty = int(chunk_qty_m.group(1)) if chunk_qty_m else 1
            # Clean the chunk to get product name
            chunk_clean = chunk
            for kws in _ACTION_KEYWORDS.values():
                for kw in kws:
                    chunk_clean = re.sub(rf"\b{re.escape(kw)}\b", "", chunk_clean, flags=re.IGNORECASE)
            if chunk_qty_m:
                chunk_clean = chunk_clean.replace(chunk_qty_m.group(0), "", 1)
            chunk_clean = _FILLER_PATTERN.sub("", chunk_clean)
            chunk_product = " ".join(chunk_clean.split()).strip(" .,?!;:'\"")
            if chunk_product:
                multi_items.append({"product_name": chunk_product, "quantity": chunk_qty})

    is_multi = len(multi_items) > 1

    # ── Stage 6: Sentiment Analysis ───────────────────────────────────────
    pos_count = sum(1 for w in words if w.strip(".,!?;:'\"") in _POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w.strip(".,!?;:'\"") in _NEGATIVE_WORDS)
    total_sentiment_hits = pos_count + neg_count

    if total_sentiment_hits == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = round((pos_count - neg_count) / total_sentiment_hits, 3)

    if sentiment_score > 0.2:
        sentiment_label = "positive"
    elif sentiment_score < -0.2:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # ── Stage 7: Urgency Detection ────────────────────────────────────────
    matched_urgency = [kw for kw in _URGENCY_KEYWORDS if kw in query_lower]
    is_urgent = len(matched_urgency) > 0

    # ── Stage 8: Category Inference ───────────────────────────────────────
    inferred_category = "General"
    best_cat_score = 0
    product_lower = product_name.lower()
    for category, hints in _CATEGORY_HINTS.items():
        score = sum(1 for h in hints if h in product_lower)
        if score > best_cat_score:
            best_cat_score = score
            inferred_category = category

    # ── Stage 9: Confidence Scoring ───────────────────────────────────────
    # Weighted composite: action clarity (40%), product clarity (40%),
    # quantity presence (20%)
    action_score = 1.0 if detected_action != "unknown" else 0.0
    product_score = 1.0 if product_name != "unspecified" else 0.0
    qty_score = 1.0 if qty_match else 0.5  # partial credit if defaulting to 1

    confidence_score = round(
        0.40 * action_score + 0.40 * product_score + 0.20 * qty_score, 3
    )

    if confidence_score >= 0.8:
        confidence_label = "high"
    elif confidence_score >= 0.5:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    # ── Build Response ────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "action":            detected_action,
        "product_name":      product_name,
        "quantity":          quantity,
        "raw_query":         raw_query,
        "confidence":        confidence_label,
        "confidence_score":  confidence_score,
        "sentiment":         sentiment_label,
        "sentiment_score":   sentiment_score,
        "is_urgent":         is_urgent,
        "urgency_keywords":  matched_urgency,
        "detected_entities": entities,
        "multi_item":        is_multi,
        "items":             multi_items if is_multi else [],
        "inferred_category": inferred_category,
        "word_count":        len(words),
        "processed_at":      datetime.now(timezone.utc).isoformat(),
    }

    return result

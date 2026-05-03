"""
tools/parse_customer_query.py
------------------------------
Student 1 – Custom Tool
Tool: parse_customer_query

Extracts structured intent information from a raw customer query string.
Uses simple rule-based NLP so it works without any API calls.

Example
-------
    >>> from tools.parse_customer_query import parse_customer_query
    >>> parse_customer_query("I want to buy 2 Wireless Headphones Pro")
    {'action': 'buy', 'product_name': 'Wireless Headphones Pro', 'quantity': 2, 'raw_query': '...'}
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Intent keyword maps  (ORDER MATTERS – checked top to bottom, first match wins)
# ---------------------------------------------------------------------------
_ACTION_KEYWORDS: dict[str, list[str]] = {
    "return":   ["return", "refund", "send back", "exchange"],
    "cancel":   ["cancel", "remove", "delete", "stop"],
    "track":    ["track", "where is", "status", "shipping", "delivery", "when will"],
    "complain": ["complain", "complaint", "problem", "issue", "broken", "damaged", "wrong"],
    "enquire":  ["what is", "do you have", "available", "how much", "price"],
    "buy":      ["buy", "purchase", "order", "get", "want", "need", "add"],
}


def parse_customer_query(raw_query: str) -> dict[str, Any]:
    """
    Parse a raw customer query into a structured intent dictionary.

    Extracts:
    - action      : The primary intent (buy / return / track / cancel /
                    complain / enquire / unknown)
    - product_name: Best-guess product name extracted from the query
    - quantity    : Number of units requested (defaults to 1)
    - raw_query   : The original input string for downstream agents

    Parameters
    ----------
    raw_query : str
        Free-text customer input, e.g. "I want to buy 3 Mechanical Keyboards".

    Returns
    -------
    dict[str, Any]
        Keys: action (str), product_name (str), quantity (int),
              raw_query (str), confidence (str).

    Raises
    ------
    ValueError
        If raw_query is empty or not a string.
    """
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError("raw_query must be a non-empty string.")

    query_lower = raw_query.lower().strip()

    # ── 1. Detect action ────────────────────────────────────────────────────
    detected_action: str = "unknown"
    for action, keywords in _ACTION_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            detected_action = action
            break

    # ── 2. Extract quantity ──────────────────────────────────────────────────
    qty_match = re.search(r"\b([1-9]\d*)\b", raw_query)   # only positive integers
    quantity: int = int(qty_match.group(1)) if qty_match else 1
    # Cap unreasonably large quantities
    quantity = min(quantity, 999)

    # ── 3. Extract product name ──────────────────────────────────────────────
    # Remove action words and quantity digits to isolate the product mention
    clean = raw_query
    for kws in _ACTION_KEYWORDS.values():
        for kw in kws:
            clean = re.sub(rf"\b{re.escape(kw)}\b", "", clean, flags=re.IGNORECASE)
    if qty_match:
        clean = clean.replace(qty_match.group(0), "")

    # Strip filler words
    filler = r"\b(i|me|my|please|a|an|the|some|for|to|would|like|want|need|can|could|just)\b"
    clean = re.sub(filler, "", clean, flags=re.IGNORECASE)
    product_name: str = " ".join(clean.split()).strip(" .,?!")

    if not product_name:
        product_name = "unspecified"

    # ── 4. Confidence heuristic ──────────────────────────────────────────────
    confidence = "high" if (detected_action != "unknown" and product_name != "unspecified") else \
                 "medium" if detected_action != "unknown" else "low"

    return {
        "action":       detected_action,
        "product_name": product_name,
        "quantity":     quantity,
        "raw_query":    raw_query,
        "confidence":   confidence,
    }

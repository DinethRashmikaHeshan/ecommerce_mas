"""
tools/check_inventory.py
------------------------
Student 2 – Custom Tool
Tool: check_inventory

Queries the local SQLite inventory database to check product availability.
Supports fuzzy product-name matching so minor spelling differences still
resolve to the correct SKU.

Example
-------
    >>> from tools.check_inventory import check_inventory
    >>> check_inventory("Wireless Headphones", 2)
    {'found': True, 'sku': 'SKU-001', 'name': 'Wireless Headphones Pro',
     'price': 89.99, 'stock': 45, 'sufficient_stock': True, ...}
"""

import os
import sqlite3
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "inventory.db")


def _fuzzy_match(query: str, candidate: str) -> float:
    """
    Simple token-overlap similarity score between two strings.

    Parameters
    ----------
    query     : str  - The search term.
    candidate : str  - The product name from the database.

    Returns
    -------
    float  Score between 0.0 (no overlap) and 1.0 (perfect match).
    """
    q_tokens = set(query.lower().split())
    c_tokens = set(candidate.lower().split())
    if not q_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    return len(overlap) / len(q_tokens)


def check_inventory(product_name: str, quantity_requested: int = 1) -> dict[str, Any]:
    """
    Check whether a product is available in the inventory database.

    Performs a case-insensitive fuzzy name search across all products and
    returns stock details for the best match.

    Parameters
    ----------
    product_name       : str  - Product name (partial or full) to search for.
    quantity_requested : int  - Number of units the customer wants (default 1).

    Returns
    -------
    dict[str, Any]
        found            (bool)  – Whether any matching product was found.
        sku              (str)   – Stock-keeping unit identifier.
        name             (str)   – Full product name.
        category         (str)   – Product category.
        price            (float) – Unit price in USD.
        stock            (int)   – Current stock level.
        sufficient_stock (bool)  – True if stock >= quantity_requested.
        match_score      (float) – Fuzzy match confidence (0–1).
        message          (str)   – Human-readable status message.

    Raises
    ------
    ValueError
        If product_name is empty or quantity_requested < 1.
    FileNotFoundError
        If the inventory database has not been initialised (run init_db.py).
    """
    if not isinstance(product_name, str) or not product_name.strip():
        raise ValueError("product_name must be a non-empty string.")
    if not isinstance(quantity_requested, int) or quantity_requested < 1:
        raise ValueError("quantity_requested must be a positive integer.")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Inventory DB not found at {DB_PATH}. Run init_db.py first."
        )

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sku, name, category, price, stock, description FROM products")
    rows   = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "found": False, "sku": "", "name": "", "category": "",
            "price": 0.0, "stock": 0, "sufficient_stock": False,
            "match_score": 0.0, "message": "Inventory database is empty.",
        }

    # Find best fuzzy match
    best_row   = None
    best_score = 0.0
    for row in rows:
        score = _fuzzy_match(product_name, row[1])
        if score > best_score:
            best_score = score
            best_row   = row

    # Threshold – below 0.2 we treat as not found
    if best_score < 0.2 or best_row is None:
        return {
            "found": False, "sku": "", "name": product_name, "category": "",
            "price": 0.0, "stock": 0, "sufficient_stock": False,
            "match_score": round(best_score, 3),
            "message": f"No product matching '{product_name}' was found.",
        }

    sku, name, category, price, stock, description = best_row
    sufficient = stock >= quantity_requested

    message = (
        f"'{name}' is in stock ({stock} units available)."
        if sufficient
        else f"Insufficient stock for '{name}': requested {quantity_requested}, only {stock} available."
    )

    return {
        "found":            True,
        "sku":              sku,
        "name":             name,
        "category":         category,
        "price":            price,
        "stock":            stock,
        "description":      description,
        "sufficient_stock": sufficient,
        "match_score":      round(best_score, 3),
        "message":          message,
    }

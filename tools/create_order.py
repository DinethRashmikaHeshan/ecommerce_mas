"""
tools/create_order.py
---------------------
Student 3 – Custom Tool
Tool: create_order

Creates a new order record in the local JSON order store and decrements
inventory stock in the SQLite database atomically.

Example
-------
    >>> from tools.create_order import create_order
    >>> create_order("SKU-001", "Wireless Headphones Pro", 2, 89.99)
    {'success': True, 'order_id': 'ORD-20240601-0001', 'total': 179.98, ...}
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
ORDERS_PATH = os.path.join(BASE_DIR, "data", "orders.json")
DB_PATH     = os.path.join(BASE_DIR, "data", "inventory.db")


def _next_order_id(orders: list[dict]) -> str:
    """Generate a sequential order ID based on today's date."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"ORD-{today}-"
    existing = [o["order_id"] for o in orders if o["order_id"].startswith(prefix)]
    seq = len(existing) + 1
    return f"{prefix}{seq:04d}"


def create_order(
    sku:           str,
    product_name:  str,
    quantity:      int,
    unit_price:    float,
    customer_note: str = "",
) -> dict[str, Any]:
    """
    Create a confirmed order and persist it to the local JSON store.

    Also decrements the product's stock count in the inventory database
    to prevent overselling.

    Parameters
    ----------
    sku           : str   – Product SKU (e.g. 'SKU-001').
    product_name  : str   – Human-readable product name for the receipt.
    quantity      : int   – Number of units ordered (must be >= 1).
    unit_price    : float – Price per unit in USD (must be > 0).
    customer_note : str   – Optional free-text note attached to the order.

    Returns
    -------
    dict[str, Any]
        success    (bool)  – Whether the order was created successfully.
        order_id   (str)   – Unique order identifier.
        sku        (str)   – Product SKU.
        product    (str)   – Product name.
        quantity   (int)   – Units ordered.
        unit_price (float) – Price per unit.
        total      (float) – Total order value (quantity × unit_price).
        status     (str)   – Order status: 'confirmed' or 'failed'.
        created_at (str)   – ISO-8601 UTC timestamp.
        message    (str)   – Human-readable outcome description.

    Raises
    ------
    ValueError
        If any numeric parameter is out of range or strings are empty.
    FileNotFoundError
        If the orders file or inventory DB has not been initialised.
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if not sku or not isinstance(sku, str):
        raise ValueError("sku must be a non-empty string.")
    if not product_name or not isinstance(product_name, str):
        raise ValueError("product_name must be a non-empty string.")
    if not isinstance(quantity, int) or quantity < 1:
        raise ValueError("quantity must be a positive integer.")
    if not isinstance(unit_price, (int, float)) or unit_price <= 0:
        raise ValueError("unit_price must be a positive number.")
    if not os.path.exists(ORDERS_PATH):
        raise FileNotFoundError(f"Orders file not found at {ORDERS_PATH}. Run init_db.py first.")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Inventory DB not found at {DB_PATH}. Run init_db.py first.")

    # ── Load existing orders ─────────────────────────────────────────────────
    with open(ORDERS_PATH, "r", encoding="utf-8") as fh:
        orders: list[dict] = json.load(fh)

    # ── Re-verify stock inside a DB transaction ──────────────────────────────
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT stock FROM products WHERE sku = ?", (sku,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return {
            "success": False, "order_id": "", "sku": sku,
            "product": product_name, "quantity": quantity,
            "unit_price": unit_price, "total": 0.0,
            "status": "failed", "created_at": datetime.now(timezone.utc).isoformat(),
            "message": f"SKU '{sku}' not found in inventory.",
        }

    current_stock = row[0]
    if current_stock < quantity:
        conn.close()
        return {
            "success": False, "order_id": "", "sku": sku,
            "product": product_name, "quantity": quantity,
            "unit_price": unit_price, "total": 0.0,
            "status": "failed", "created_at": datetime.now(timezone.utc).isoformat(),
            "message": f"Insufficient stock: requested {quantity}, available {current_stock}.",
        }

    # ── Decrement stock ──────────────────────────────────────────────────────
    cursor.execute(
        "UPDATE products SET stock = stock - ? WHERE sku = ?",
        (quantity, sku)
    )
    conn.commit()
    conn.close()

    # ── Build order record ───────────────────────────────────────────────────
    order_id   = _next_order_id(orders)
    total      = round(unit_price * quantity, 2)
    created_at = datetime.now(timezone.utc).isoformat()

    order = {
        "order_id":      order_id,
        "sku":           sku,
        "product":       product_name,
        "quantity":      quantity,
        "unit_price":    unit_price,
        "total":         total,
        "status":        "confirmed",
        "customer_note": customer_note,
        "created_at":    created_at,
    }

    orders.append(order)

    with open(ORDERS_PATH, "w", encoding="utf-8") as fh:
        json.dump(orders, fh, indent=2)

    return {
        "success":    True,
        "order_id":   order_id,
        "sku":        sku,
        "product":    product_name,
        "quantity":   quantity,
        "unit_price": unit_price,
        "total":      total,
        "status":     "confirmed",
        "created_at": created_at,
        "message":    f"Order {order_id} confirmed for {quantity}x '{product_name}'. Total: ${total:.2f}.",
    }

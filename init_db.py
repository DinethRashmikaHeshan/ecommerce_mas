"""
init_db.py
----------
Initializes the local SQLite inventory database and seeds it with sample
e-commerce product data. Run this once before starting the MAS.
"""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "inventory.db")
ORDERS_PATH = os.path.join(os.path.dirname(__file__), "data", "orders.json")


def init_inventory_db() -> None:
    """Create and seed the inventory database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sku         TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL,
            price       REAL NOT NULL,
            stock       INTEGER NOT NULL,
            description TEXT
        )
    """)

    products = [
        ("SKU-001", "Wireless Headphones Pro",    "Electronics",  89.99,  45, "Noise-cancelling over-ear headphones"),
        ("SKU-002", "USB-C Charging Cable 2m",    "Accessories",   9.99, 200, "Fast-charging braided cable"),
        ("SKU-003", "Mechanical Keyboard TKL",    "Electronics", 129.99,  18, "Tenkeyless mechanical gaming keyboard"),
        ("SKU-004", "Laptop Stand Aluminium",     "Accessories",  34.99,  60, "Adjustable aluminium laptop riser"),
        ("SKU-005", "Webcam 1080p HD",            "Electronics",  59.99,   5, "Full HD webcam with built-in mic"),
        ("SKU-006", "Desk Lamp LED",              "Home Office",  24.99,  80, "Dimmable LED desk lamp, USB powered"),
        ("SKU-007", "Wireless Mouse Ergonomic",   "Electronics",  44.99,  30, "Silent ergonomic wireless mouse"),
        ("SKU-008", "HDMI Cable 4K 1.5m",         "Accessories",  12.99, 150, "4K@60Hz certified HDMI 2.0 cable"),
        ("SKU-009", "Monitor 27-inch IPS",        "Electronics", 299.99,   3, "27 inch IPS panel, 144Hz"),
        ("SKU-010", "Cable Management Kit",       "Accessories",  15.99,  95, "Velcro ties, clips and cable box"),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO products (sku, name, category, price, stock, description) VALUES (?,?,?,?,?,?)",
        products
    )

    conn.commit()
    conn.close()
    print(f"[init_db] Inventory DB ready at: {DB_PATH}")


def init_orders_file() -> None:
    """Create an empty orders JSON store if it doesn't exist."""
    os.makedirs(os.path.dirname(ORDERS_PATH), exist_ok=True)
    if not os.path.exists(ORDERS_PATH):
        with open(ORDERS_PATH, "w") as f:
            json.dump([], f, indent=2)
    print(f"[init_db] Orders file ready at: {ORDERS_PATH}")


if __name__ == "__main__":
    init_inventory_db()
    init_orders_file()
    print("[init_db] All data stores initialised successfully.")

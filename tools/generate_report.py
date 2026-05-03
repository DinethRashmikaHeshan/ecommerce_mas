"""
tools/generate_report.py
------------------------
Student 4 – Custom Tool
Tool: generate_report

Reads all order records from the local JSON store and writes a formatted
plain-text / CSV session summary report to the logs/ directory.

Example
-------
    >>> from tools.generate_report import generate_report
    >>> generate_report()
    {'success': True, 'report_path': 'logs/report_20240601_143022.txt',
     'total_orders': 3, 'total_revenue': 359.97, ...}
"""

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
ORDERS_PATH = os.path.join(BASE_DIR, "data", "orders.json")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")


def generate_report(session_summary: str = "") -> dict[str, Any]:
    """
    Aggregate all orders and write a human-readable session report.

    Produces two files in the logs/ directory:
    - A plain-text (.txt) narrative report with totals and per-order details.
    - A CSV (.csv) export of all orders for spreadsheet analysis.

    Parameters
    ----------
    session_summary : str
        Optional free-text paragraph (e.g. from the LLM) to prepend to the
        report as an executive summary.

    Returns
    -------
    dict[str, Any]
        success        (bool)  – Whether the report was written successfully.
        report_path    (str)   – Absolute path to the .txt report file.
        csv_path       (str)   – Absolute path to the .csv export file.
        total_orders   (int)   – Number of confirmed orders in this session.
        total_revenue  (float) – Sum of all order totals (USD).
        categories     (dict)  – Revenue broken down by product category.
        message        (str)   – Human-readable outcome.

    Raises
    ------
    FileNotFoundError
        If the orders JSON file does not exist (run init_db.py first).
    """
    if not os.path.exists(ORDERS_PATH):
        raise FileNotFoundError(
            f"Orders file not found at {ORDERS_PATH}. Run init_db.py first."
        )

    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(ORDERS_PATH, "r", encoding="utf-8") as fh:
        orders: list[dict] = json.load(fh)

    if not orders:
        return {
            "success":       True,
            "report_path":   "",
            "csv_path":      "",
            "total_orders":  0,
            "total_revenue": 0.0,
            "categories":    {},
            "message":       "No orders to report.",
        }

    # ── Aggregations ──────────────────────────────────────────────────────────
    total_revenue = round(sum(o.get("total", 0) for o in orders), 2)
    confirmed     = [o for o in orders if o.get("status") == "confirmed"]

    # ── Timestamp ─────────────────────────────────────────────────────────────
    ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    txt_path   = os.path.join(LOGS_DIR, f"report_{ts}.txt")
    csv_path   = os.path.join(LOGS_DIR, f"orders_{ts}.csv")

    # ── Plain-text report ─────────────────────────────────────────────────────
    lines: list[str] = [
        "=" * 60,
        "  E-COMMERCE MAS  –  SESSION REPORT",
        f"  Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "=" * 60,
        "",
    ]

    if session_summary:
        lines += ["EXECUTIVE SUMMARY", "-" * 40, session_summary, ""]

    lines += [
        "ORDER SUMMARY",
        "-" * 40,
        f"  Total orders     : {len(confirmed)}",
        f"  Total revenue    : ${total_revenue:.2f}",
        "",
        "ORDER DETAILS",
        "-" * 40,
    ]

    for o in confirmed:
        lines.append(
            f"  [{o['order_id']}]  {o['product']}  x{o['quantity']}"
            f"  @ ${o['unit_price']:.2f}  =  ${o['total']:.2f}"
            f"  ({o['created_at'][:10]})"
        )

    lines += ["", "=" * 60, "  END OF REPORT", "=" * 60]

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_fields = ["order_id", "sku", "product", "quantity",
                  "unit_price", "total", "status", "created_at"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(confirmed)

    return {
        "success":       True,
        "report_path":   txt_path,
        "csv_path":      csv_path,
        "total_orders":  len(confirmed),
        "total_revenue": total_revenue,
        "message":       f"Report written to {txt_path}",
    }

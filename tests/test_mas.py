"""
tests/test_mas.py
-----------------
Unified testing harness for the E-Commerce MAS.

Each student section validates their own agent's tool with:
  - Happy-path functional tests
  - Edge-case / boundary tests
  - Property-based tests (hypothesis)
  - LLM-as-a-Judge style output validation (rule-based approximation
    so it runs without a live LLM during CI)

Run
---
    pytest tests/test_mas.py -v
    pytest tests/test_mas.py -v --tb=short   # shorter tracebacks
"""

import json
import os
import sqlite3
import sys
import tempfile
import shutil

import pytest
from hypothesis import given, settings, strategies as st

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import modules (not just functions) so monkeypatch can reach module-level constants
import tools.parse_customer_query as pcq_mod
import tools.check_inventory      as ci_mod
import tools.create_order         as co_mod
import tools.generate_report      as gr_mod
import init_db                    as idb_mod

from tools.parse_customer_query import parse_customer_query
from tools.check_inventory      import check_inventory
from tools.create_order         import create_order
from tools.generate_report      import generate_report
from init_db                    import init_inventory_db, init_orders_file


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    """Create a temporary data directory with a fresh DB and orders file."""
    d = tmp_path_factory.mktemp("data")
    return d


@pytest.fixture(autouse=True)
def patch_paths(monkeypatch, tmp_path):
    """
    Redirect DB_PATH and ORDERS_PATH to a fresh temp directory for each test
    so tests never pollute each other.
    """
    db_path     = str(tmp_path / "inventory.db")
    orders_path = str(tmp_path / "orders.json")

    monkeypatch.setattr(ci_mod,  "DB_PATH",      db_path)
    monkeypatch.setattr(co_mod,  "DB_PATH",      db_path)
    monkeypatch.setattr(co_mod,  "ORDERS_PATH",  orders_path)
    monkeypatch.setattr(gr_mod,  "ORDERS_PATH",  orders_path)
    monkeypatch.setattr(idb_mod, "DB_PATH",      db_path)
    monkeypatch.setattr(idb_mod, "ORDERS_PATH",  orders_path)

    # Seed fresh DB for each test
    idb_mod.init_inventory_db()
    idb_mod.init_orders_file()


# ===========================================================================
# STUDENT 1 – parse_customer_query  (CustomerIntentAgent)
# ===========================================================================

class TestParseCustomerQuery:
    """Student 1 – Unit tests for parse_customer_query tool."""

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_buy_action_detected(self):
        result = parse_customer_query("I want to buy 2 Wireless Headphones Pro")
        assert result["action"] == "buy"

    def test_product_name_extracted(self):
        result = parse_customer_query("I want to buy 2 Wireless Headphones Pro")
        assert "Headphones" in result["product_name"] or "Wireless" in result["product_name"]

    def test_quantity_extracted(self):
        result = parse_customer_query("Please order 5 USB-C cables for me")
        assert result["quantity"] == 5

    def test_default_quantity_is_one(self):
        result = parse_customer_query("I need a webcam")
        assert result["quantity"] == 1

    def test_return_action(self):
        result = parse_customer_query("I want to return my broken keyboard")
        assert result["action"] == "return"

    def test_track_action(self):
        result = parse_customer_query("Where is my order? I need tracking info")
        assert result["action"] == "track"

    def test_cancel_action(self):
        result = parse_customer_query("Please cancel my keyboard order")
        assert result["action"] == "cancel"

    def test_enquire_action(self):
        result = parse_customer_query("How much does the monitor cost?")
        assert result["action"] == "enquire"

    def test_raw_query_preserved(self):
        q = "I want to buy 3 Mechanical Keyboards"
        result = parse_customer_query(q)
        assert result["raw_query"] == q

    def test_confidence_high_when_both_detected(self):
        result = parse_customer_query("Buy 1 USB-C Cable")
        assert result["confidence"] == "high"

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            parse_customer_query("")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(ValueError):
            parse_customer_query("   ")

    def test_raises_on_non_string(self):
        with pytest.raises((ValueError, TypeError)):
            parse_customer_query(None)  # type: ignore

    def test_quantity_capped_at_999(self):
        result = parse_customer_query("Buy 99999 headphones")
        assert result["quantity"] <= 999

    def test_unknown_action_for_gibberish(self):
        result = parse_customer_query("xyzzy frobble wibble")
        assert result["action"] == "unknown"

    # ── Property-based tests ────────────────────────────────────────────────

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_always_returns_required_keys(self, query):
        try:
            result = parse_customer_query(query)
            assert "action" in result
            assert "product_name" in result
            assert "quantity" in result
            assert "confidence" in result
            assert isinstance(result["quantity"], int)
            assert result["quantity"] >= 1
        except ValueError:
            pass  # empty / whitespace-only strings are allowed to raise

    # ── LLM-as-a-Judge style assertion ──────────────────────────────────────

    def test_judge_output_structure_and_validity(self):
        """
        Rule-based LLM-judge: verifies the output meets the contract
        that downstream agents depend on.
        """
        result = parse_customer_query("I'd like to purchase 3 laptop stands")
        # Judge criterion 1: action must be a known value
        valid_actions = {"buy","return","track","cancel","complain","enquire","unknown"}
        assert result["action"] in valid_actions, f"Invalid action: {result['action']}"
        # Judge criterion 2: quantity must be a positive integer
        assert isinstance(result["quantity"], int) and result["quantity"] > 0
        # Judge criterion 3: product_name must be a non-empty string
        assert isinstance(result["product_name"], str) and len(result["product_name"]) > 0
        # Judge criterion 4: confidence must be a known value
        assert result["confidence"] in {"high", "medium", "low"}


# ===========================================================================
# STUDENT 2 – check_inventory  (InventoryAgent)
# ===========================================================================

class TestCheckInventory:
    """Student 2 – Unit tests for check_inventory tool."""

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_known_product_found(self):
        result = check_inventory("Wireless Headphones", 1)
        assert result["found"] is True

    def test_returns_correct_sku(self):
        result = check_inventory("Wireless Headphones Pro", 1)
        assert result["sku"] == "SKU-001"

    def test_sufficient_stock_true(self):
        result = check_inventory("Wireless Headphones Pro", 5)
        assert result["sufficient_stock"] is True

    def test_insufficient_stock_false(self):
        # Request more than available for Monitor (stock=3)
        result = check_inventory("Monitor 27", 10)
        assert result["found"] is True
        assert result["sufficient_stock"] is False

    def test_price_is_positive(self):
        result = check_inventory("USB-C Charging Cable", 1)
        assert result["found"] is True
        assert result["price"] > 0

    def test_match_score_returned(self):
        result = check_inventory("Keyboard", 1)
        assert "match_score" in result
        assert 0.0 <= result["match_score"] <= 1.0

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_unknown_product_not_found(self):
        result = check_inventory("invisible unicorn product xyz", 1)
        assert result["found"] is False

    def test_raises_on_empty_product_name(self):
        with pytest.raises(ValueError):
            check_inventory("", 1)

    def test_raises_on_zero_quantity(self):
        with pytest.raises(ValueError):
            check_inventory("Headphones", 0)

    def test_raises_on_negative_quantity(self):
        with pytest.raises(ValueError):
            check_inventory("Headphones", -3)

    def test_missing_db_raises(self, monkeypatch):
        monkeypatch.setattr(ci_mod, "DB_PATH", "/nonexistent/path.db")
        with pytest.raises(FileNotFoundError):
            check_inventory("Headphones", 1)

    # ── Property-based tests ────────────────────────────────────────────────

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_quantity_does_not_affect_product_lookup(self, qty):
        result = check_inventory("Wireless Headphones Pro", qty)
        assert result["found"] is True
        assert result["sku"] == "SKU-001"

    # ── LLM-as-a-Judge style assertion ──────────────────────────────────────

    def test_judge_inventory_response_contract(self):
        """
        Judge: inventory response must satisfy the contract consumed by
        the OrderProcessingAgent.
        """
        result = check_inventory("Mechanical Keyboard", 2)
        assert result["found"] is True, "Should find keyboard"
        # Contract fields
        required = ["sku", "name", "price", "stock", "sufficient_stock", "message"]
        for field in required:
            assert field in result, f"Missing field: {field}"
        assert isinstance(result["price"], float)
        assert isinstance(result["stock"], int) and result["stock"] >= 0
        assert isinstance(result["sufficient_stock"], bool)
        # message must be informative
        assert len(result["message"]) > 10


# ===========================================================================
# STUDENT 3 – create_order  (OrderProcessingAgent)
# ===========================================================================

class TestCreateOrder:
    """Student 3 – Unit tests for create_order tool."""

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_order_created_successfully(self):
        result = create_order("SKU-001", "Wireless Headphones Pro", 1, 89.99)
        assert result["success"] is True
        assert result["status"] == "confirmed"

    def test_order_id_generated(self):
        result = create_order("SKU-002", "USB-C Charging Cable 2m", 3, 9.99)
        assert result["order_id"].startswith("ORD-")

    def test_total_is_correct(self):
        result = create_order("SKU-004", "Laptop Stand Aluminium", 2, 34.99)
        assert result["total"] == pytest.approx(69.98, rel=1e-4)

    def test_order_persisted_to_file(self, tmp_path, monkeypatch):
        orders_path = str(tmp_path / "orders2.json")
        with open(orders_path, "w") as f:
            json.dump([], f)
        monkeypatch.setattr(co_mod, "ORDERS_PATH", orders_path)

        create_order("SKU-001", "Wireless Headphones Pro", 1, 89.99)
        with open(orders_path) as f:
            orders = json.load(f)
        assert len(orders) == 1
        assert orders[0]["sku"] == "SKU-001"

    def test_stock_decremented_after_order(self):
        import tools.check_inventory as ci
        # Get initial stock
        before = check_inventory("Wireless Headphones Pro", 1)
        initial_stock = before["stock"]
        create_order("SKU-001", "Wireless Headphones Pro", 3, 89.99)
        after = check_inventory("Wireless Headphones Pro", 1)
        assert after["stock"] == initial_stock - 3

    def test_customer_note_stored(self, tmp_path, monkeypatch):
        orders_path = str(tmp_path / "orders_note.json")
        with open(orders_path, "w") as f:
            json.dump([], f)
        monkeypatch.setattr(co_mod, "ORDERS_PATH", orders_path)

        create_order("SKU-001", "Wireless Headphones Pro", 1, 89.99,
                     customer_note="Gift wrap please")
        with open(orders_path) as f:
            orders = json.load(f)
        assert orders[0]["customer_note"] == "Gift wrap please"

    # ── Edge cases / failure paths ──────────────────────────────────────────

    def test_fails_on_oversell(self):
        # Monitor has stock=3; request 99
        result = create_order("SKU-009", "Monitor 27-inch IPS", 99, 299.99)
        assert result["success"] is False
        assert result["status"] == "failed"

    def test_raises_on_invalid_sku(self):
        with pytest.raises(ValueError):
            create_order("", "Some Product", 1, 9.99)

    def test_raises_on_zero_quantity(self):
        with pytest.raises(ValueError):
            create_order("SKU-001", "Headphones", 0, 89.99)

    def test_raises_on_negative_price(self):
        with pytest.raises(ValueError):
            create_order("SKU-001", "Headphones", 1, -5.0)

    def test_raises_on_missing_orders_file(self, monkeypatch):
        monkeypatch.setattr(co_mod, "ORDERS_PATH", "/no/such/file.json")
        with pytest.raises(FileNotFoundError):
            create_order("SKU-001", "Headphones", 1, 89.99)

    # ── Property-based ──────────────────────────────────────────────────────

    @given(
        qty   = st.integers(min_value=1, max_value=10),
        price = st.floats(min_value=0.01, max_value=9999.99, allow_nan=False,
                          allow_infinity=False),
    )
    @settings(max_examples=15)
    def test_total_always_equals_qty_times_price(self, qty, price):
        result = create_order("SKU-002", "USB-C Cable", qty, round(price, 2))
        if result["success"]:
            assert result["total"] == pytest.approx(qty * round(price, 2), rel=1e-4)

    # ── LLM-as-a-Judge style assertion ──────────────────────────────────────

    def test_judge_order_confirmation_contract(self):
        """
        Judge: a confirmed order must carry all fields the ReportAgent needs.
        """
        result = create_order("SKU-007", "Wireless Mouse Ergonomic", 1, 44.99)
        assert result["success"] is True
        required = ["order_id", "sku", "product", "quantity",
                    "unit_price", "total", "status", "created_at", "message"]
        for field in required:
            assert field in result, f"Missing contract field: {field}"
        # order_id format check
        assert result["order_id"].startswith("ORD-")
        # created_at must be ISO-8601
        from datetime import datetime
        datetime.fromisoformat(result["created_at"])   # raises if malformed


# ===========================================================================
# STUDENT 4 – generate_report  (ReportAgent)
# ===========================================================================

class TestGenerateReport:
    """Student 4 – Unit tests for generate_report tool."""

    def _seed_orders(self, orders_path: str, orders: list[dict]) -> None:
        with open(orders_path, "w") as f:
            json.dump(orders, f)

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_report_created_successfully(self):
        # Place a real order first so there is something to report
        create_order("SKU-001", "Wireless Headphones Pro", 2, 89.99)
        result = generate_report("Test session summary.")
        assert result["success"] is True

    def test_report_file_exists(self):
        create_order("SKU-002", "USB-C Cable", 1, 9.99)
        result = generate_report()
        assert os.path.isfile(result["report_path"])

    def test_csv_file_exists(self):
        create_order("SKU-004", "Laptop Stand", 1, 34.99)
        result = generate_report()
        assert os.path.isfile(result["csv_path"])

    def test_total_orders_count(self, monkeypatch, tmp_path):
        op = str(tmp_path / "orders_rep.json")
        monkeypatch.setattr(gr_mod, "ORDERS_PATH", op)
        orders = [
            {"order_id": "ORD-1", "sku": "SKU-001", "product": "Headphones",
             "quantity": 1, "unit_price": 89.99, "total": 89.99,
             "status": "confirmed", "created_at": "2024-01-01T00:00:00+00:00"},
            {"order_id": "ORD-2", "sku": "SKU-002", "product": "Cable",
             "quantity": 2, "unit_price": 9.99,  "total": 19.98,
             "status": "confirmed", "created_at": "2024-01-01T00:00:00+00:00"},
        ]
        self._seed_orders(op, orders)
        result = gr_mod.generate_report()
        assert result["total_orders"] == 2

    def test_total_revenue_calculated(self, monkeypatch, tmp_path):
        op = str(tmp_path / "orders_rev.json")
        monkeypatch.setattr(gr_mod, "ORDERS_PATH", op)
        orders = [
            {"order_id": "ORD-1", "sku": "SKU-001", "product": "Headphones",
             "quantity": 1, "unit_price": 89.99, "total": 89.99,
             "status": "confirmed", "created_at": "2024-01-01T00:00:00+00:00"},
        ]
        self._seed_orders(op, orders)
        result = gr_mod.generate_report()
        assert result["total_revenue"] == pytest.approx(89.99, rel=1e-4)

    def test_empty_orders_returns_gracefully(self, monkeypatch, tmp_path):
        op = str(tmp_path / "empty.json")
        monkeypatch.setattr(gr_mod, "ORDERS_PATH", op)
        with open(op, "w") as f:
            json.dump([], f)
        result = gr_mod.generate_report()
        assert result["success"] is True
        assert result["total_orders"] == 0

    def test_session_summary_appears_in_report(self, tmp_path, monkeypatch):
        op = str(tmp_path / "orders_summ.json")
        monkeypatch.setattr(gr_mod, "ORDERS_PATH", op)
        orders = [
            {"order_id": "ORD-1", "sku": "SKU-001", "product": "Headphones",
             "quantity": 1, "unit_price": 89.99, "total": 89.99,
             "status": "confirmed", "created_at": "2024-01-01T00:00:00+00:00"},
        ]
        self._seed_orders(op, orders)
        summary = "Excellent session with one confirmed order."
        result = gr_mod.generate_report(summary)
        with open(result["report_path"]) as f:
            content = f.read()
        assert summary in content

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_raises_on_missing_orders_file(self, monkeypatch):
        monkeypatch.setattr(gr_mod, "ORDERS_PATH", "/no/such/file.json")
        with pytest.raises(FileNotFoundError):
            generate_report()

    # ── Property-based ──────────────────────────────────────────────────────

    @given(summary = st.text(max_size=120))
    @settings(max_examples=10)
    def test_summary_never_crashes_generate(self, summary):
        import tempfile
        import tools.generate_report as gr
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            op = os.path.join(d, "orders_hyp.json")
            with open(op, "w") as f:
                json.dump([], f)
            with patch.object(gr, "ORDERS_PATH", op):
                result = gr.generate_report(summary)
            assert result["success"] is True

    # ── LLM-as-a-Judge style assertion ──────────────────────────────────────

    def test_judge_report_output_contract(self, monkeypatch, tmp_path):
        """
        Judge: the report result must satisfy the contract the ReportAgent
        presents to the end user.
        """
        op = str(tmp_path / "judge_orders.json")
        monkeypatch.setattr(gr_mod, "ORDERS_PATH", op)
        orders = [
            {"order_id": "ORD-J1", "sku": "SKU-003", "product": "Keyboard",
             "quantity": 1, "unit_price": 129.99, "total": 129.99,
             "status": "confirmed", "created_at": "2024-06-01T10:00:00+00:00"},
        ]
        self._seed_orders(op, orders)
        result = gr_mod.generate_report("Judge test session.")
        # Contract fields
        required = ["success", "report_path", "csv_path",
                    "total_orders", "total_revenue", "message"]
        for field in required:
            assert field in result, f"Missing contract field: {field}"
        # Numeric sanity
        assert result["total_orders"] == 1
        assert result["total_revenue"] == pytest.approx(129.99, rel=1e-4)
        # Files exist
        assert os.path.isfile(result["report_path"])
        assert os.path.isfile(result["csv_path"])
        # CSV has a data row
        with open(result["csv_path"]) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # header + at least 1 order

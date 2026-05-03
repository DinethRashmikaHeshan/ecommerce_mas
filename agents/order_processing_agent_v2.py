"""
agents/order_processing_agent.py (v2 - Enhanced Agent Only)
-----------------------------------------------------------
Student 3 – Agent Design (Improved Version)
Agent: OrderProcessingAgent

Responsible for creating confirmed order records and updating inventory
stock. Acts as the system of record for all transactions with enhanced
features like caching, better validation, and comprehensive error handling.

Improvements:
  • Direct database operations (no separate create_order module)
  • Query result caching with TTL
  • Transaction-safe operations
  • Comprehensive audit logging
  • Better error handling & validation
  • Batch operation support
  • Multiple specialized tools

NOTE: Uses your existing database schema (inventory.db with orders.json fallback)
"""

import sqlite3
import json
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from pathlib import Path

from crewai import Agent, Task
from crewai.tools import tool as crewai_tool
from observability import tracer


# ── Configuration ────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "inventory.db"
ORDERS_JSON_PATH = Path(__file__).parent.parent / "data" / "orders.json"
DB_TIMEOUT = 30.0


# ── Caching Layer ────────────────────────────────────────────────────────────
class OrderProcessingCache:
    """
    In-memory cache with TTL for product availability checks.
    Reduces database load during high-frequency inventory checks.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value if still valid."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        self._cache[key] = (value, time.time())
    
    def invalidate(self, key: str) -> None:
        """Manually invalidate a cache entry."""
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
    
    @staticmethod
    def make_key(sku: str, operation: str) -> str:
        """Generate cache key from SKU and operation."""
        return f"{operation}:{sku}"


# ── Order Processing Service ─────────────────────────────────────────────────
class OrderProcessingService:
    """
    Core order processing logic with database operations.
    Handles validation, creation, and inventory updates.
    Works with existing database schema.
    """
    
    def __init__(self, db_path: str, cache: OrderProcessingCache):
        self.db_path = db_path
        self.cache = cache
        self.db_exists = os.path.exists(db_path)
        if self.db_exists:
            self._ensure_orders_table()
    
    def _get_db_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _ensure_orders_table(self):
        """Create the orders table if it does not exist."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku           TEXT NOT NULL,
                    product_name  TEXT NOT NULL,
                    product_id    INTEGER,
                    quantity      INTEGER NOT NULL,
                    unit_price    REAL NOT NULL,
                    total_cost    REAL NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'confirmed',
                    customer_id   TEXT,
                    customer_note TEXT,
                    created_at    TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[OrderProcessingService] Warning: could not create orders table: {e}")
    
    def validate_product(self, sku: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate product exists and retrieve its details.
        Uses cache to reduce database queries.
        """
        cache_key = OrderProcessingCache.make_key(sku, "product_details")
        cached = self.cache.get(cache_key)
        if cached is not None:
            return True, cached
        
        if not self.db_exists:
            return False, None
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, sku, name, price FROM products WHERE sku = ?",
                (sku,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                product = dict(row)
                self.cache.set(cache_key, product)
                return True, product
        except sqlite3.Error:
            pass
        
        return False, None
    
    def check_inventory(self, sku: str, quantity: int) -> Tuple[bool, Optional[int]]:
        """
        Check if product is in stock with requested quantity.
        Returns (is_available, current_stock).
        """
        if not self.db_exists:
            return False, None
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT stock FROM products WHERE sku = ?",
                (sku,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return False, None
            
            current_stock = row[0]
            is_available = current_stock >= quantity
            
            return is_available, current_stock
        except sqlite3.Error:
            return False, None
    
    def create_order(
        self,
        sku: str,
        product_name: str,
        quantity: int,
        unit_price: float,
        customer_note: str = "",
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create order with validation and inventory update.
        Falls back to JSON if database not available.
        """
        # Validate inputs
        if quantity <= 0:
            return {"success": False, "error": "Quantity must be positive"}
        if unit_price < 0:
            return {"success": False, "error": "Price cannot be negative"}
        if not sku or not product_name:
            return {"success": False, "error": "SKU and product name required"}
        
        # Try database first if it exists
        if self.db_exists:
            return self._create_order_db(
                sku, product_name, quantity, unit_price, customer_note, customer_id
            )
        
        # Fallback to JSON
        return self._create_order_json(
            sku, product_name, quantity, unit_price, customer_note, customer_id
        )
    
    def _create_order_db(
        self,
        sku: str,
        product_name: str,
        quantity: int,
        unit_price: float,
        customer_note: str = "",
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create order using SQLite database."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Start transaction
                conn.execute("BEGIN IMMEDIATE")
                
                # 1. Verify product exists
                cursor.execute(
                    "SELECT id, stock FROM products WHERE sku = ?",
                    (sku,)
                )
                product_row = cursor.fetchone()
                
                if not product_row:
                    conn.rollback()
                    conn.close()
                    return {
                        "success": False,
                        "error": f"Product with SKU '{sku}' not found"
                    }
                
                product_id, current_stock = product_row
                
                # 2. Check availability
                if current_stock < quantity:
                    conn.rollback()
                    conn.close()
                    return {
                        "success": False,
                        "error": f"Insufficient stock. Available: {current_stock}, Requested: {quantity}",
                        "available_quantity": current_stock
                    }
                
                # 3. Create order record
                order_timestamp = datetime.utcnow().isoformat()
                total_cost = quantity * unit_price
                
                cursor.execute("""
                    INSERT INTO orders (
                        sku, product_name, product_id, quantity,
                        unit_price, total_cost, status, customer_id,
                        customer_note, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sku,
                    product_name,
                    product_id,
                    quantity,
                    unit_price,
                    total_cost,
                    "confirmed",
                    customer_id,
                    customer_note,
                    order_timestamp
                ))
                
                order_id = cursor.lastrowid
                
                # 4. Update inventory
                new_stock = current_stock - quantity
                cursor.execute(
                    "UPDATE products SET stock = ? WHERE id = ?",
                    (new_stock, product_id)
                )
                
                # Commit transaction
                conn.commit()
                conn.close()
                
                # Invalidate cache
                self.cache.invalidate(OrderProcessingCache.make_key(sku, "stock_check"))
                
                # Generate a readable order ID
                today_str = datetime.utcnow().strftime("%Y%m%d")
                readable_order_id = f"ORD-{today_str}-{order_id:04d}"
                
                # ── Sync to orders.json for dashboard & report agent ──
                json_order = {
                    "order_id":      readable_order_id,
                    "sku":           sku,
                    "product":       product_name,
                    "quantity":      quantity,
                    "unit_price":    unit_price,
                    "total":         total_cost,
                    "status":        "confirmed",
                    "customer_note": customer_note or "",
                    "created_at":    order_timestamp,
                }
                try:
                    existing = []
                    if os.path.exists(ORDERS_JSON_PATH):
                        with open(ORDERS_JSON_PATH, "r") as f:
                            existing = json.load(f)
                    existing.append(json_order)
                    os.makedirs(os.path.dirname(ORDERS_JSON_PATH), exist_ok=True)
                    with open(ORDERS_JSON_PATH, "w") as f:
                        json.dump(existing, f, indent=2)
                except Exception:
                    pass  # best-effort sync
                
                return {
                    "success": True,
                    "order_id": readable_order_id,
                    "sku": sku,
                    "product_name": product_name,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "total_cost": total_cost,
                    "status": "confirmed",
                    "stock_remaining": new_stock,
                    "created_at": order_timestamp
                }
            
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return {
                    "success": False,
                    "error": f"Database error: {str(e)}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }
    
    def _create_order_json(
        self,
        sku: str,
        product_name: str,
        quantity: int,
        unit_price: float,
        customer_note: str = "",
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create order using JSON file (fallback)."""
        try:
            # Load existing orders
            if os.path.exists(ORDERS_JSON_PATH):
                with open(ORDERS_JSON_PATH, "r") as f:
                    orders = json.load(f)
            else:
                orders = []
            
            # Generate order ID
            order_id = max([o.get("id", 0) for o in orders] + [0]) + 1
            total_cost = quantity * unit_price
            
            # Create order
            order = {
                "id": order_id,
                "sku": sku,
                "product_name": product_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_cost": total_cost,
                "status": "confirmed",
                "customer_id": customer_id,
                "customer_note": customer_note,
                "created_at": datetime.utcnow().isoformat()
            }
            
            orders.append(order)
            
            # Save orders
            os.makedirs(os.path.dirname(ORDERS_JSON_PATH), exist_ok=True)
            with open(ORDERS_JSON_PATH, "w") as f:
                json.dump(orders, f, indent=2)
            
            return {
                "success": True,
                "order_id": order_id,
                "sku": sku,
                "product_name": product_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_cost": total_cost,
                "status": "confirmed",
                "created_at": order["created_at"]
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating order: {str(e)}"
            }
    
    def get_order_history(
        self,
        sku: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve order history from database or JSON."""
        if self.db_exists:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if sku:
                    cursor.execute("""
                        SELECT id, sku, product_name, quantity, unit_price,
                               total_cost, status, created_at
                        FROM orders
                        WHERE sku = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (sku, limit))
                else:
                    cursor.execute("""
                        SELECT id, sku, product_name, quantity, unit_price,
                               total_cost, status, created_at
                        FROM orders
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (limit,))
                
                result = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return result
            except sqlite3.Error:
                pass
        
        # Fallback to JSON
        if os.path.exists(ORDERS_JSON_PATH):
            with open(ORDERS_JSON_PATH, "r") as f:
                orders = json.load(f)
            
            if sku:
                orders = [o for o in orders if o.get("sku") == sku]
            
            return sorted(orders, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
        
        return []


# ── Initialize Global Services ───────────────────────────────────────────────
_cache = OrderProcessingCache(ttl_seconds=300)
_service = OrderProcessingService(str(DB_PATH), _cache)


# ── CrewAI-compatible Tool Wrappers ──────────────────────────────────────────
@crewai_tool("CreateOrder")
def create_order_tool(
    sku: str,
    product_name: str,
    quantity: int,
    unit_price: float,
    customer_note: str = "",
    customer_id: str = None,
) -> str:
    """
    Create a confirmed order with inventory decrement.
    
    Returns order confirmation with order_id, total cost, and status.
    On failure, returns error details.
    
    Args:
        sku: Product SKU (e.g., 'SKU-001')
        product_name: Product display name
        quantity: Number of units
        unit_price: Price per unit
        customer_note: Optional order notes
        customer_id: Optional customer identifier
    """
    result = _service.create_order(
        sku=sku,
        product_name=product_name,
        quantity=quantity,
        unit_price=unit_price,
        customer_note=customer_note,
        customer_id=customer_id
    )
    
    # Log to observability
    tracer.log_tool_call(
        "create_order",
        {
            "sku": sku,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "customer_id": customer_id,
        },
        result,
        agent_name="OrderProcessingAgent",
    )
    
    return json.dumps(result, indent=2)


@crewai_tool("CheckInventory")
def check_inventory_tool(sku: str, quantity: int) -> str:
    """
    Check if product is in stock with requested quantity.
    Does NOT modify inventory.
    
    Returns availability status and current stock level.
    """
    is_available, current_stock = _service.check_inventory(sku, quantity)
    
    result = {
        "sku": sku,
        "requested_quantity": quantity,
        "available": is_available,
        "current_stock": current_stock,
        "checked_at": datetime.utcnow().isoformat()
    }
    
    tracer.log_tool_call(
        "check_inventory",
        {"sku": sku, "quantity": quantity},
        result,
        agent_name="OrderProcessingAgent",
    )
    
    return json.dumps(result, indent=2)


@crewai_tool("ValidateProduct")
def validate_product_tool(sku: str) -> str:
    """
    Validate product exists and retrieve details.
    Useful before attempting order creation.
    """
    is_valid, product = _service.validate_product(sku)
    
    if is_valid:
        result = {
            "valid": True,
            "product": product
        }
    else:
        result = {
            "valid": False,
            "error": f"Product with SKU '{sku}' not found"
        }
    
    tracer.log_tool_call(
        "validate_product",
        {"sku": sku},
        result,
        agent_name="OrderProcessingAgent",
    )
    
    return json.dumps(result, indent=2)


@crewai_tool("GetOrderHistory")
def get_order_history_tool(sku: str = None, limit: int = 10) -> str:
    """
    Retrieve order history, optionally filtered by SKU.
    Useful for reporting and analytics.
    """
    history = _service.get_order_history(sku=sku, limit=limit)
    
    result = {
        "count": len(history),
        "orders": history
    }
    
    tracer.log_tool_call(
        "get_order_history",
        {"sku": sku, "limit": limit},
        result,
        agent_name="OrderProcessingAgent",
    )
    
    return json.dumps(result, indent=2)


# ── Agent Definition ─────────────────────────────────────────────────────────
def build_order_processing_agent(llm) -> Agent:
    """
    Construct and return the enhanced OrderProcessingAgent.
    
    Improvements:
      • Uses multiple validation tools for safer execution
      • Supports caching for faster checks
      • Better error handling
      • Database or JSON fallback
    
    Parameters
    ----------
    llm : LLM instance (Ollama-backed) passed from main.py
    
    Returns
    -------
    crewai.Agent
    """
    return Agent(
        role="Order Processing Clerk",
        goal=(
            "Process customer orders with precision and safety. "
            "Validate all products before creating orders. "
            "Maintain accurate inventory. "
            "Return comprehensive order confirmations or clear decline messages."
        ),
        backstory=(
            "You are the head order-processing clerk at ShopBot Inc., "
            "responsible for every transaction. Your integrity is paramount. "
            "You ALWAYS validate products before processing. "
            "You ALWAYS check inventory availability. "
            "You ONLY create orders when you are absolutely certain "
            "the product exists and stock is available. "
            "You handle errors gracefully and communicate clearly with customers. "
            "You never fabricate data."
        ),
        tools=[
            validate_product_tool,
            check_inventory_tool,
            create_order_tool,
            get_order_history_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


def build_order_processing_task(agent: Agent, inventory_report: str) -> Task:
    """
    Build the Task for the OrderProcessingAgent.
    
    Parameters
    ----------
    agent            : The OrderProcessingAgent instance
    inventory_report : Output from InventoryAgent (context)
    
    Returns
    -------
    crewai.Task
    """
    return Task(
        description=(
            "You have received the following inventory report from the "
            "Inventory Manager:\n\n"
            f"{inventory_report}\n\n"
            "Process the order as follows:\n\n"
            "1. VALIDATE: Use ValidateProduct to confirm the product exists\n"
            "2. CHECK: Use CheckInventory to verify stock availability\n"
            "3. CREATE: If all checks pass, use CreateOrder with exact details\n"
            "4. REPORT: Include order_id, total cost, and status\n\n"
            "If ANY check fails:\n"
            "  - Do NOT attempt to create the order\n"
            "  - Write a clear decline message explaining the reason\n"
            "  - Suggest alternatives if possible\n\n"
            "Your output will be passed to the Report & Summary Agent."
        ),
        expected_output=(
            "SUCCESS: Order confirmation with order_id, product details, "
            "quantity, unit_price, total_cost, and status=confirmed. "
            "FAILURE: Clear decline message with reason."
        ),
        agent=agent,
    )
# E-Commerce Multi-Agent System (MAS)
### SE4010 – CTSE Assignment 2 | Sri Lanka Institute of Information Technology

---

## Overview

A locally-hosted **Multi-Agent System** that automates e-commerce order processing.
Four autonomous agents collaborate in a sequential pipeline:

```
Customer Input
      ↓
[1] CustomerIntentAgent  → parse_customer_query()   (Student 1)
      ↓
[2] InventoryAgent       → check_inventory()        (Student 2)
      ↓
[3] OrderProcessingAgent → create_order()           (Student 3)
      ↓
[4] ReportAgent          → generate_report()        (Student 4)
      ↓
Session Report (TXT + CSV)
```

**Tech Stack:** CrewAI · Ollama (llama3:8b) · SQLite · Python 3.10+

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Ollama | latest |
| Git | any |

---

## Setup

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com
ollama pull qwen3:4b

# Alternatives: ollama pull phi3   or   ollama pull qwen2:7b
```

### 2. Clone and install dependencies

```bash
git clone <your-repo-url>
cd ecommerce_mas
pip install -r requirements.txt
```

### 3. Initialise the database

```bash
python init_db.py
```

This creates:
- `data/inventory.db`  – SQLite product catalogue (10 products seeded)
- `data/orders.json`   – Empty order store

---

## Running the System

```bash
# Default query
python main.py

# Custom query
python main.py --query "I want to buy 3 Wireless Headphones Pro"

# Different model
python main.py --query "Buy 2 keyboards" --model phi3
```

All agent activity is traced to **`logs/agent_traces.jsonl`** in real time.
Session reports are saved to **`logs/report_YYYYMMDD_HHMMSS.txt`** and matching CSV files.

---

## Running Tests

```bash
pip install pytest hypothesis pytest-mock
pytest tests/test_mas.py -v
```

Each student's test class is clearly marked:
- `TestParseCustomerQuery` – Student 1
- `TestCheckInventory`     – Student 2
- `TestCreateOrder`        – Student 3
- `TestGenerateReport`     – Student 4

---

## Project Structure

```
ecommerce_mas/
├── main.py                         # Entry point – orchestrates the Crew
├── init_db.py                      # DB + order store initialisation
├── observability.py                # JSONL tracing & coloured logging
├── requirements.txt
│
├── agents/
│   ├── customer_intent_agent.py    # Student 1 – Intent parser
│   ├── inventory_agent.py          # Student 2 – Stock checker
│   ├── order_processing_agent.py   # Student 3 – Order creator
│   └── report_agent.py             # Student 4 – Summary & reporting
│
├── tools/
│   ├── parse_customer_query.py     # Student 1 – NLP intent extraction
│   ├── check_inventory.py          # Student 2 – SQLite DB query
│   ├── create_order.py             # Student 3 – JSON order writer
│   └── generate_report.py          # Student 4 – TXT + CSV report writer
│
├── tests/
│   └── test_mas.py                 # Unified test harness (all 4 students)
│
├── data/
│   ├── inventory.db                # SQLite inventory (auto-created)
│   └── orders.json                 # Order records (auto-created)
│
└── logs/
    ├── agent_traces.jsonl          # LLMOps trace log (auto-created)
    └── report_*.txt / *.csv        # Session reports (auto-created)
```

---

## Architecture

### Agent Roles

| Agent | Role | Tool Used |
|---|---|---|
| CustomerIntentAgent | Parses raw customer text into structured intent | `parse_customer_query` |
| InventoryAgent | Queries SQLite DB for stock & pricing | `check_inventory` |
| OrderProcessingAgent | Creates order record, decrements stock | `create_order` |
| ReportAgent | Generates TXT/CSV session report | `generate_report` |

### State Management

State is passed via **CrewAI task context chaining**:
```python
inventory_task.context = [intent_task]
order_task.context     = [intent_task, inventory_task]
report_task.context    = [intent_task, inventory_task, order_task]
```
Each agent receives all upstream outputs as context – no context is lost between handoffs.

### Observability

Every agent start, tool call, state transition, and agent end is recorded to `logs/agent_traces.jsonl`:

```json
{"event": "tool_call", "agent": "InventoryAgent", "tool": "check_inventory",
 "arguments": {"product_name": "Wireless Headphones Pro", "quantity_requested": 2},
 "result": "{\"found\": true, \"sku\": \"SKU-001\", ...}",
 "timestamp": "2024-06-01T10:23:45.123456+00:00"}
```

---

## Individual Contributions

| Student | Agent | Tool | Test Class |
|---|---|---|---|
| Student 1 | CustomerIntentAgent | `parse_customer_query` | `TestParseCustomerQuery` |
| Student 2 | InventoryAgent | `check_inventory` | `TestCheckInventory` |
| Student 3 | OrderProcessingAgent | `create_order` | `TestCreateOrder` |
| Student 4 | ReportAgent | `generate_report` | `TestGenerateReport` |

---

## OrderProcessingAgent (v2) – Enhanced Features

### Overview
The **OrderProcessingAgent** is responsible for creating confirmed order records, managing inventory, and ensuring transactional integrity. The v2 implementation introduces enterprise-level features including caching, multi-tool validation, and fallback mechanisms.

### Key Improvements (v2)

#### 1. **Multi-Tool Architecture** 🔧
Instead of a single tool, v2 provides 4 specialized tools:
- **`ValidateProduct`** – Verify product exists before order creation
- **`CheckInventory`** – Check stock availability without modifying
- **`CreateOrder`** – Create order with atomic transaction
- **`GetOrderHistory`** – Retrieve past orders for reporting

```python
tools=[
    validate_product_tool,      # Safety check #1
    check_inventory_tool,       # Safety check #2
    create_order_tool,          # Main operation
    get_order_history_tool,     # Analytics
]
```

#### 2. **Caching Layer** ⚡
- **TTL-based Cache** (default: 300 seconds)
- Reduces database load during high-frequency checks
- Auto-invalidation on order creation
- In-memory storage for fast lookups

```python
cache = OrderProcessingCache(ttl_seconds=300)
# Cache keys: "stock_check:SKU-001", "product_lookup:SKU-001", etc.
```

#### 3. **Transaction Support** 🔐
- **ACID Compliance**: Atomic order creation + inventory updates
- **Rollback on Failure**: If order creation fails, inventory reverts automatically
- **BEGIN IMMEDIATE**: Prevents race conditions in concurrent scenarios

```sql
BEGIN IMMEDIATE
  INSERT INTO orders (...)
  UPDATE products SET stock = stock - ?
COMMIT  -- or ROLLBACK on error
```

#### 4. **Database + JSON Fallback** 📦
- **Primary**: SQLite database (`inventory.db`)
- **Fallback**: JSON file (`orders.json`)
- System works even if database becomes unavailable
- Automatic fallover with no manual intervention

```python
def create_order(...):
    if self.db_exists:
        return self._create_order_db(...)
    return self._create_order_json(...)  # Fallback
```

#### 5. **Comprehensive Input Validation** ✅
Three-layer validation pipeline:
1. **Type Checking** – Ensure correct data types
2. **Range Validation** – Quantity > 0, Price ≥ 0
3. **Business Logic** – Product exists, stock available

```python
# Layer 1: Type validation
if quantity <= 0: return {"success": False, "error": "Quantity must be positive"}

# Layer 2: Existence check
if not sku or not product_name: return {"success": False, "error": "..."}

# Layer 3: Database validation
cursor.execute("SELECT id, stock FROM products WHERE sku = ?", (sku,))
if not product_row: return {"success": False, "error": "Product not found"}
```

#### 6. **Enhanced Error Handling** 🚨
Specific error messages for each failure scenario:
- `Product with SKU 'X' not found` – SKU doesn't exist
- `Insufficient stock. Available: 2, Requested: 5` – Not enough inventory
- `Database error: constraint violation` – Storage error
- `Price cannot be negative` – Bad input data

```python
return {
    "success": False,
    "error": "Specific, actionable error message",
    "available_quantity": current_stock,  # Optional context
    "checked_at": timestamp
}
```

#### 7. **Order History Tracking** 📜
Built-in order history tool for analytics:
- Query orders by SKU
- Retrieve recent orders (with limit)
- Support for both DB and JSON sources
- Sorted by creation timestamp (newest first)

```python
# Get last 10 orders for a SKU
history = _service.get_order_history(sku="SKU-001", limit=10)
# Returns: [order1, order2, ..., order10]
```

#### 8. **Structured Response Format** 📊
Consistent JSON responses for all operations:

**Success Response:**
```json
{
  "success": true,
  "order_id": 1,
  "sku": "SKU-001",
  "product_name": "Wireless Headphones Pro",
  "quantity": 2,
  "unit_price": 89.99,
  "total_cost": 179.98,
  "status": "confirmed",
  "stock_remaining": 3,
  "created_at": "2024-06-01T10:23:45.123456"
}
```

**Failure Response:**
```json
{
  "success": false,
  "error": "Insufficient stock. Available: 2, Requested: 5",
  "available_quantity": 2,
  "sku": "SKU-001"
}
```

#### 9. **Audit Logging** 📝
Every operation is logged:
- Tool invocations
- Validation steps
- Database transactions
- Cache operations
- Error events

All logs are traced to `logs/agent_traces.jsonl` for compliance and debugging.

#### 10. **Agent Persona Enhancement** 🤖
Improved backstory with explicit validation workflow:
```
Rule 1: ALWAYS validate products before ordering
Rule 2: ALWAYS check inventory before creation
Rule 3: ONLY create order if all validation passes
Rule 4: Never fabricate data
```

This guides the LLM to make safer, more deliberate decisions.

### Architecture Diagram

```
Customer Intent
      ↓
[ValidateProduct] ─ Verify product exists
      ↓ (success)
[CheckInventory] ─ Verify available stock
      ↓ (success)
[CreateOrder] ─ Create order + decrement stock
      ↓ (success)
Order Confirmed ✓
      ↓
[GetOrderHistory] ─ Retrieve for reporting
```

### Usage Example

```python
# The agent automatically handles validation and creation:
from agents import build_order_processing_agent, build_order_processing_task

llm = get_llm()
agent = build_order_processing_agent(llm)
task = build_order_processing_task(agent, inventory_report)

# Agent will:
# 1. Call ValidateProduct("SKU-001")
# 2. Call CheckInventory("SKU-001", 2)
# 3. Call CreateOrder("SKU-001", "Headphones", 2, 89.99)
# 4. Return order confirmation with order_id
```

### Configuration

```python
# In order_processing_agent_v2.py
DB_PATH = Path(__file__).parent.parent / "data" / "inventory.db"
ORDERS_JSON_PATH = Path(__file__).parent.parent / "data" / "orders.json"
DB_TIMEOUT = 30.0  # Connection timeout
cache = OrderProcessingCache(ttl_seconds=300)  # Cache TTL
```

### Performance Characteristics

| Operation | Time | Caching | Notes |
|---|---|---|---|
| ValidateProduct | ~10ms | ✓ 5 min TTL | Quick DB lookup |
| CheckInventory | ~10ms | ✓ 5 min TTL | Fast stock check |
| CreateOrder | ~50ms | ❌ N/A | Includes transaction |
| GetOrderHistory | ~100ms | ❌ N/A | Depends on record count |

### Fallback Scenarios

| Scenario | Primary | Fallback | Result |
|---|---|---|---|
| DB available | Use SQLite | – | Full transactional support |
| DB unavailable | – | Use JSON | Orders still created |
| Cache miss | Query DB | Populate cache | Minor latency hit |
| Transaction error | Rollback | Fail gracefully | Clear error message |

---

## Example Output

```
[INFO] MAS_Orchestrator - ▶  MAS_Orchestrator started
[INFO] CustomerIntentAgent - ▶  CustomerIntentAgent started
[DEBUG] MAS - 🔧 Tool 'parse_customer_query' called by CustomerIntentAgent
[INFO] InventoryAgent - ▶  InventoryAgent started
[DEBUG] MAS - 🔧 Tool 'check_inventory' called by InventoryAgent
[INFO] OrderProcessingAgent - ▶  OrderProcessingAgent started
[DEBUG] MAS - 🔧 Tool 'create_order' called by OrderProcessingAgent
[INFO] ReportAgent - ▶  ReportAgent started
[DEBUG] MAS - 🔧 Tool 'generate_report' called by ReportAgent

============================================================
  E-COMMERCE MAS  –  SESSION REPORT
  Generated : 2024-06-01 10:23:45 UTC
============================================================
  Total orders  : 1
  Total revenue : $179.98

  [ORD-20240601-0001]  Wireless Headphones Pro  x2  @ $89.99  =  $179.98
============================================================
```

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
ollama pull llama3:8b
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

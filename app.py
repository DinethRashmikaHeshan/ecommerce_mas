"""
app.py
------
Flask web server for the E-Commerce MAS Dashboard.
Provides a visual frontend and REST API for the multi-agent pipeline.

Run
---
    python app.py
    # Then open http://localhost:5000 in your browser
"""

import json
import os
import sqlite3
import threading

from flask import Flask, render_template, jsonify, request

from init_db import init_inventory_db, init_orders_file

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "data", "inventory.db")
ORDERS_PATH = os.path.join(BASE_DIR, "data", "orders.json")
TRACE_FILE = os.path.join(BASE_DIR, "logs", "agent_traces.jsonl")

app = Flask(__name__)

# Store pipeline run state
pipeline_state = {
    "running": False,
    "current_agent": None,
    "steps": [],
    "result": None,
    "error": None,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/inventory")
def get_inventory():
    """Return all products from the inventory database."""
    if not os.path.exists(DB_PATH):
        init_inventory_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(products)


@app.route("/api/orders")
def get_orders():
    """Return all orders from the JSON store."""
    if not os.path.exists(ORDERS_PATH):
        init_orders_file()
    with open(ORDERS_PATH, "r") as f:
        orders = json.load(f)
    return jsonify(orders)


@app.route("/api/traces")
def get_traces():
    """Return the last 100 trace events."""
    if not os.path.exists(TRACE_FILE):
        return jsonify([])
    traces = []
    with open(TRACE_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return jsonify(traces[-100:])


@app.route("/api/status")
def get_status():
    """Return current pipeline execution status."""
    return jsonify(pipeline_state)


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Run the MAS pipeline with a customer query."""
    global pipeline_state

    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    data = request.get_json()
    query = data.get("query", "I want to buy 2 Wireless Headphones Pro")
    model = data.get("model", "qwen3:4b")

    pipeline_state = {
        "running": True,
        "current_agent": "Initializing",
        "steps": [],
        "result": None,
        "error": None,
    }

    def run_in_thread():
        global pipeline_state
        try:
            from main import run_mas
            pipeline_state["current_agent"] = "CustomerIntentAgent"
            pipeline_state["steps"].append({
                "agent": "Pipeline",
                "status": "started",
                "message": f"Processing: {query}"
            })
            result = run_mas(query, model)
            pipeline_state["result"] = result
            pipeline_state["current_agent"] = "Complete"
            pipeline_state["steps"].append({
                "agent": "Pipeline",
                "status": "complete",
                "message": "All agents finished successfully"
            })
        except Exception as e:
            pipeline_state["error"] = str(e)
            pipeline_state["current_agent"] = "Error"
            pipeline_state["steps"].append({
                "agent": "Pipeline",
                "status": "error",
                "message": str(e)
            })
        finally:
            pipeline_state["running"] = False

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return jsonify({"status": "started", "query": query, "model": model})


@app.route("/api/reset-orders", methods=["POST"])
def reset_orders():
    """Reset the orders file to empty."""
    with open(ORDERS_PATH, "w") as f:
        json.dump([], f, indent=2)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    init_inventory_db()
    init_orders_file()
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    print("\n" + "=" * 55)
    print("  E-COMMERCE MAS DASHBOARD")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000)

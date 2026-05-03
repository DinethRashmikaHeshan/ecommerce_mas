"""
main.py
-------
Entry point for the E-Commerce Multi-Agent System (MAS).

Run
---
    # 1. Make sure Ollama is running and the model is pulled:
    #    ollama pull llama3:8b
    #
    # 2. Initialise data stores (only needed once):
    #    python init_db.py
    #
    # 3. Run the system:
    #    python main.py
    #    python main.py --query "I want to buy 2 Wireless Headphones Pro"
    #    python main.py --model qwen2:7b

Architecture
------------
CustomerIntentAgent  →  InventoryAgent  →  OrderProcessingAgent  →  ReportAgent
     (parse)               (check DB)          (write order)          (summarise)

State is passed sequentially via CrewAI task context chaining.
All agent I/O is traced to logs/agent_traces.jsonl via observability.py.
"""

import argparse
import sys

from crewai import Crew, Process

from init_db import init_inventory_db, init_orders_file
from observability import tracer, console_logger
from agents import (
    build_customer_intent_agent, build_customer_intent_task,
    build_inventory_agent,       build_inventory_task,
    build_order_processing_agent,build_order_processing_task,
    build_report_agent,          build_report_task,
)


# ---------------------------------------------------------------------------
# Default customer query (used when no --query flag is given)
# ---------------------------------------------------------------------------
DEFAULT_QUERY = "I would like to purchase 2 Wireless Headphones Pro please"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E-Commerce Multi-Agent System")
    parser.add_argument(
        "--query", type=str, default=DEFAULT_QUERY,
        help="Customer query to process (wrap in quotes)"
    )
    parser.add_argument(
        "--model", type=str, default="llama3:8b",
        help="Ollama model name (default: llama3:8b). Try phi3, qwen2:7b, mistral."
    )
    return parser.parse_args()


def build_llm(model_name: str) -> str:
    """Return the Ollama model identifier string for CrewAI."""
    console_logger.info(f"Loading Ollama model: {model_name}")
    return f"ollama/{model_name}"


def run_mas(customer_query: str, model_name: str = "llama3:8b") -> str:
    """
    Execute the full 4-agent pipeline for a given customer query.

    Parameters
    ----------
    customer_query : str  – Raw customer input.
    model_name     : str  – Ollama model identifier.

    Returns
    -------
    str  – Final report output from the ReportAgent.
    """
    # ── Ensure data stores exist ──────────────────────────────────────────
    init_inventory_db()
    init_orders_file()

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = build_llm(model_name)

    # ── Role → agent-name mapping for tracing ─────────────────────────────
    _ROLE_TO_AGENT = {
        "Customer Intent Analyst": "CustomerIntentAgent",
        "Inventory Manager":       "InventoryAgent",
        "Order Processing Clerk":  "OrderProcessingAgent",
        "Business Report Analyst": "ReportAgent",
    }

    # Sequential task index tracker for state transitions
    _task_index = {"current": 0}
    _agent_order = ["CustomerIntentAgent", "InventoryAgent",
                    "OrderProcessingAgent", "ReportAgent"]

    def _task_callback(task_output):
        """Log each task completion so the dashboard can track progress."""
        agent_role = getattr(task_output, "agent", None) or "Unknown"
        agent_name = _ROLE_TO_AGENT.get(str(agent_role), str(agent_role))
        raw = str(getattr(task_output, "raw", ""))[:300]
        tracer.log_agent_end(agent_name, raw)
        # Log state transition to next agent
        idx = _task_index["current"]
        _task_index["current"] = idx + 1
        if idx + 1 < len(_agent_order):
            tracer.log_state_transition(
                _agent_order[idx], _agent_order[idx + 1],
                ["task_output"]
            )

    # ── Build agents ──────────────────────────────────────────────────────
    tracer.log_agent_start("MAS_Orchestrator", customer_query)
    console_logger.info("=" * 55)
    console_logger.info(" E-COMMERCE MULTI-AGENT SYSTEM  –  ShopBot Inc.")
    console_logger.info("=" * 55)
    console_logger.info(f"Customer query : {customer_query}")
    console_logger.info(f"Model          : {model_name}")
    console_logger.info("=" * 55)

    intent_agent    = build_customer_intent_agent(llm)
    inventory_agent = build_inventory_agent(llm)
    order_agent     = build_order_processing_agent(llm)
    report_agent    = build_report_agent(llm)

    # ── Build tasks with context chaining ────────────────────────────────
    intent_task    = build_customer_intent_task(intent_agent,    customer_query)
    inventory_task = build_inventory_task(inventory_agent,       "{intent_task_output}")
    order_task     = build_order_processing_task(order_agent,    "{inventory_task_output}")
    report_task    = build_report_task(report_agent,             "{order_task_output}")

    # Wire up context so each task receives the previous task's output
    inventory_task.context = [intent_task]
    order_task.context     = [intent_task, inventory_task]
    report_task.context    = [intent_task, inventory_task, order_task]

    # Log first agent start
    tracer.log_agent_start("CustomerIntentAgent", customer_query)

    # ── Assemble crew ─────────────────────────────────────────────────────
    crew = Crew(
        agents=[intent_agent, inventory_agent, order_agent, report_agent],
        tasks=[intent_task, inventory_task, order_task, report_task],
        process=Process.sequential,   # Coordinator-Worker sequential pipeline
        verbose=True,
        task_callback=_task_callback,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    tracer.log_state_transition("Orchestrator", "CustomerIntentAgent", ["raw_query"])
    result = crew.kickoff()

    tracer.log_agent_end("MAS_Orchestrator", str(result))

    console_logger.info("=" * 55)
    console_logger.info(" PIPELINE COMPLETE")
    console_logger.info("=" * 55)

    return str(result)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    try:
        output = run_mas(args.query, args.model)
        print("\n" + "=" * 55)
        print(" FINAL OUTPUT")
        print("=" * 55)
        print(output)
    except Exception as exc:
        tracer.log_error("MAS_Orchestrator", exc)
        console_logger.error(f"Fatal error: {exc}")
        sys.exit(1)

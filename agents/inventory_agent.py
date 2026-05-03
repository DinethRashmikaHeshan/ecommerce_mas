"""
agents/inventory_agent.py
--------------------------
Student 2 – Agent Design
Agent: InventoryAgent

Responsible for checking product availability in the local SQLite database
and deciding whether the order can proceed.

Persona
-------
A diligent warehouse manager who gives accurate, no-nonsense stock reports
and always flags low-stock situations proactively.
"""

from crewai import Agent, Task
from crewai.tools import tool as crewai_tool
from observability import tracer
from tools.check_inventory import check_inventory as _check


# ── CrewAI-compatible tool wrapper ───────────────────────────────────────────
@crewai_tool("CheckInventory")
def check_inventory_tool(product_name: str, quantity_requested: int = 1) -> str:
    """
    Check the local inventory database for a product by name.
    Returns stock details including availability and unit price.
    """
    import json
    result = _check(product_name, quantity_requested)
    tracer.log_tool_call(
        "check_inventory",
        {"product_name": product_name, "quantity_requested": quantity_requested},
        result,
        agent_name="InventoryAgent",
    )
    return json.dumps(result, indent=2)


# ── Agent definition ─────────────────────────────────────────────────────────
def build_inventory_agent(llm) -> Agent:
    """
    Construct and return the InventoryAgent.

    Parameters
    ----------
    llm : LLM instance (Ollama-backed) passed in from main.py.

    Returns
    -------
    crewai.Agent
    """
    return Agent(
        role="Inventory Manager",
        goal=(
            "Determine whether the requested product is in stock in sufficient "
            "quantity. Provide the SKU, current stock level, unit price, and a "
            "clear AVAILABLE or UNAVAILABLE verdict for the Order Processing Agent."
        ),
        backstory=(
            "You are the head of inventory management at ShopBot Inc. "
            "You have direct read access to the warehouse database and take "
            "pride in giving accurate, real-time stock information. "
            "You always use the CheckInventory tool – never guess stock levels. "
            "If stock is low (under 5 units remaining after the order), you must "
            "include a LOW STOCK WARNING in your output. "
            "If the product is not found, suggest the closest alternative if one exists."
        ),
        tools=[check_inventory_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_inventory_task(agent: Agent, intent_summary: str) -> Task:
    """
    Build the Task for the InventoryAgent.

    Parameters
    ----------
    agent          : The InventoryAgent instance.
    intent_summary : Output from the CustomerIntentAgent task (context).

    Returns
    -------
    crewai.Task
    """
    return Task(
        description=(
            "You have received the following customer intent summary from the "
            "Customer Intent Analyst:\n\n"
            f"{intent_summary}\n\n"
            "Use the CheckInventory tool to verify product availability. "
            "Your response MUST include:\n"
            "- Product SKU\n"
            "- Product name (as it appears in the database)\n"
            "- Current stock level\n"
            "- Unit price\n"
            "- Verdict: AVAILABLE or UNAVAILABLE\n"
            "- Any LOW STOCK WARNING if applicable\n\n"
            "Pass this information to the Order Processing Agent."
        ),
        expected_output=(
            "An inventory report with SKU, product name, stock level, unit price, "
            "AVAILABLE/UNAVAILABLE verdict, and optional LOW STOCK WARNING."
        ),
        agent=agent,
    )

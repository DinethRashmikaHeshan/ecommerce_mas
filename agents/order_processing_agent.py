"""
agents/order_processing_agent.py
---------------------------------
Student 3 – Agent Design
Agent: OrderProcessingAgent

Responsible for creating confirmed order records and updating inventory
stock. Acts as the system of record for all transactions.

Persona
-------
A precise order-desk clerk who confirms every detail before committing
a transaction, and never places an order without a valid SKU and price.
"""

from crewai import Agent, Task
from crewai.tools import tool as crewai_tool
from observability import tracer
from tools.create_order import create_order as _create


# ── CrewAI-compatible tool wrapper ───────────────────────────────────────────
@crewai_tool("CreateOrder")
def create_order_tool(
    sku: str,
    product_name: str,
    quantity: int,
    unit_price: float,
    customer_note: str = "",
) -> str:
    """
    Create a confirmed order in the local JSON order store and decrement
    inventory. Returns order confirmation details including order_id and total.
    """
    import json
    result = _create(sku, product_name, quantity, unit_price, customer_note)
    tracer.log_tool_call(
        "create_order",
        {
            "sku": sku,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
        },
        result,
        agent_name="OrderProcessingAgent",
    )
    return json.dumps(result, indent=2)


# ── Agent definition ─────────────────────────────────────────────────────────
def build_order_processing_agent(llm) -> Agent:
    """
    Construct and return the OrderProcessingAgent.

    Parameters
    ----------
    llm : LLM instance (Ollama-backed) passed in from main.py.

    Returns
    -------
    crewai.Agent
    """
    return Agent(
        role="Order Processing Clerk",
        goal=(
            "Create a confirmed order record for every AVAILABLE product request. "
            "Decline to process orders for UNAVAILABLE items. "
            "Return a full order confirmation including order_id, total cost, "
            "and estimated status for the Report Agent."
        ),
        backstory=(
            "You are the head order-processing clerk at ShopBot Inc. "
            "You handle all purchase confirmations with extreme precision. "
            "You ONLY call the CreateOrder tool when the Inventory Agent has "
            "confirmed the product is AVAILABLE. "
            "If the product is UNAVAILABLE, you write a polite decline note "
            "without calling any tool. "
            "You never fabricate order IDs or prices – all data comes from the "
            "Inventory Agent's report. "
            "Every order you confirm must include the SKU, product name, quantity, "
            "unit price, and the resulting order_id."
        ),
        tools=[create_order_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_order_processing_task(agent: Agent, inventory_report: str) -> Task:
    """
    Build the Task for the OrderProcessingAgent.

    Parameters
    ----------
    agent            : The OrderProcessingAgent instance.
    inventory_report : Output from the InventoryAgent task (context).

    Returns
    -------
    crewai.Task
    """
    return Task(
        description=(
            "You have received the following inventory report from the "
            "Inventory Manager:\n\n"
            f"{inventory_report}\n\n"
            "If the verdict is AVAILABLE:\n"
            "  - Call the CreateOrder tool with the exact SKU, product name, "
            "    quantity, and unit price from the inventory report.\n"
            "  - Report the resulting order_id, total cost, and status.\n\n"
            "If the verdict is UNAVAILABLE:\n"
            "  - Do NOT call any tool.\n"
            "  - Write a decline message explaining why the order cannot be placed.\n\n"
            "Your output will be passed to the Report & Summary Agent."
        ),
        expected_output=(
            "Either: an order confirmation with order_id, product, quantity, "
            "unit_price, total, and status=confirmed. "
            "Or: a clear decline message if the item is unavailable."
        ),
        agent=agent,
    )

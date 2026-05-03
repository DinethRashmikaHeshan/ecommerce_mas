"""
agents/customer_intent_agent.py
--------------------------------
Student 1 – Agent Design
Agent: CustomerIntentAgent

Responsible for receiving raw customer input and producing a structured
intent payload that downstream agents can act on.

Persona
-------
A meticulous customer-service analyst who never assumes – if the intent
is ambiguous, the agent flags it clearly rather than guessing.
"""

from crewai import Agent, Task
from crewai.tools import tool as crewai_tool
from observability import tracer
from tools.parse_customer_query import parse_customer_query as _parse


# ── CrewAI-compatible tool wrapper ───────────────────────────────────────────
@crewai_tool("ParseCustomerQuery")
def parse_query_tool(raw_query: str) -> str:
    """
    Parse a raw customer query string into a structured intent dictionary.
    Returns a JSON-formatted string with keys: action, product_name,
    quantity, confidence.
    """
    import json
    result = _parse(raw_query)
    tracer.log_tool_call(
        "parse_customer_query",
        {"raw_query": raw_query},
        result,
        agent_name="CustomerIntentAgent",
    )
    return json.dumps(result, indent=2)


# ── Agent definition ─────────────────────────────────────────────────────────
def build_customer_intent_agent(llm) -> Agent:
    """
    Construct and return the CustomerIntentAgent.

    Parameters
    ----------
    llm : LLM instance (Ollama-backed) passed in from main.py.

    Returns
    -------
    crewai.Agent
    """
    return Agent(
        role="Customer Intent Analyst",
        goal=(
            "Accurately interpret the customer's request and extract: "
            "(1) the desired action (buy/return/track/cancel/complain/enquire), "
            "(2) the product name, and (3) the quantity required. "
            "Produce a clean, structured summary to hand off to the Inventory Agent."
        ),
        backstory=(
            "You are a senior customer-experience analyst at ShopBot Inc. "
            "You have processed thousands of customer queries and pride yourself "
            "on never misinterpreting intent. You always use the ParseCustomerQuery "
            "tool to extract structured data, and you clearly flag any ambiguity "
            "rather than guessing. Your output MUST include the action, product name, "
            "and quantity so that the next agent can proceed without asking follow-up questions."
        ),
        tools=[parse_query_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_customer_intent_task(agent: Agent, customer_query: str) -> Task:
    """
    Build the Task for the CustomerIntentAgent.

    Parameters
    ----------
    agent          : The CustomerIntentAgent instance.
    customer_query : Raw text from the customer.

    Returns
    -------
    crewai.Task
    """
    return Task(
        description=(
            f"A customer has submitted the following query:\n\n"
            f'  "{customer_query}"\n\n'
            "Use the ParseCustomerQuery tool to extract the intent. "
            "Then write a concise structured summary including:\n"
            "- Action (buy / return / track / cancel / complain / enquire)\n"
            "- Product name\n"
            "- Quantity\n"
            "- Confidence level\n"
            "- Any ambiguities or special notes\n\n"
            "Your output will be passed directly to the Inventory Agent."
        ),
        expected_output=(
            "A structured intent summary with: action, product_name, quantity, "
            "confidence, and any notes about ambiguity."
        ),
        agent=agent,
    )

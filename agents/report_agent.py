"""
agents/report_agent.py
-----------------------
Student 4 – Agent Design
Agent: ReportAgent

Responsible for synthesising all upstream agent outputs into a polished
session report and persisting it to disk via the GenerateReport tool.

Persona
-------
A sharp business analyst who turns raw transaction data into clean,
executive-ready summaries – never vague, always data-driven.
"""

from crewai import Agent, Task
from crewai.tools import tool as crewai_tool
from observability import tracer
from tools.generate_report import generate_report as _report


# ── CrewAI-compatible tool wrapper ───────────────────────────────────────────
@crewai_tool("GenerateReport")
def generate_report_tool(session_summary: str = "") -> str:
    """
    Aggregate all orders and write a formatted session report (.txt + .csv)
    to the logs/ directory. Returns file paths and revenue totals.
    """
    import json
    result = _report(session_summary)
    tracer.log_tool_call(
        "generate_report",
        {"session_summary": session_summary[:100]},
        result,
        agent_name="ReportAgent",
    )
    return json.dumps(result, indent=2)


# ── Agent definition ─────────────────────────────────────────────────────────
def build_report_agent(llm) -> Agent:
    """
    Construct and return the ReportAgent.

    Parameters
    ----------
    llm : LLM instance (Ollama-backed) passed in from main.py.

    Returns
    -------
    crewai.Agent
    """
    return Agent(
        role="Business Report Analyst",
        goal=(
            "Produce a concise executive summary of the session, call the "
            "GenerateReport tool to persist the data, and present the final "
            "outcome clearly to the user including total revenue and order count."
        ),
        backstory=(
            "You are the senior business analyst at ShopBot Inc. "
            "After every operational session you synthesise outputs from the "
            "entire agent pipeline into a clean, data-driven report. "
            "You always call the GenerateReport tool so nothing is lost. "
            "Your written summary must cover: what the customer wanted, "
            "whether inventory was available, the final order outcome, "
            "total revenue for this session, and any warnings or issues. "
            "Keep the executive summary under 120 words – sharp and professional."
        ),
        tools=[generate_report_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_report_task(agent: Agent, order_outcome: str) -> Task:
    """
    Build the Task for the ReportAgent.

    Parameters
    ----------
    agent         : The ReportAgent instance.
    order_outcome : Output from the OrderProcessingAgent task (context).

    Returns
    -------
    crewai.Task
    """
    return Task(
        description=(
            "You have received the following order outcome from the "
            "Order Processing Clerk:\n\n"
            f"{order_outcome}\n\n"
            "Your tasks:\n"
            "1. Write a concise executive summary (max 120 words) covering the "
            "   entire session: customer intent → inventory check → order result.\n"
            "2. Call the GenerateReport tool with your executive summary as the "
            "   session_summary argument.\n"
            "3. Present the final output to the user, including:\n"
            "   - Session executive summary\n"
            "   - Total orders confirmed\n"
            "   - Total revenue\n"
            "   - Path to the saved report file\n"
        ),
        expected_output=(
            "A final session report containing an executive summary, order count, "
            "total revenue, and the file path of the saved report."
        ),
        agent=agent,
    )

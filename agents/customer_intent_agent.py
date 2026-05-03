"""
agents/customer_intent_agent.py
--------------------------------
Student 1 – Agent Design (Enhanced)
Agent: CustomerIntentAgent

Responsible for receiving raw customer input and producing a comprehensive,
structured intent payload that downstream agents can act on.

Enhancement Summary
-------------------
  • Multi-stage NLP pipeline with sentiment & urgency analysis
  • Two specialised tools: ParseCustomerQuery + AnalyzeQuerySentiment
  • Detailed chain-of-thought task description for richer LLM output
  • Structured classification output with confidence scoring
  • Entity extraction (emails, phone numbers, order IDs)
  • Multi-item detection for complex orders
  • Category inference for product routing

Persona
-------
A meticulous, senior customer-service analyst who never assumes – if the
intent is ambiguous the agent flags it clearly rather than guessing. The
agent also evaluates customer sentiment and urgency so that downstream
agents can prioritise appropriately.
"""

from crewai import Agent, Task
from crewai.tools import tool as crewai_tool
from observability import tracer
from tools.parse_customer_query import parse_customer_query as _parse


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 1: ParseCustomerQuery – Full intent extraction
# ═══════════════════════════════════════════════════════════════════════════════
@crewai_tool("ParseCustomerQuery")
def parse_query_tool(raw_query: str) -> str:
    """
    Parse a raw customer query through a multi-stage NLP pipeline.
    Extracts: action intent, product name, quantity, confidence score,
    sentiment polarity, urgency flags, and detected entities (emails,
    phones, order IDs).

    Returns a JSON string with all extracted fields.
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 2: AnalyzeQuerySentiment – Focused sentiment + urgency analysis
# ═══════════════════════════════════════════════════════════════════════════════
@crewai_tool("AnalyzeQuerySentiment")
def analyze_sentiment_tool(raw_query: str) -> str:
    """
    Perform a focused sentiment and urgency analysis on a customer query.
    Returns sentiment label (positive/neutral/negative), sentiment score,
    urgency flag, matched urgency keywords, and word count.

    Use this tool when you need a second-pass analysis on customer tone
    and urgency, especially for complaints or return requests.
    """
    import json
    result = _parse(raw_query)
    # Extract only sentiment/urgency fields for focused output
    analysis = {
        "sentiment":         result["sentiment"],
        "sentiment_score":   result["sentiment_score"],
        "is_urgent":         result["is_urgent"],
        "urgency_keywords":  result["urgency_keywords"],
        "word_count":        result["word_count"],
        "detected_entities": result["detected_entities"],
        "analysis_note":     _build_sentiment_note(result),
    }
    tracer.log_tool_call(
        "analyze_query_sentiment",
        {"raw_query": raw_query},
        analysis,
        agent_name="CustomerIntentAgent",
    )
    return json.dumps(analysis, indent=2)


def _build_sentiment_note(result: dict) -> str:
    """Generate a human-readable sentiment/urgency summary note."""
    parts = []
    if result["sentiment"] == "negative":
        parts.append("⚠️ Customer appears dissatisfied – handle with care.")
    elif result["sentiment"] == "positive":
        parts.append("😊 Customer has positive sentiment.")
    else:
        parts.append("Customer sentiment is neutral.")

    if result["is_urgent"]:
        parts.append(f"🚨 URGENT request detected (keywords: {', '.join(result['urgency_keywords'])}).")

    if result["detected_entities"].get("order_ids"):
        parts.append(f"📋 References existing order(s): {', '.join(result['detected_entities']['order_ids'])}.")

    if result["multi_item"]:
        parts.append(f"📦 Multi-item request detected ({len(result['items'])} products).")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════
def build_customer_intent_agent(llm) -> Agent:
    """
    Construct and return the enhanced CustomerIntentAgent.

    This agent uses two specialised tools:
      1. ParseCustomerQuery   – Full multi-stage NLP intent extraction
      2. AnalyzeQuerySentiment – Focused sentiment & urgency analysis

    The agent is designed to:
      • Accurately classify the customer's primary action
      • Extract product names and quantities
      • Evaluate customer sentiment for downstream prioritisation
      • Detect urgency for expedited processing
      • Flag ambiguities rather than guessing
      • Support multi-item orders

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
            "Accurately interpret the customer's request using a multi-stage "
            "NLP analysis pipeline. Extract ALL of the following:\n"
            "  (1) The primary action intent (buy/return/track/cancel/complain/enquire)\n"
            "  (2) The exact product name(s) mentioned\n"
            "  (3) The quantity requested for each product\n"
            "  (4) Customer sentiment (positive/neutral/negative)\n"
            "  (5) Urgency level and any time-sensitive keywords\n"
            "  (6) Any detected entities (emails, phone numbers, order IDs)\n"
            "  (7) Confidence score for the overall classification\n"
            "Produce a comprehensive, structured analysis report to hand off "
            "to the Inventory Agent."
        ),
        backstory=(
            "You are a senior customer-experience analyst at ShopBot Inc. with "
            "5+ years of experience processing thousands of customer interactions. "
            "You are known for your meticulous attention to detail and your ability "
            "to read between the lines of customer messages.\n\n"
            "Your analysis methodology follows a strict multi-stage pipeline:\n"
            "  Stage 1: Use ParseCustomerQuery to extract structured intent data\n"
            "  Stage 2: Use AnalyzeQuerySentiment for sentiment and urgency scoring\n"
            "  Stage 3: Synthesize both analyses into a comprehensive report\n\n"
            "You pride yourself on:\n"
            "  • Never misinterpreting customer intent\n"
            "  • Always flagging ambiguity rather than guessing\n"
            "  • Detecting frustrated or urgent customers so they can be prioritised\n"
            "  • Identifying multi-item orders correctly\n"
            "  • Providing confidence scores so downstream agents know how reliable "
            "the extraction is\n\n"
            "Your output MUST include ALL extracted fields (action, product, quantity, "
            "sentiment, urgency, confidence) so downstream agents can proceed "
            "without asking follow-up questions."
        ),
        tools=[parse_query_tool, analyze_sentiment_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════
def build_customer_intent_task(agent: Agent, customer_query: str) -> Task:
    """
    Build the Task for the CustomerIntentAgent.

    The task description guides the LLM through a structured analysis
    workflow to produce a comprehensive, multi-dimensional intent report.

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
            "═══════════════════════════════════════════════════════\n"
            "  ANALYSIS WORKFLOW (follow these steps in order)\n"
            "═══════════════════════════════════════════════════════\n\n"
            "STEP 1 — INTENT EXTRACTION\n"
            "  Use the ParseCustomerQuery tool with the exact customer query.\n"
            "  This will return structured data including: action, product_name,\n"
            "  quantity, confidence_score, sentiment, urgency, and entities.\n\n"
            "STEP 2 — SENTIMENT & URGENCY ANALYSIS\n"
            "  Use the AnalyzeQuerySentiment tool with the same query to get\n"
            "  a focused sentiment breakdown and urgency assessment.\n\n"
            "STEP 3 — SYNTHESIZE FINAL REPORT\n"
            "  Combine the results from both tools into a structured summary:\n\n"
            "  • ACTION CLASSIFICATION\n"
            "    - Primary action: buy / return / track / cancel / complain / enquire\n"
            "    - Matched keyword that triggered the classification\n\n"
            "  • PRODUCT IDENTIFICATION\n"
            "    - Product name (as extracted from the query)\n"
            "    - Inferred product category\n"
            "    - Multi-item flag (if multiple products detected)\n\n"
            "  • QUANTITY & ORDER DETAILS\n"
            "    - Quantity requested (default 1 if not specified)\n"
            "    - Any special notes or customer requirements\n\n"
            "  • CUSTOMER SENTIMENT\n"
            "    - Sentiment: positive / neutral / negative\n"
            "    - Sentiment score (-1.0 to +1.0)\n"
            "    - Urgency flag and matched keywords\n\n"
            "  • CONFIDENCE ASSESSMENT\n"
            "    - Overall confidence: high / medium / low\n"
            "    - Confidence score (0.0 to 1.0)\n"
            "    - Any ambiguities or uncertainties detected\n\n"
            "  • DETECTED ENTITIES\n"
            "    - Email addresses, phone numbers, existing order IDs\n\n"
            "Your output will be passed directly to the Inventory Agent."
        ),
        expected_output=(
            "A comprehensive structured intent analysis report containing:\n"
            "1. Action classification with confidence\n"
            "2. Product name and inferred category\n"
            "3. Quantity requested\n"
            "4. Customer sentiment analysis (label + score)\n"
            "5. Urgency assessment\n"
            "6. Detected entities (emails, phones, order IDs)\n"
            "7. Any ambiguity flags or special notes\n"
            "8. Overall confidence score"
        ),
        agent=agent,
    )

"""
observability.py
----------------
AgentOps-style tracing logger for the E-Commerce MAS.

Writes structured JSONL trace records to logs/agent_traces.jsonl so every
agent invocation, tool call, and result can be audited offline.

Usage
-----
    from observability import tracer

    tracer.log_agent_start("CustomerIntentAgent", input_text)
    tracer.log_tool_call("parse_customer_query", args, result)
    tracer.log_agent_end("CustomerIntentAgent", output_text)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import colorlog  # pip install colorlog

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(__file__)
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
TRACE_FILE = os.path.join(LOGS_DIR, "agent_traces.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Coloured console logger
# ---------------------------------------------------------------------------
_handler = colorlog.StreamHandler()
_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s %(cyan)s%(name)s%(reset)s - %(message)s",
    log_colors={
        "DEBUG":    "white",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    }
))

console_logger = colorlog.getLogger("MAS")
console_logger.addHandler(_handler)
console_logger.setLevel(logging.DEBUG)
console_logger.propagate = False


# ---------------------------------------------------------------------------
# JSONL trace writer
# ---------------------------------------------------------------------------
class AgentTracer:
    """Records structured trace events for every agent action."""

    def _write(self, record: dict[str, Any]) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(TRACE_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    def log_agent_start(self, agent_name: str, input_data: Any) -> None:
        """Record the start of an agent run."""
        msg = f"▶  {agent_name} started"
        console_logger.info(msg)
        self._write({
            "event":      "agent_start",
            "agent":      agent_name,
            "input":      str(input_data)[:500],   # truncate huge payloads
        })

    def log_agent_end(self, agent_name: str, output_data: Any) -> None:
        """Record the successful completion of an agent run."""
        msg = f"✔  {agent_name} finished"
        console_logger.info(msg)
        self._write({
            "event":      "agent_end",
            "agent":      agent_name,
            "output":     str(output_data)[:500],
        })

    def log_tool_call(
        self,
        tool_name:  str,
        arguments:  dict[str, Any],
        result:     Any,
        agent_name: str = "unknown",
    ) -> None:
        """Record a tool invocation together with its result."""
        console_logger.debug(f"🔧 Tool '{tool_name}' called by {agent_name}")
        self._write({
            "event":     "tool_call",
            "agent":     agent_name,
            "tool":      tool_name,
            "arguments": arguments,
            "result":    str(result)[:500],
        })

    def log_error(self, agent_name: str, error: Exception) -> None:
        """Record an unexpected error during agent execution."""
        console_logger.error(f"✖  {agent_name} error: {error}")
        self._write({
            "event":   "error",
            "agent":   agent_name,
            "error":   str(error),
        })

    def log_state_transition(
        self,
        from_agent: str,
        to_agent:   str,
        state_keys: list[str],
    ) -> None:
        """Record a state handoff between agents."""
        console_logger.info(f"↪  State: {from_agent} → {to_agent}  keys={state_keys}")
        self._write({
            "event":       "state_transition",
            "from_agent":  from_agent,
            "to_agent":    to_agent,
            "state_keys":  state_keys,
        })


# Singleton used everywhere
tracer = AgentTracer()

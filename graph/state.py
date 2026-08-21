"""state.py -- the shared object every agent node reads/writes.
LangGraph merges each node's partial-dict return into this state."""
from __future__ import annotations

from typing import Any, TypedDict


class EDAState(TypedDict, total=False):
    user_query: str
    schema: dict[str, Any]
    execution_plan: list[str]
    completed_tasks: list[str]
    retry_count: int
    status: str
    tool_outputs: dict[str, Any]
    statistical_results: dict[str, Any]
    visualizations: list[dict[str, Any]]
    anomalies: dict[str, Any]
    insights: list[dict[str, Any]]
    critic_feedback: dict[str, Any]
    final_report: str

"""critic_agent.py -- deterministic verification (spec: safety net
shouldn't itself depend on an LLM). Checks the shape of the evidence,
not the numbers themselves: were planned tasks completed, is sample
size adequate, is there unsupported causal language in the insights."""
from __future__ import annotations

import re

from graph.state import EDAState

CAUSAL_PATTERNS = re.compile(r"\bcauses?\b|\bcausing\b|\bdue to\b|\bbecause of\b", re.IGNORECASE)
MIN_SAMPLE_SIZE = 30


def critic_agent(state: EDAState) -> dict:
    issues: list[str] = []
    missing_analysis: list[str] = []

    plan = state.get("execution_plan", [])
    completed = state.get("completed_tasks", [])
    for task in plan:
        if task not in completed:
            issues.append(f"Planned task '{task}' did not complete.")
            missing_analysis.append(task)

    has_evidence = bool(
        state.get("tool_outputs") or state.get("statistical_results") or state.get("anomalies")
    )
    if not has_evidence:
        issues.append("No tool outputs were produced.")

    row_count = state.get("tool_outputs", {}).get("profiling", {}).get("profile", {}).get("row_count")
    if row_count is not None and row_count < MIN_SAMPLE_SIZE:
        issues.append(f"Dataset has only {row_count} rows (< {MIN_SAMPLE_SIZE}); statistical power is low.")

    for finding in state.get("insights", []):
        text = f"{finding.get('fact', '')} {finding.get('interpretation', '')}"
        if CAUSAL_PATTERNS.search(text):
            issues.append(f"Possible unsupported causal claim: '{text.strip()[:100]}...'")

    if not plan:
        issues.append("Execution plan is empty.")
        missing_analysis.append("profiling")

    return {
        "critic_feedback": {"passed": len(issues) == 0, "issues": issues, "missing_analysis": missing_analysis},
        "status": "critiquing",
    }

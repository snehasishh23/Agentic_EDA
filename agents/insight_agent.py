"""insight_agent.py -- LLM turns tool-computed numbers into plain-
language findings, explicitly separating fact from interpretation and
forbidding causal language from correlation alone. Falls back to a
deterministic template if the LLM is unavailable or returns bad JSON."""
from __future__ import annotations

import json

from agents.llm import call_llm
from graph.state import EDAState

SYSTEM_PROMPT = """You are the Insight Agent in a data-analysis pipeline.
You are given ONLY tool-computed statistics -- do not invent numbers.
For each finding, separate "fact" (the literal computed result) from
"interpretation" (what it plausibly means). Never claim causation from
correlation or regression alone -- use "associated with" language.
Return ONLY a JSON list: [{"fact": "...", "interpretation": "...", "source": "..."}]"""


def _deterministic_findings(state: EDAState) -> list[dict]:
    findings = []
    profiling = state.get("tool_outputs", {}).get("profiling", {})
    for issue in profiling.get("data_quality_issues", []):
        findings.append({"fact": issue, "interpretation": "Address before trusting results.", "source": "profiling"})

    corr = state.get("statistical_results", {}).get("correlation", {})
    for pair in corr.get("pairs", [])[:3]:
        findings.append(
            {
                "fact": f"{pair['column_a']} and {pair['column_b']} have r={pair['r']} ({pair['strength']}).",
                "interpretation": f"These variables are {pair['strength']}ly associated; not evidence of causation.",
                "source": "correlation",
            }
        )

    for method, result in state.get("anomalies", {}).items():
        if "outlier_count" in result:
            findings.append(
                {
                    "fact": f"{method} flagged {result['outlier_count']} anomalous rows ({result.get('outlier_pct', 0)}%).",
                    "interpretation": "Warrants manual review -- may be errors or genuine extremes.",
                    "source": f"anomaly_{method}",
                }
            )
    return findings


def insight_agent(state: EDAState) -> dict:
    payload = {
        "profiling": state.get("tool_outputs", {}).get("profiling", {}),
        "statistical_results": state.get("statistical_results", {}),
        "anomalies": state.get("anomalies", {}),
        "user_query": state.get("user_query", ""),
    }
    try:
        raw = call_llm(SYSTEM_PROMPT, json.dumps(payload, default=str))
        findings = json.loads(raw)
        if not isinstance(findings, list) or not findings:
            raise ValueError("empty or non-list")
    except Exception:
        findings = _deterministic_findings(state)

    completed = list(state.get("completed_tasks", []))
    if "insight" not in completed:
        completed.append("insight")
    return {"insights": findings, "completed_tasks": completed}

"""supervisor.py -- builds a dynamic execution plan from the dataset's
schema and the user's question. Rule-based core (always works, even
with no LLM) plus an optional LLM pass to add tasks the rules missed."""
from __future__ import annotations

from agents.llm import safe_json_call
from graph.state import EDAState

ALL_TASKS = [
    "profiling",
    "correlation",
    "groupby",
    "time_series",
    "regression",
    "visualization",
    "anomaly_iqr",
    "anomaly_zscore",
    "anomaly_isolation_forest",
]


def _rule_based_plan(schema: dict, user_query: str) -> list[str]:
    plan = ["profiling"]
    q = user_query.lower()
    numeric = schema.get("numeric_columns", [])
    categorical = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])

    if len(numeric) >= 2:
        plan.append("correlation")
    if categorical and numeric:
        plan.append("groupby")
    if datetime_cols and numeric:
        plan.append("time_series")
    if schema.get("likely_target_column") and len(numeric) >= 2 and any(
        k in q for k in ("drive", "factor", "cause", "predict", "influence")
    ):
        plan.append("regression")
    if numeric or datetime_cols:
        plan.append("visualization")

    if any(k in q for k in ("anomal", "outlier", "unusual")):
        if numeric:
            plan += ["anomaly_iqr", "anomaly_zscore"]
            if len(numeric) >= 2:
                plan.append("anomaly_isolation_forest")
    elif numeric:
        plan.append("anomaly_iqr")

    seen, ordered = set(), []
    for t in plan:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


def supervisor_agent(state: EDAState) -> dict:
    schema = state.get("schema", {})
    user_query = state.get("user_query", "")
    retry_count = state.get("retry_count", 0)

    if retry_count == 0:
        plan = _rule_based_plan(schema, user_query)
        system = (
            "You are the Supervisor of a data-analysis agent team. "
            f"Valid tasks are exactly: {ALL_TASKS}. Return JSON: "
            '{"additional_tasks": [...]} using only tasks NOT already planned '
            "that would materially help answer the question. Empty list if plan is sufficient."
        )
        user = f"Schema: {schema}\nQuestion: {user_query}\nAlready planned: {plan}"
        result = safe_json_call(system, user, fallback={"additional_tasks": []})
        extra = [t for t in result.get("additional_tasks", []) if t in ALL_TASKS and t not in plan]
        plan.extend(extra)
        return {"execution_plan": plan, "completed_tasks": [], "retry_count": 0, "status": "running"}

    feedback = state.get("critic_feedback", {})
    missing = feedback.get("missing_analysis", [])
    plan = state.get("execution_plan", [])
    plan = plan + [t for t in missing if t in ALL_TASKS and t not in plan]
    return {"execution_plan": plan, "status": "running"}

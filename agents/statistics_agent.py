"""statistics_agent.py -- pure computation, no LLM."""
from __future__ import annotations

import pandas as pd

from graph.state import EDAState
from tools import statistics as stats_tools


def statistics_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    plan = state.get("execution_plan", [])
    numeric_cols = schema.get("numeric_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])
    target = schema.get("likely_target_column")

    results: dict = {}
    if "correlation" in plan and len(numeric_cols) >= 2:
        results["correlation"] = stats_tools.correlation_analysis(df, numeric_cols)
    if "groupby" in plan and categorical_cols and numeric_cols:
        results["groupby"] = stats_tools.groupby_analysis(df, categorical_cols[0], target or numeric_cols[0])
    if "time_series" in plan and datetime_cols and numeric_cols:
        results["time_series"] = stats_tools.time_series_analysis(df, datetime_cols[0], target or numeric_cols[0])
    if "regression" in plan and target and len(numeric_cols) >= 2:
        features = [c for c in numeric_cols if c != target]
        results["regression"] = stats_tools.regression_analysis(df, target, features)

    statistical_results = dict(state.get("statistical_results", {}))
    statistical_results.update(results)
    completed = list(state.get("completed_tasks", []))
    for t in ("correlation", "groupby", "time_series", "regression"):
        if t in results and t not in completed:
            completed.append(t)
    return {"statistical_results": statistical_results, "completed_tasks": completed}

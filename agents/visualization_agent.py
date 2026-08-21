"""visualization_agent.py -- picks a small, relevant set of charts
rather than plotting every column (avoids chart spam)."""
from __future__ import annotations

import pandas as pd

from graph.state import EDAState
from tools import visualization as viz_tools


def visualization_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    plan = state.get("execution_plan", [])
    if "visualization" not in plan:
        return {}

    numeric_cols = schema.get("numeric_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])
    target = schema.get("likely_target_column")

    charts = []
    if numeric_cols:
        charts.append(viz_tools.create_histogram(df, target or numeric_cols[0]))
    if categorical_cols and numeric_cols:
        charts.append(viz_tools.create_boxplot(df, target or numeric_cols[0], categorical_cols[0]))
    if len(numeric_cols) >= 2:
        charts.append(viz_tools.create_heatmap(df, numeric_cols))

    corr_result = state.get("statistical_results", {}).get("correlation")
    if corr_result and corr_result.get("pairs"):
        top = corr_result["pairs"][0]
        charts.append(viz_tools.create_scatterplot(df, top["column_a"], top["column_b"]))
    elif len(numeric_cols) >= 2:
        charts.append(viz_tools.create_scatterplot(df, numeric_cols[0], numeric_cols[1]))

    if datetime_cols and numeric_cols:
        charts.append(viz_tools.create_time_series_plot(df, datetime_cols[0], target or numeric_cols[0]))

    charts = [c for c in charts if "error" not in c]
    existing = list(state.get("visualizations", []))
    existing.extend(charts)
    completed = list(state.get("completed_tasks", []))
    if "visualization" not in completed:
        completed.append("visualization")
    return {"visualizations": existing, "completed_tasks": completed}

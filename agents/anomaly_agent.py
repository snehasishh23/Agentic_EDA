"""anomaly_agent.py -- runs whichever anomaly-detection methods were
planned. Pure computation, no LLM."""
from __future__ import annotations

import pandas as pd

from graph.state import EDAState
from tools import anomaly as anomaly_tools


def anomaly_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    plan = state.get("execution_plan", [])
    numeric_cols = schema.get("numeric_columns", [])
    target = schema.get("likely_target_column")
    focus_col = target or (numeric_cols[0] if numeric_cols else None)

    results: dict = {}
    completed = list(state.get("completed_tasks", []))

    if "anomaly_iqr" in plan and focus_col:
        results["iqr"] = anomaly_tools.detect_iqr_outliers(df, focus_col)
        completed.append("anomaly_iqr")
    if "anomaly_zscore" in plan and focus_col:
        results["zscore"] = anomaly_tools.detect_zscore_outliers(df, focus_col)
        completed.append("anomaly_zscore")
    if "anomaly_isolation_forest" in plan and len(numeric_cols) >= 2:
        results["isolation_forest"] = anomaly_tools.detect_isolation_forest_anomalies(df, numeric_cols)
        completed.append("anomaly_isolation_forest")

    anomalies = dict(state.get("anomalies", {}))
    anomalies.update(results)
    return {"anomalies": anomalies, "completed_tasks": completed}

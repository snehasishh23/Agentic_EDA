"""profiling_agent.py -- pure computation, no LLM."""
from __future__ import annotations

import pandas as pd

from graph.state import EDAState
from tools import profiling


def profiling_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    numeric_cols = schema.get("numeric_columns", [])
    result = {
        "profile": profiling.profile_dataset(df),
        "missing_values": profiling.missing_value_report(df),
        "duplicates": profiling.duplicate_report(df),
        "descriptive_statistics": profiling.descriptive_statistics(df, numeric_cols),
        "data_quality_issues": profiling.data_quality_issues(df, schema),
    }
    tool_outputs = dict(state.get("tool_outputs", {}))
    tool_outputs["profiling"] = result
    completed = list(state.get("completed_tasks", [])) + ["profiling"]
    return {"tool_outputs": tool_outputs, "completed_tasks": completed}

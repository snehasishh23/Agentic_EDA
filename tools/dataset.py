"""dataset.py -- schema inspection. Operates purely in-memory on a
DataFrame Streamlit already loaded; no disk I/O and no path handling
needed since this app never persists an uploaded file."""
from __future__ import annotations

from typing import Any

import pandas as pd


def inspect_schema(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols, categorical_cols, datetime_cols, boolean_cols, text_cols = [], [], [], [], []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        else:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            parse_rate = parsed.notna().mean() if len(series) else 0
            looks_like_date = any(k in col.lower() for k in ("date", "time", "timestamp"))
            if parse_rate > 0.8 and (looks_like_date or parse_rate > 0.95):
                datetime_cols.append(col)
            else:
                nunique = series.nunique(dropna=True)
                if nunique <= max(50, int(0.05 * len(series))):
                    categorical_cols.append(col)
                else:
                    text_cols.append(col)

    likely_target = None
    for candidate in ("revenue", "sales", "target", "label", "price", "amount", "profit"):
        for col in numeric_cols:
            if candidate in col.lower():
                likely_target = col
                break
        if likely_target:
            break

    return {
        "columns": list(df.columns),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "text_columns": text_cols,
        "likely_target_column": likely_target,
    }

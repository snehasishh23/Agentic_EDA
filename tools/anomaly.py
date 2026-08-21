"""anomaly.py -- IQR, Z-score, and Isolation Forest outlier detection."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_iqr_outliers(df: pd.DataFrame, column: str, k: float = 1.5) -> dict[str, Any]:
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    series = df[column].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    idx = df[mask].index.tolist()
    return {
        "method": "iqr",
        "column": column,
        "outlier_count": len(idx),
        "outlier_pct": round(len(idx) / len(df) * 100, 2) if len(df) else 0.0,
        "outlier_indices": idx[:50],
        "explanation": f"Values outside [{round(float(lower),2)}, {round(float(upper),2)}] (1.5x IQR) are flagged.",
    }


def detect_zscore_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> dict[str, Any]:
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    series = df[column].dropna()
    if series.std() == 0 or series.empty:
        return {"method": "zscore", "column": column, "outlier_count": 0, "outlier_indices": []}
    z = (series - series.mean()) / series.std()
    mask = z.abs() > threshold
    idx = series[mask].index.tolist()
    return {
        "method": "zscore",
        "column": column,
        "outlier_count": len(idx),
        "outlier_pct": round(len(idx) / len(df) * 100, 2) if len(df) else 0.0,
        "outlier_indices": idx[:50],
        "explanation": f"Values more than {threshold} standard deviations from the mean are flagged.",
    }


def detect_isolation_forest_anomalies(
    df: pd.DataFrame, numeric_columns: list[str], contamination: float = 0.02
) -> dict[str, Any]:
    cols = [c for c in numeric_columns if c in df.columns]
    if len(cols) < 2:
        return {"error": "Isolation Forest needs at least 2 numeric columns."}
    sub = df[cols].dropna()
    if len(sub) < 20:
        return {"error": "Not enough rows for a reliable fit."}
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(sub.values)
    idx = sub.index[preds == -1].tolist()
    return {
        "method": "isolation_forest",
        "columns_used": cols,
        "outlier_count": len(idx),
        "outlier_pct": round(len(idx) / len(df) * 100, 2) if len(df) else 0.0,
        "outlier_indices": idx[:50],
        "explanation": f"Rows flagged as anomalous based on a combination of {', '.join(cols)}.",
    }

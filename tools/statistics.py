"""statistics.py -- correlation, group comparison, trend, and regression
tools. Reports association, never causation, in every note field."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def correlation_analysis(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    cols = [c for c in numeric_columns if c in df.columns]
    if len(cols) < 2:
        return {"pairs": []}
    sub = df[cols].dropna()
    corr_matrix = sub.corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr_matrix.loc[a, b]
            if pd.isna(r):
                continue
            _, p = scipy_stats.pearsonr(sub[a], sub[b]) if len(sub) > 2 else (None, None)
            pairs.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "r": round(float(r), 4),
                    "p_value": round(float(p), 6) if p is not None else None,
                    "strength": _strength_label(r),
                }
            )
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {"pairs": pairs, "note": "Correlation indicates association, not causation."}


def _strength_label(r: float) -> str:
    a = abs(r)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


def groupby_analysis(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    if group_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing column(s): {group_col}, {value_col}"}
    grouped = df.groupby(group_col)[value_col].agg(["count", "mean", "median"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)
    return {"group_col": group_col, "value_col": value_col, "groups": grouped.round(4).to_dict(orient="records")}


def time_series_analysis(df: pd.DataFrame, date_col: str, value_col: str) -> dict[str, Any]:
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing column(s): {date_col}, {value_col}"}
    sub = df[[date_col, value_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce", format="mixed")
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)
    if sub.empty:
        return {"error": "No parseable dates."}
    monthly = sub.set_index(date_col)[value_col].resample("ME").mean().dropna()
    if len(monthly) < 2:
        return {"trend": "insufficient_data", "date_col": date_col, "value_col": value_col}
    slope = np.polyfit(range(len(monthly)), monthly.values, 1)[0]
    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    pct_change = (
        round(float((monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0]) * 100), 2)
        if monthly.iloc[0] != 0
        else None
    )
    return {
        "date_col": date_col,
        "value_col": value_col,
        "trend": trend,
        "pct_change_start_to_end": pct_change,
    }


def regression_analysis(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict[str, Any]:
    features = [c for c in feature_cols if c in df.columns and c != target_col]
    if target_col not in df.columns or not features:
        return {"error": "Target or feature columns not found."}
    sub = df[[target_col] + features].dropna()
    if len(sub) < max(10, len(features) + 2):
        return {"error": "Not enough rows for a reliable regression."}
    X, y = sub[features].values, sub[target_col].values
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return {
        "target_col": target_col,
        "feature_cols": features,
        "r_squared": round(float(r2), 4),
        "coefficients": {f: round(float(c), 4) for f, c in zip(features, model.coef_)},
        "note": "Coefficients describe linear association, not proven causal effect.",
    }

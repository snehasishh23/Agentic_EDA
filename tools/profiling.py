"""profiling.py -- deterministic profiling/data-quality tools. No LLM
involvement -- every number here is computed, never generated."""
from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missing_value_report(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    missing = df.isna().sum()
    report = {
        col: {
            "missing_count": int(missing[col]),
            "missing_pct": round(float(missing[col]) / total * 100, 2) if total else 0.0,
        }
        for col in df.columns
        if missing[col] > 0
    }
    return {"columns_with_missing": report, "any_missing": bool(missing.sum() > 0)}


def duplicate_report(df: pd.DataFrame) -> dict[str, Any]:
    dup_mask = df.duplicated(keep="first")
    return {
        "duplicate_row_count": int(dup_mask.sum()),
        "duplicate_pct": round(float(dup_mask.mean()) * 100, 2) if len(df) else 0.0,
    }


def descriptive_statistics(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4) if series.count() > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "max": round(float(series.max()), 4),
        }
    return stats


def data_quality_issues(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = missing_value_report(df)
    if missing["any_missing"]:
        worst = max(missing["columns_with_missing"].items(), key=lambda kv: kv[1]["missing_pct"])
        issues.append(f"Column '{worst[0]}' has {worst[1]['missing_pct']}% missing values.")
    dup = duplicate_report(df)
    if dup["duplicate_row_count"] > 0:
        issues.append(f"{dup['duplicate_row_count']} duplicate rows detected ({dup['duplicate_pct']}%).")
    if df.shape[0] < 30:
        issues.append("Dataset has fewer than 30 rows -- statistical tests will have low power.")
    return issues

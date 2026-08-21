"""visualization.py -- chart tools that return matplotlib Figure objects
directly (rendered in-app via st.pyplot). Nothing is ever saved to
disk -- charts live only for the duration of the Streamlit session."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_histogram(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    fig, ax = plt.subplots(figsize=(6, 4))
    df[column].dropna().plot(kind="hist", bins=30, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    fig.tight_layout()
    return {"chart_type": "histogram", "figure": fig, "description": f"Histogram of {column}."}


def create_boxplot(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> dict[str, Any]:
    if value_col not in df.columns:
        return {"error": f"Column not found: {value_col}"}
    fig, ax = plt.subplots(figsize=(6, 4))
    if group_col and group_col in df.columns:
        df.boxplot(column=value_col, by=group_col, ax=ax)
        ax.set_title(f"{value_col} by {group_col}")
        plt.suptitle("")
    else:
        df.boxplot(column=value_col, ax=ax)
        ax.set_title(f"Boxplot of {value_col}")
    fig.tight_layout()
    return {
        "chart_type": "boxplot",
        "figure": fig,
        "description": f"Boxplot of {value_col}" + (f" by {group_col}." if group_col else "."),
    }


def create_scatterplot(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": f"Missing column(s): {x_col}, {y_col}"}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x_col], df[y_col], alpha=0.5, color="#DD8452")
    ax.set_title(f"{y_col} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return {"chart_type": "scatterplot", "figure": fig, "description": f"{y_col} vs {x_col}."}


def create_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    cols = [c for c in numeric_columns if c in df.columns]
    if len(cols) < 2:
        return {"error": "Need at least 2 numeric columns."}
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(1 + len(cols) * 0.8, 1 + len(cols) * 0.8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return {"chart_type": "heatmap", "figure": fig, "description": "Correlation heatmap."}


def create_time_series_plot(df: pd.DataFrame, date_col: str, value_col: str) -> dict[str, Any]:
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing column(s): {date_col}, {value_col}"}
    sub = df[[date_col, value_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce", format="mixed")
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)
    if sub.empty:
        return {"error": "No parseable dates."}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sub[date_col], sub[value_col], color="#55A868")
    ax.set_title(f"{value_col} over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    return {"chart_type": "time_series", "figure": fig, "description": f"{value_col} over {date_col}."}

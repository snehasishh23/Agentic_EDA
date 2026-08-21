"""report_agent.py -- assembles the final markdown report. Every number
quoted comes directly from state (tool_outputs/statistical_results/
anomalies) -- the LLM only writes the Executive Summary prose, so
figures can never drift from what was actually computed."""
from __future__ import annotations

from datetime import datetime, timezone

from agents.llm import safe_text_call
from graph.state import EDAState

SUMMARY_SYSTEM_PROMPT = """You write the Executive Summary for a data analysis report.
You will be given the user's question and a JSON dump of verified findings.
Write 3-5 sentences, plain language, no invented numbers -- only reference
figures present in the JSON. Do not use causal language for anything that
came from a correlation or regression. Return prose only."""


def _fallback_summary(state: EDAState) -> str:
    n_findings = len(state.get("insights", []))
    n_charts = len(state.get("visualizations", []))
    row_count = state.get("tool_outputs", {}).get("profiling", {}).get("profile", {}).get("row_count", "unknown")
    return (
        f"This report analyzes a dataset of {row_count} rows in response to: "
        f"\"{state.get('user_query', '')}\". The analysis produced {n_findings} findings "
        f"and {n_charts} supporting charts. See sections below for details."
    )


def _section_data_quality(state: EDAState) -> str:
    issues = state.get("tool_outputs", {}).get("profiling", {}).get("data_quality_issues", [])
    return "\n".join(f"- {i}" for i in issues) if issues else "No significant data-quality issues detected."


def _section_statistics(state: EDAState) -> str:
    lines = []
    corr = state.get("statistical_results", {}).get("correlation", {})
    for pair in corr.get("pairs", [])[:5]:
        lines.append(f"- **{pair['column_a']} vs {pair['column_b']}**: r = {pair['r']} ({pair['strength']})")
    regression = state.get("statistical_results", {}).get("regression")
    if regression and "error" not in regression:
        lines.append(
            f"- Regression on **{regression['target_col']}**: R² = {regression['r_squared']}, "
            f"coefficients: {regression['coefficients']}"
        )
    return "\n".join(lines) if lines else "No statistical tests were applicable to this question."


def _section_trends(state: EDAState) -> str:
    ts = state.get("statistical_results", {}).get("time_series")
    if not ts or "error" in ts:
        return "No time-series trend analysis was applicable."
    pct = ts.get("pct_change_start_to_end")
    return f"- Trend for **{ts.get('value_col')}**: **{ts.get('trend')}**" + (f", {pct}% change." if pct is not None else ".")


def _section_anomalies(state: EDAState) -> str:
    anomalies = state.get("anomalies", {})
    lines = []
    for method, result in anomalies.items():
        if "outlier_count" in result:
            lines.append(f"- **{method}**: {result['outlier_count']} anomalies ({result.get('outlier_pct', 0)}%). {result.get('explanation', '')}")
    return "\n".join(lines) if lines else "No anomaly detection was run."


def _section_insights(state: EDAState) -> str:
    findings = state.get("insights", [])
    if not findings:
        return "No verified insights available."
    return "\n".join(f"- **Fact:** {f.get('fact','')}\n  **Interpretation:** {f.get('interpretation','')}" for f in findings)


def report_agent(state: EDAState) -> dict:
    summary = safe_text_call(
        SUMMARY_SYSTEM_PROMPT,
        f"Question: {state.get('user_query')}\nFindings: {state.get('insights', [])}",
        fallback=_fallback_summary(state),
    )

    schema = state.get("schema", {})
    profile = state.get("tool_outputs", {}).get("profiling", {}).get("profile", {})

    report = f"""# Autonomous EDA Report

*Generated {datetime.now(timezone.utc).isoformat()}*

**Question:** {state.get('user_query', '')}

## 1. Executive Summary
{summary}

## 2. Dataset Overview
- Rows: {profile.get('row_count', 'n/a')} | Columns: {profile.get('column_count', 'n/a')}
- Numeric: {', '.join(schema.get('numeric_columns', [])) or 'none'}
- Categorical: {', '.join(schema.get('categorical_columns', [])) or 'none'}
- Datetime: {', '.join(schema.get('datetime_columns', [])) or 'none'}

## 3. Data Quality
{_section_data_quality(state)}

## 4. Key Statistical Findings
{_section_statistics(state)}

## 5. Trends
{_section_trends(state)}

## 6. Anomaly / Outlier Findings
{_section_anomalies(state)}

## 7. Business Insights
{_section_insights(state)}

## 8. Limitations
- Findings are based on this dataset only and may not generalize.
- Correlational/regression results describe association, not proven causation.

## 9. Recommended Next Steps
- Review flagged data-quality issues before acting on this report.
- Validate top associations with domain knowledge before assuming causality.
- Investigate flagged anomalies individually.
"""
    completed = list(state.get("completed_tasks", []))
    if "report" not in completed:
        completed.append("report")
    return {"final_report": report, "completed_tasks": completed, "status": "done"}

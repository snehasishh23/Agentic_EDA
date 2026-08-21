"""
workflow.py -- wires every agent into a LangGraph StateGraph.

    START -> dataset_inspector -> supervisor -> profiling -> statistics
          -> visualization -> anomaly -> insight -> critic
          -> [conditional] -> report -> END
                            -> replan -> supervisor  (loop, capped)

A DataFrame is bound to the tool-calling nodes via functools.partial at
graph-build time rather than passed through state, since it's not JSON-
serializable and this app never checkpoints state to disk anyway.
"""
from __future__ import annotations

from functools import partial

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.anomaly_agent import anomaly_agent
from agents.critic_agent import critic_agent
from agents.insight_agent import insight_agent
from agents.profiling_agent import profiling_agent
from agents.report_agent import report_agent
from agents.statistics_agent import statistics_agent
from agents.supervisor import supervisor_agent
from agents.visualization_agent import visualization_agent
from graph.router import route_after_critic
from graph.state import EDAState
from tools import dataset as dataset_tools
from tools import profiling as profiling_tools


def _dataset_inspector_node(state: EDAState, df: pd.DataFrame) -> dict:
    return {"schema": dataset_tools.inspect_schema(df)}


def _replan_node(state: EDAState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_workflow(df: pd.DataFrame):
    graph = StateGraph(EDAState)

    graph.add_node("dataset_inspector", partial(_dataset_inspector_node, df=df))
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("profiling", partial(profiling_agent, df=df))
    graph.add_node("statistics", partial(statistics_agent, df=df))
    graph.add_node("visualization", partial(visualization_agent, df=df))
    graph.add_node("anomaly", partial(anomaly_agent, df=df))
    graph.add_node("insight", insight_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("replan", _replan_node)
    graph.add_node("report", report_agent)

    graph.set_entry_point("dataset_inspector")
    graph.add_edge("dataset_inspector", "supervisor")
    graph.add_edge("supervisor", "profiling")
    graph.add_edge("profiling", "statistics")
    graph.add_edge("statistics", "visualization")
    graph.add_edge("visualization", "anomaly")
    graph.add_edge("anomaly", "insight")
    graph.add_edge("insight", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"report": "report", "replan": "replan"})
    graph.add_edge("replan", "supervisor")
    graph.add_edge("report", END)

    return graph.compile()


def run_eda(df: pd.DataFrame, user_query: str) -> EDAState:
    app = build_workflow(df)
    initial_state: EDAState = {
        "user_query": user_query,
        "retry_count": 0,
        "completed_tasks": [],
        "tool_outputs": {},
        "statistical_results": {},
        "visualizations": [],
        "anomalies": {},
        "insights": [],
        "status": "planning",
    }
    return app.invoke(initial_state, config={"recursion_limit": 50})

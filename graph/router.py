"""router.py -- conditional edge logic: pass to report, or loop back
through the Supervisor to re-plan, capped by MAX_RETRIES."""
from __future__ import annotations

import os

from graph.state import EDAState

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def route_after_critic(state: EDAState) -> str:
    feedback = state.get("critic_feedback", {})
    if feedback.get("passed", False):
        return "report"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "report"
    return "replan"

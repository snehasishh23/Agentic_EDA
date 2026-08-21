# Agentic EDA -- Simple Edition

A single-file Streamlit app that turns a CSV and a plain-English
question into a verified, evidence-backed EDA report -- using a real
multi-agent LangGraph pipeline (not just one LLM call). No FastAPI, no
database, nothing written to disk. Good for a quick resume demo without
the infrastructure overhead of the full version.

## How it works

```
CSV upload -> dataset_inspector (schema detection)
           -> supervisor (LLM-assisted, rule-based dynamic plan)
           -> profiling -> statistics -> visualization -> anomaly   (pure computation, no LLM)
           -> insight (LLM turns tool output into findings)
           -> critic (deterministic verification)
           -> [passed?] -> report (LLM writes summary; everything else is templated from state)
           -> [missing evidence?] -> replan -> supervisor (loop, capped by MAX_RETRIES)
```

**Core rule: the LLM plans and writes prose, tools compute.** Every
number in the report -- correlations, p-values, R², outlier counts --
comes from `tools/` (pandas/scipy/scikit-learn), never from the LLM.
The Critic agent re-checks the evidence (were planned tasks completed,
is sample size adequate, is there unsupported causal language) before
the report is allowed through; if not, the Supervisor re-plans and
re-runs, up to `MAX_RETRIES` times.

Everything lives in Streamlit's session state for the life of the
browser tab -- the uploaded CSV is never written to disk, charts are
matplotlib `Figure` objects rendered directly with `st.pyplot()`, and
the report only leaves the app if you click "Download".

## Project layout

```
app.py              single entrypoint -- upload, ask, run, render
tools/               pure, deterministic functions (no LLM), unit-testable
  dataset.py         schema detection
  profiling.py       missing values, duplicates, descriptive stats
  statistics.py      correlation, groupby, time series, regression
  anomaly.py         IQR, Z-score, Isolation Forest
  visualization.py   returns matplotlib Figures directly, nothing saved to disk
agents/              one file per agent
  llm.py             thin wrapper around Ollama, with deterministic fallbacks
  supervisor.py       dynamic plan builder
  profiling_agent.py / statistics_agent.py / visualization_agent.py / anomaly_agent.py
  insight_agent.py    turns tool output into plain-language findings
  critic_agent.py     deterministic verification, no LLM
  report_agent.py     assembles the final markdown report
graph/
  state.py            shared state every agent reads/writes
  router.py           retry/replan routing logic
  workflow.py          LangGraph wiring
```

## Setup

```bash
cd simple
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3:1b     # if you don't already have a model pulled
streamlit run app.py
```

That's it -- one command, one process, no backend to keep running
separately. Open the URL Streamlit prints (usually `localhost:8501`),
upload a CSV, ask a question, click Run Analysis.

## Why this is still a genuine multi-agent system, not "one big prompt"

- Each agent is a separate, independently testable Python function with
  a single responsibility (spec: single-responsibility agents).
- The Supervisor's plan is *dynamic* -- it depends on the actual schema
  and question, not a fixed pipeline. A dataset with no datetime column
  never gets a time-series step; a question with no causal language
  never triggers a regression.
- The Critic agent creates a real feedback loop: if evidence is
  missing, the Supervisor adds tasks and the graph re-runs those
  specific agents, rather than just re-prompting the LLM to "try
  harder."
- Every LLM call has a deterministic fallback, so a small/unreliable
  local model (like `gemma3:1b`) degrades to templated output instead
  of crashing the pipeline -- this separation of "LLM for judgment,
  code for computation" is the main design idea worth explaining in an
  interview.

## Extending it

- Add a new analysis: write the function in the relevant `tools/*.py`
  module, call it from the matching agent, and add its task name to
  `ALL_TASKS` in `agents/supervisor.py` so the planner can select it.
- Swap the model: everything routes through `agents/llm.py` --
  change `OLLAMA_MODEL` in `.env` to point at any Ollama model.

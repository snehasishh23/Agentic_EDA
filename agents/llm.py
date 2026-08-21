"""llm.py -- single point of contact with local Ollama. The LLM is used
ONLY for planning, interpretation, and prose -- never to compute a
number. If Ollama is down or returns bad output, callers fall back to
deterministic templates so the app still works."""
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        format="json" if json_mode else None,
    )
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


def safe_json_call(system_prompt: str, user_prompt: str, fallback: dict) -> dict:
    try:
        raw = call_llm(system_prompt, user_prompt, json_mode=True)
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed, using fallback: %s", exc)
        return fallback


def safe_text_call(system_prompt: str, user_prompt: str, fallback: str) -> str:
    try:
        return call_llm(system_prompt, user_prompt).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed, using fallback: %s", exc)
        return fallback

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("GRANT_DATA_DIR", ROOT / "data"))
PROMPT_DIR = ROOT / "prompts"
CACHE_DIR = Path(os.getenv("GRANT_CACHE_DIR", ROOT / ".cache"))

ANNOUNCEMENTS_PATH = DATA_DIR / "announcements.jsonl"
PROFILES_PATH = DATA_DIR / "profiles.json"
EVAL_SET_PATH = DATA_DIR / "eval_set.jsonl"

EXPERIMENT = "grant-agent"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


def agent_model() -> str:
    return os.getenv("AGENT_MODEL", "claude-sonnet-5-5")

from __future__ import annotations

import json
from datetime import date

from pydantic import BaseModel

from grantagent.config import EVAL_SET_PATH, PROFILES_PATH
from grantagent.kstartup import load_snapshot
from grantagent.schemas import Announcement, CompanyProfile, Verdict


class EvalCase(BaseModel):
    case_id: str
    announcement_id: str
    profile_id: str
    as_of: date
    expected: Verdict
    note: str = ""
    labeled_at: str | None = None


def load_profiles(path=PROFILES_PATH) -> dict[str, CompanyProfile]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {p["id"]: CompanyProfile.model_validate(p) for p in items}


def load_announcements() -> dict[str, Announcement]:
    return load_snapshot()


def load_eval_set(path=EVAL_SET_PATH) -> list[EvalCase]:
    if not path.exists():
        return []
    return [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_eval_case(case: EvalCase, path=EVAL_SET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json() + "\n")

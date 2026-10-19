"""API·데모·피드백 흐름을 가짜 LLM + 로컬 MLflow로 검증."""

import json

import mlflow
import pytest
from conftest import AS_OF
from fastapi.testclient import TestClient
from test_agent import FakeLLM, criteria, make_agent

from grantagent import feedback as fb_mod
from grantagent.api import create_app
from grantagent.data import EvalCase
from grantagent.schemas import JudgeOutput, OtherCheck


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("grant-agent")


@pytest.fixture
def client(ann, young_seoul, old_busan, tmp_path):
    judge = JudgeOutput(checks=[OtherCheck(requirement="x", status="pass", reason="ICT")])
    versions = iter(["1", "2"])

    def factory():
        agent = make_agent(FakeLLM(criteria(), judge), tmp_path)
        v = next(versions)
        agent.extract_prompt = agent.extract_prompt.__class__(
            "extract", "grant-extract", v, agent.extract_prompt.template
        )
        return agent

    app = create_app(
        agent_factory=factory, anns={"T001": ann}, profiles={"y1": young_seoul, "o1": old_busan},
        feedback_path=tmp_path / "feedback.jsonl", tracing_enabled=True,
    )  # fmt: skip
    return TestClient(app)


def test_demo_page_and_lists(client):
    assert "적격성 판정" in client.get("/").text
    assert client.get("/announcements").json()[0]["id"] == "T001"
    assert {p["id"] for p in client.get("/profiles").json()} == {"y1", "o1"}


def test_assess_returns_verdict_with_evidence(client):
    r = client.post("/assess", json={"announcement_id": "T001", "profile_id": "y1",
                                     "as_of": AS_OF.isoformat()}).json()  # fmt: skip
    assert r["verdict"] == "eligible" and r["verdict_ko"] == "지원 가능"
    assert r["trace_id"] and all(c["quote_valid"] for c in r["checks"])
    assert (
        client.post("/assess", json={"announcement_id": "X", "profile_id": "y1"}).status_code == 404
    )


def test_feedback_is_logged_to_trace_and_file(client, tmp_path):
    r = client.post("/assess", json={"announcement_id": "T001", "profile_id": "o1",
                                     "as_of": AS_OF.isoformat()}).json()  # fmt: skip
    assert r["verdict"] == "ineligible"
    # 틀렸다면서 정답을 안 주면 거부
    bad = client.post("/feedback", json={"trace_id": r["trace_id"], "correct": False})
    assert bad.status_code == 422
    ok = client.post(
        "/feedback",
        json={
            "trace_id": r["trace_id"],
            "correct": False,
            "expected": "needs_review",
            "comment": "지역 불명확",
        },
    )
    assert ok.json() == {"saved": True, "logged_to_trace": True, "expected": "needs_review"}

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(r["trace_id"])
    got = {a.name: a for a in trace.info.assessments}
    assert got["user_says_correct"].value is False
    assert got["expected_verdict"].value == "needs_review"
    assert got["user_says_correct"].source.source_type == "HUMAN"

    saved = json.loads((tmp_path / "feedback.jsonl").read_text("utf-8"))
    assert saved["announcement_id"] == "T001" and saved["predicted"] == "ineligible"
    assert client.post("/feedback", json={"trace_id": "nope", "correct": True}).status_code == 404


def test_reload_swaps_prompts_without_restart(client):
    assert client.get("/info").json()["prompt_versions"]["extract"] == "1"
    r = client.post("/admin/reload").json()
    assert r["before"]["extract"] == "1" and r["after"]["extract"] == "2"


def fb(case, expected, predicted, correct=None):
    a, p = case.split("-")
    correct = expected == predicted if correct is None else correct
    return {"announcement_id": a, "profile_id": p, "as_of": "2026-10-19", "expected": expected,
            "predicted": predicted, "correct": correct, "comment": ""}  # fmt: skip


def test_feedback_review_sorts_new_conflict_known():
    existing = [EvalCase(case_id="A-p1", announcement_id="A", profile_id="p1", as_of=AS_OF,
                         expected="eligible"),
                EvalCase(case_id="A-p2", announcement_id="A", profile_id="p2", as_of=AS_OF,
                         expected="ineligible")]  # fmt: skip
    feedback = [
        fb("A-p1", "eligible", "eligible"),  # 평가셋과 같음
        fb("A-p2", "eligible", "ineligible"),  # 평가셋 라벨과 충돌 → 덮어쓰지 않음
        fb("B-p1", "needs_review", "ineligible"),  # 새 사례
        fb("B-p2", "eligible", "eligible"),
        fb("B-p2", "ineligible", "eligible"),  # 피드백끼리 충돌
    ]
    r = fb_mod.review(feedback, existing)
    assert [c for c, *_ in r["known"]] == ["A-p1"]
    assert sorted(c for c, *_ in r["conflict"]) == ["A-p2", "B-p2"]
    assert [c for c, *_ in r["new"]] == ["B-p1"]
    case = fb_mod.to_case("B-p1", r["new"][0][2])
    assert case.expected == "needs_review" and "정정" in case.note

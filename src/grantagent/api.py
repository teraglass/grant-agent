"""판정 API + 데모 페이지 + 사용자 피드백.

    GET  /                 데모 페이지 (공고·회사 선택 → 판정 → 피드백)
    GET  /announcements    스냅샷 공고 목록
    GET  /profiles         회사 프로필 목록
    POST /assess           판정 (응답에 trace_id 포함)
    POST /feedback         판정이 맞았는지/정답이 무엇인지 → MLflow 트레이스 + data/feedback.jsonl
    GET  /info             현재 모델·프롬프트 버전
    POST /admin/reload     @production 프롬프트를 다시 불러오기 (게이트 승격 후 재시작 없이 반영)

피드백은 해당 판정의 트레이스에 사람 평가(assessment)로 붙어 MLflow UI에서 함께 보이고,
`make feedback`으로 검토한 뒤 평가셋에 추가한다. (피드백도 틀릴 수 있으므로 자동 반영하지 않는다)
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from grantagent.config import DATA_DIR, agent_model, tracking_uri
from grantagent.schemas import VERDICT_KO, Verdict

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"


class AssessRequest(BaseModel):
    announcement_id: str
    profile_id: str
    as_of: date | None = Field(default=None, description="기준일 (기본: 오늘)")


class FeedbackRequest(BaseModel):
    trace_id: str
    correct: bool
    expected: Verdict | None = Field(default=None, description="틀렸다면 올바른 판정")
    comment: str = ""
    user: str = "demo-user"


def _summary(a) -> dict:
    return {
        "verdict": a.verdict,
        "verdict_ko": VERDICT_KO[a.verdict],
        "summary": a.summary,
        "checks": [c.model_dump() for c in a.checks],
        "model": a.model,
        "prompt_versions": a.prompt_versions,
        "trace_id": a.trace_id,
    }


def create_app(
    agent_factory=None,
    anns: dict | None = None,
    profiles: dict | None = None,
    feedback_path: Path = FEEDBACK_PATH,
    tracing_enabled: bool | None = None,
) -> FastAPI:
    """agent_factory: @production 프롬프트로 새 에이전트를 만드는 함수 (테스트에서 교체)."""
    from grantagent.data import load_announcements, load_profiles

    if agent_factory is None:
        agent_factory = _default_agent_factory
    if tracing_enabled is None:
        tracing_enabled = _setup_tracing()
    anns = anns if anns is not None else load_announcements()
    profiles = profiles if profiles is not None else load_profiles()

    app = FastAPI(title="지원사업 적격성 판정 에이전트")
    state = {"agent": agent_factory()}
    # 피드백은 trace_id만 받고, 어떤 공고·회사에 대한 판정이었는지는 서버가 기억한다
    recent: OrderedDict[str, dict] = OrderedDict()
    lock = threading.Lock()

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        agent = state["agent"]
        return {"model": agent.llm.model, "prompt_versions": agent.prompt_versions,
                "tracing": tracing_enabled}  # fmt: skip

    @app.get("/announcements")
    def list_announcements():
        return [
            {"id": a.id, "title": a.title, "org": a.org, "region": a.region,
             "receipt_end": a.receipt_end, "url": a.url}
            for a in anns.values()
        ]  # fmt: skip

    @app.get("/profiles")
    def list_profiles():
        return [p.model_dump(mode="json") for p in profiles.values()]

    @app.post("/assess")
    def assess(req: AssessRequest):
        if req.announcement_id not in anns:
            raise HTTPException(404, f"공고 {req.announcement_id}가 없습니다")
        if req.profile_id not in profiles:
            raise HTTPException(404, f"프로필 {req.profile_id}가 없습니다")
        as_of = req.as_of or date.today()
        a = state["agent"].assess(anns[req.announcement_id], profiles[req.profile_id], as_of)
        out = _summary(a)
        if a.trace_id:
            with lock:
                recent[a.trace_id] = {
                    "announcement_id": req.announcement_id,
                    "profile_id": req.profile_id,
                    "as_of": as_of.isoformat(),
                    "predicted": a.verdict,
                    "prompt_versions": a.prompt_versions,
                    "model": a.model,
                }
                while len(recent) > 1000:
                    recent.popitem(last=False)
        return out

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        with lock:
            ctx = recent.get(req.trace_id)
        if ctx is None:
            raise HTTPException(404, "알 수 없는 trace_id입니다 (서버 재시작 전 판정일 수 있음)")
        expected = ctx["predicted"] if req.correct else req.expected
        if expected is None:
            raise HTTPException(422, "틀렸다면 올바른 판정(expected)을 골라 주세요")

        logged = False
        if tracing_enabled:
            import mlflow
            from mlflow.entities import AssessmentSource, AssessmentSourceType

            source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=req.user)

            def log_to_trace():
                mlflow.log_feedback(
                    trace_id=req.trace_id, name="user_says_correct", value=req.correct,
                    source=source, rationale=req.comment or None,
                )  # fmt: skip
                mlflow.log_expectation(
                    trace_id=req.trace_id, name="expected_verdict", value=expected, source=source
                )

            try:
                try:
                    log_to_trace()
                except Exception:
                    # 트레이스는 백그라운드로 저장되므로, 판정 직후 피드백이 오면 아직 없을 수 있다
                    mlflow.flush_trace_async_logging()
                    log_to_trace()
                logged = True
            except Exception as e:  # MLflow가 잠시 내려가도 피드백 자체는 파일에 남긴다
                log.warning("피드백을 트레이스에 기록하지 못함: %s", e)

        record = {
            "trace_id": req.trace_id,
            **ctx,
            "expected": expected,
            "correct": req.correct,
            "comment": req.comment,
            "user": req.user,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with lock, feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"saved": True, "logged_to_trace": logged, "expected": expected}

    @app.post("/admin/reload")
    def reload():
        """게이트가 @production을 옮긴 뒤 서버 재시작 없이 새 프롬프트로 교체."""
        new_agent = agent_factory()
        old = state["agent"].prompt_versions
        state["agent"] = new_agent
        return {"before": old, "after": new_agent.prompt_versions}

    return app


def _setup_tracing() -> bool:
    from grantagent import prompts, tracing

    if not prompts.server_reachable(tracking_uri()):
        import mlflow

        log.warning("MLflow 서버(%s)에 연결할 수 없어 트레이싱 없이 실행합니다", tracking_uri())
        mlflow.tracing.disable()
        return False
    tracing.setup()
    return True


def _default_agent_factory():
    from grantagent.agent import EligibilityAgent
    from grantagent.llm import ClaudeLLM

    return EligibilityAgent(ClaudeLLM(agent_model()))


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="판정 API + 데모 페이지")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

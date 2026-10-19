"""평가 파이프라인과 승격 게이트를 가짜 LLM + 로컬 MLflow로 검증."""

import mlflow
import pytest
from conftest import AS_OF
from mlflow import MlflowClient
from test_agent import FakeLLM, criteria, make_agent

from grantagent import evaluate, gate, prompts
from grantagent.data import EvalCase
from grantagent.schemas import JudgeOutput, OtherCheck


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")
    monkeypatch.setattr(evaluate, "REPORT_DIR", tmp_path / "reports")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("grant-agent")
    yield uri


def row(expected, verdict, quote_valid=True, latency=1.0, cost=0.01):
    checks = [{"source": "rule", "quote_valid": quote_valid}]
    usage = {"input_tokens": 100, "output_tokens": 10}
    return dict(expected=expected, verdict=verdict, checks=checks, latency_s=latency,
                cost_usd=cost, usage=usage)  # fmt: skip


def test_compute_metrics():
    rows = [
        row("eligible", "eligible"),
        row("eligible", "ineligible"),  # 잘못된 탈락
        row("needs_review", "needs_review"),
        row("ineligible", "ineligible"),
        row("ineligible", "eligible", quote_valid=False),  # 잘못된 통과 + 환각 인용
    ]
    m = evaluate.compute_metrics(rows)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["false_rejection_rate"] == pytest.approx(1 / 3)  # 정답이 지원불가 아닌 3건 중 1건
    assert m["false_accept_rate"] == pytest.approx(1 / 3)
    assert m["recall_eligible"] == pytest.approx(0.5)
    assert m["evidence_valid_rate"] == pytest.approx(4 / 5)
    assert m["baseline_majority_accuracy"] == pytest.approx(2 / 5)
    assert m["cost_usd_total"] == pytest.approx(0.05)


def test_cost_unknown_model_is_none():
    assert evaluate.cost_usd("unknown-model", {"input_tokens": 1}) is None
    assert evaluate.cost_usd("claude-sonnet-5-5", {"input_tokens": 1_000_000}) == 2.0


def test_run_eval_end_to_end(ann, young_seoul, old_busan, tmp_path):
    judge = JudgeOutput(checks=[OtherCheck(requirement="x", status="pass", reason="ICT")])
    llm = FakeLLM(criteria(), judge)
    agent = make_agent(llm, tmp_path)
    cases = [
        EvalCase(case_id="T001-y1", announcement_id="T001", profile_id="y1", as_of=AS_OF,
                 expected="eligible"),
        EvalCase(case_id="T001-o1", announcement_id="T001", profile_id="o1", as_of=AS_OF,
                 expected="needs_review"),
    ]  # fmt: skip
    run_id, m = evaluate.run_eval(
        agent, cases, run_name="t", anns={"T001": ann},
        profiles={"y1": young_seoul, "o1": old_busan},
    )  # fmt: skip
    assert m["accuracy"] == 0.5  # o1은 나이·지역 탈락 → ineligible (정답은 확인 필요)
    assert m["false_rejection_rate"] == 0.5
    assert llm.calls.count("CriteriaExtraction") == 1  # 병렬이어도 공고당 1회 추출

    run = MlflowClient().get_run(run_id)
    assert run.data.tags["eval.model"] == "fake-model"
    assert run.data.tags["eval.prompt.extract"] == "1"
    assert run.data.metrics["accuracy"] == 0.5
    assert (tmp_path / "reports" / f"eval-{run_id[:8]}.json").exists()

    # 사례별 scorer 결과가 트레이스에 남는다
    traces = mlflow.search_traces(run_id=run_id, return_type="list")
    names = {a.name for t in traces for a in t.info.assessments}
    assert {"verdict_correct", "no_false_rejection", "evidence_grounded"} <= names


def log_eval_run(versions, metrics, model="m", md5="d"):
    with mlflow.start_run(run_name=f"eval v{versions}") as r:
        mlflow.set_tags(
            {
                "eval.kind": "grant-eval",
                "eval.model": model,
                "eval.dataset_md5": md5,
                "eval.prompt.extract": versions,
                "eval.prompt.judge": versions,
            }
        )
        mlflow.log_metrics(metrics)  # fmt: skip
    return r.info.run_id


GOOD = {"accuracy": 0.8, "false_rejection_rate": 0.1, "evidence_valid_rate": 1.0,
        "cost_usd_per_case": 0.01, "n_errors": 0}  # fmt: skip


def register_two_versions(tmp_path, monkeypatch):
    prompts.register("v1")  # v1 → @production
    for key in prompts.PROMPTS:
        (tmp_path / f"{key}.txt").write_text(prompts.local_template(key) + "\n# v2", "utf-8")
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    prompts.register("v2")  # v2 등록, @production은 v1 유지


def test_gate_promotes_when_not_worse(tmp_path, monkeypatch):
    register_two_versions(tmp_path, monkeypatch)
    log_eval_run("1", GOOD)
    cand = log_eval_run("2", {**GOOD, "accuracy": 0.85})
    assert gate.promote(cand) is True
    assert gate.production_versions(MlflowClient()) == {"extract": "2", "judge": "2"}
    assert MlflowClient().get_run(cand).data.tags["gate"] == "passed"


def test_gate_blocks_more_false_rejections(tmp_path, monkeypatch):
    register_two_versions(tmp_path, monkeypatch)
    log_eval_run("1", GOOD)
    # 정확도는 올랐지만 지원 가능한 회사를 탈락시키는 비율이 늘었다 → 차단
    cand = log_eval_run("2", {**GOOD, "accuracy": 0.9, "false_rejection_rate": 0.2})
    assert gate.promote(cand) is False
    assert gate.production_versions(MlflowClient()) == {"extract": "1", "judge": "1"}


def test_gate_requires_baseline_with_same_model(tmp_path, monkeypatch):
    register_two_versions(tmp_path, monkeypatch)
    log_eval_run("1", GOOD, model="other-model")
    cand = log_eval_run("2", GOOD)
    with pytest.raises(SystemExit, match="기준 run이 없습니다"):
        gate.promote(cand)


def test_gate_force_rolls_back(tmp_path, monkeypatch):
    register_two_versions(tmp_path, monkeypatch)
    log_eval_run("1", GOOD)
    gate.promote(log_eval_run("2", GOOD))
    old = log_eval_run("1", GOOD)
    assert gate.promote(old, force=True) is True
    assert gate.production_versions(MlflowClient()) == {"extract": "1", "judge": "1"}


def test_check_floor_on_evidence():
    results = gate.check({**GOOD, "evidence_valid_rate": 0.9}, {**GOOD, "evidence_valid_rate": 0.8})
    ok = {g.metric: passed for g, passed, _ in results}
    assert ok["evidence_valid_rate"] is False  # 기준보다 좋아도 최소 0.95 미달


def test_gate_ignores_baseline_runs_with_errors(tmp_path, monkeypatch):
    register_two_versions(tmp_path, monkeypatch)
    log_eval_run("1", GOOD)
    # 더 최근 v1 run이지만 API 오류로 전부 실패 (정확도 0) → 기준으로 쓰면 아무 후보나 통과
    log_eval_run("1", {**GOOD, "accuracy": 0.0, "n_errors": 40})
    cand = log_eval_run("2", {**GOOD, "accuracy": 0.5})
    assert gate.promote(cand) is False  # 정상 기준(0.8)과 비교해 차단


def test_run_with_api_errors_is_marked_failed(ann, young_seoul, tmp_path):
    class BrokenLLM:
        model = "fake-model"

        def structured(self, *a, **k):
            raise RuntimeError("Your credit balance is too low")

    cases = [EvalCase(case_id="T001-y1", announcement_id="T001", profile_id="y1", as_of=AS_OF,
                      expected="eligible")]  # fmt: skip
    run_id, m = evaluate.run_eval(
        make_agent(BrokenLLM(), tmp_path), cases, anns={"T001": ann}, profiles={"y1": young_seoul}
    )
    assert m["n_errors"] == 1
    run = MlflowClient().get_run(run_id)
    assert run.info.status == "FAILED" and "credit" in run.data.tags["eval.error"]

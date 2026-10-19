"""LLM 없이(가짜 응답으로) 에이전트 흐름, 트레이싱, 프롬프트 레지스트리를 검증."""

import mlflow
import pytest
from conftest import AS_OF

from grantagent import prompts
from grantagent.agent import EligibilityAgent
from grantagent.schemas import CriteriaExtraction, JudgeOutput, OtherCheck, OtherRequirement


class FakeLLM:
    model = "fake-model"

    def __init__(self, criteria: CriteriaExtraction, judge: JudgeOutput | None = None):
        self.criteria, self.judge = criteria, judge
        self.calls = []

    def structured(self, prompt, schema, max_tokens=4096):
        self.calls.append(schema.__name__)
        out = self.criteria if schema is CriteriaExtraction else self.judge
        return out, {"input_tokens": 100, "output_tokens": 20}


def criteria(**kw):
    base = dict(
        age_min=None, age_max=39, age_quote="만 39세 이하",
        allows_prospective=True, prospective_only=False, business_years_max=3,
        business_quote="공고일 기준 창업 3년 이내 기업 또는 예비창업자",
        regions=["서울"], region_quote="서울",
        other_requirements=[OtherRequirement(
            requirement="제조·ICT 분야 창업기업", quote="제조·ICT 분야 창업기업"
        )],
    )  # fmt: skip
    return CriteriaExtraction(**{**base, **kw})


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test")
    yield uri


def make_agent(llm, tmp_path):
    p_ex = prompts.Prompt("extract", "grant-extract", "1", prompts.local_template("extract"))
    p_ju = prompts.Prompt("judge", "grant-judge", "1", prompts.local_template("judge"))
    import grantagent.agent as agent_mod

    agent_mod.CACHE_DIR = tmp_path / "cache"
    return EligibilityAgent(llm, p_ex, p_ju)


def test_eligible_case_with_trace(ann, young_seoul, tmp_path):
    judge = JudgeOutput(checks=[OtherCheck(requirement="제조·ICT", status="pass", reason="ICT")])
    llm = FakeLLM(criteria(), judge)
    a = make_agent(llm, tmp_path).assess(ann, young_seoul, AS_OF)

    assert a.verdict == "eligible"
    assert [c.source for c in a.checks] == ["rule", "rule", "rule", "llm"]
    assert all(c.quote_valid for c in a.checks)
    assert a.usage == {"input_tokens": 200, "output_tokens": 40}

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(a.trace_id)
    names = [s.name for s in trace.data.spans]
    assert names[0] == "assess_eligibility"
    assert {"extract_criteria", "rule_checks", "judge_other", "validate_evidence"} <= set(names)
    assert trace.info.tags["verdict"] == "eligible"


def test_rule_failure_makes_ineligible(ann, old_busan, tmp_path):
    judge = JudgeOutput(checks=[OtherCheck(requirement="x", status="fail", reason="유통업")])
    a = make_agent(FakeLLM(criteria(), judge), tmp_path).assess(ann, old_busan, AS_OF)
    assert a.verdict == "ineligible"
    assert {c.criterion for c in a.checks if c.status == "fail"} >= {"대표자 나이", "지역"}


def test_hallucinated_quote_is_not_trusted(ann, young_seoul, tmp_path):
    fake = criteria(
        other_requirements=[
            OtherRequirement(requirement="수출 실적 필수", quote="수출 실적 10만불 이상")
        ]
    )
    judge = JudgeOutput(checks=[OtherCheck(requirement="수출", status="fail", reason="없음")])
    a = make_agent(FakeLLM(fake, judge), tmp_path).assess(ann, young_seoul, AS_OF)
    bad = next(c for c in a.checks if c.source == "llm")
    assert bad.quote_valid is False
    assert a.verdict == "needs_review"  # 지원 불가로 단정하지 않음
    assert "근거 불확실" in a.summary


def test_criteria_cache_skips_second_extraction(ann, young_seoul, old_busan, tmp_path):
    judge = JudgeOutput(checks=[OtherCheck(requirement="x", status="pass", reason="")])
    llm = FakeLLM(criteria(), judge)
    agent = make_agent(llm, tmp_path)
    agent.assess(ann, young_seoul, AS_OF)
    agent.assess(ann, old_busan, AS_OF)
    assert llm.calls.count("CriteriaExtraction") == 1  # 같은 공고는 한 번만 추출


def test_missing_judge_result_becomes_unknown(ann, young_seoul, tmp_path):
    a = make_agent(FakeLLM(criteria(), JudgeOutput(checks=[])), tmp_path).assess(
        ann, young_seoul, AS_OF
    )
    assert a.verdict == "needs_review"


def test_prompt_registry_roundtrip(local_mlflow):
    first = prompts.register("v1")
    assert all("등록" in s and "@production" in s for s in first.values())
    again = prompts.register("v1 again")
    assert all("변경 없음" in s for s in again.values())

    p = prompts.load("extract")
    assert p.version == "1" and "{{announcement}}" in p.template
    assert "공고 내용 X" in p.format(announcement="공고 내용 X")


def test_prompt_falls_back_to_local_file(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # 연결 불가
    p = prompts.load("judge")
    assert p.version == "local" and "{{requirements}}" in p.template

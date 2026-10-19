"""적격성 판정 에이전트.

    공고 ──▶ [1] extract_criteria (LLM)  ── 공고당 1회, 캐시
                  │  나이·업력·지역 조건 + 기타 요건 + 원문 인용
                  ▼
    회사 ──▶ [2] rule_checks (코드)       ── 나이·업력·지역은 계산으로 판정
             [3] judge_other (LLM)       ── 업종·기업형태 등 나머지만 LLM이 판정
             [4] validate_evidence (코드) ── 인용이 공고 원문에 실제로 있는지 검사
             [5] aggregate (코드)         ── 결정 규칙으로 최종 판정

각 단계는 MLflow 트레이스의 span으로 기록되어 UI에서 단계별 입력·출력·토큰을 볼 수 있다.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from datetime import date

import mlflow

from grantagent import prompts
from grantagent.config import CACHE_DIR
from grantagent.llm import StructuredLLM
from grantagent.rules import aggregate, merge_structured, quote_in_text, rule_checks
from grantagent.schemas import (
    Announcement,
    Assessment,
    CompanyProfile,
    CriteriaExtraction,
    CriterionCheck,
    JudgeOutput,
)


def _tag_trace(tags: dict) -> None:
    # MLflow 평가는 시작 전에 첫 사례를 트레이스 없이 한 번 실행해 본다. 그때 생기는 경고를 피한다.
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(tags=tags)


class EligibilityAgent:
    def __init__(
        self,
        llm: StructuredLLM,
        extract_prompt: prompts.Prompt | None = None,
        judge_prompt: prompts.Prompt | None = None,
        use_cache: bool = True,
    ):
        self.llm = llm
        self.extract_prompt = extract_prompt or prompts.load("extract")
        self.judge_prompt = judge_prompt or prompts.load("judge")
        self.use_cache = use_cache
        # 평가는 여러 스레드에서 동시에 assess()를 부르므로 토큰 집계는 스레드별로 한다
        self._local = threading.local()
        # 같은 공고를 여러 회사로 동시에 판정할 때 요건 추출은 한 번만 (메모리 캐시 + 키별 잠금)
        self._memo: dict[str, CriteriaExtraction] = {}
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    @property
    def prompt_versions(self) -> dict[str, str]:
        return {
            "extract": self.extract_prompt.version,
            "judge": self.judge_prompt.version,
        }

    @property
    def _usage(self) -> dict[str, int]:
        if not hasattr(self._local, "usage"):
            self._local.usage = {}
        return self._local.usage

    def _add_usage(self, usage: dict) -> None:
        for k, v in usage.items():
            self._usage[k] = self._usage.get(k, 0) + v

    # ------------------------------------------------------------ [1] 요건 추출 (공고당 1회)

    def _cache_path(self, ann: Announcement):
        key = hashlib.sha256(
            f"{self.llm.model}|{self.extract_prompt.ref}|{ann.as_text()}".encode()
        ).hexdigest()[:16]
        return CACHE_DIR / "criteria" / f"{ann.id}-{key}.json"

    @mlflow.trace(name="extract_criteria", span_type="CHAIN")
    def extract_criteria(self, ann: Announcement) -> CriteriaExtraction:
        path = self._cache_path(ann)
        with self._locks_guard:
            lock = self._locks[path.name]
        with lock:
            if path.name in self._memo:
                _tag_trace({"criteria_cache": "memory"})
                return self._memo[path.name]
            if self.use_cache and path.exists():
                _tag_trace({"criteria_cache": "disk"})
                criteria = CriteriaExtraction.model_validate_json(path.read_text(encoding="utf-8"))
            else:
                prompt = self.extract_prompt.format(announcement=ann.as_text())
                criteria, usage = self.llm.structured(prompt, CriteriaExtraction)
                self._add_usage(usage)
                if self.use_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(criteria.model_dump_json(indent=2), encoding="utf-8")
            self._memo[path.name] = criteria
            return criteria

    # ------------------------------------------------------------ [3] 기타 요건 판정

    @mlflow.trace(name="judge_other", span_type="CHAIN")
    def judge_other(
        self, ann: Announcement, criteria: CriteriaExtraction, profile: CompanyProfile, as_of: date
    ) -> list[CriterionCheck]:
        reqs = criteria.other_requirements
        if not reqs:
            return []
        prompt = self.judge_prompt.format(
            title=ann.title,
            requirements="\n".join(f"{i + 1}. {r.requirement}" for i, r in enumerate(reqs)),
            profile=profile.as_text(as_of),
            as_of=as_of.isoformat(),
        )
        out, usage = self.llm.structured(prompt, JudgeOutput)
        self._add_usage(usage)

        checks = []
        for i, req in enumerate(reqs):
            # 순서로 짝을 맞추고, 모델이 빠뜨린 요건은 unknown으로 남긴다
            judged = out.checks[i] if i < len(out.checks) else None
            checks.append(
                CriterionCheck(
                    criterion=req.requirement,
                    status=judged.status if judged else "unknown",
                    reason=judged.reason if judged else "판정 결과 누락",
                    quote=req.quote,
                    source="llm",
                )
            )
        return checks

    # ------------------------------------------------------------ [4] 근거 검증

    @mlflow.trace(name="validate_evidence", span_type="PARSER")
    def validate_evidence(self, ann: Announcement, checks: list[CriterionCheck]) -> dict:
        text = ann.as_text()
        for c in checks:
            c.quote_valid = quote_in_text(c.quote, text)
        quoted = [c for c in checks if c.quote_valid is not None]
        invalid = [c.criterion for c in quoted if not c.quote_valid]
        return {"quoted": len(quoted), "invalid": invalid}

    # ------------------------------------------------------------ 전체

    @mlflow.trace(name="assess_eligibility", span_type="AGENT")
    def assess(self, ann: Announcement, profile: CompanyProfile, as_of: date) -> Assessment:
        self._local.usage = {}
        mlflow.update_current_trace(
            tags={
                "announcement_id": ann.id,
                "profile_id": profile.id,
                "model": self.llm.model,
                "prompt.extract": self.extract_prompt.ref,
                "prompt.judge": self.judge_prompt.ref,
            }
        )
        extracted = self.extract_criteria(ann)
        with mlflow.start_span(name="merge_structured", span_type="PARSER") as span:
            span.set_inputs(
                {"target_age": ann.target_age, "business_period": ann.business_period,
                 "region": ann.region, "llm": extracted.model_dump()}
            )  # fmt: skip
            criteria = merge_structured(ann, extracted)
            span.set_outputs(criteria.model_dump())
        with mlflow.start_span(name="rule_checks", span_type="TOOL") as span:
            span.set_inputs({"criteria": criteria.model_dump(), "profile_id": profile.id})
            checks = rule_checks(criteria, profile, as_of)
            span.set_outputs([c.model_dump() for c in checks])
        checks += self.judge_other(ann, criteria, profile, as_of)
        evidence = self.validate_evidence(ann, checks)
        verdict = aggregate(checks)

        failed = [c.criterion for c in checks if c.status == "fail"]
        unknown = [c.criterion for c in checks if c.status == "unknown"]
        if verdict == "eligible":
            summary = f"확인된 자격 요건 {len(checks)}개를 모두 충족합니다."
        elif verdict == "ineligible":
            summary = f"충족하지 못한 요건: {', '.join(failed)}"
        else:
            parts = [f"확인 필요: {', '.join(unknown)}"] if unknown else []
            if evidence["invalid"]:
                parts.append(f"근거 불확실: {', '.join(evidence['invalid'])}")
            summary = " / ".join(parts) or "추가 확인이 필요합니다."

        mlflow.update_current_trace(
            tags={"verdict": verdict, "invalid_quotes": str(len(evidence["invalid"]))}
        )
        return Assessment(
            announcement_id=ann.id,
            profile_id=profile.id,
            as_of=as_of,
            verdict=verdict,
            checks=checks,
            summary=summary,
            model=self.llm.model,
            prompt_versions=self.prompt_versions,
            usage=dict(self._usage),
            trace_id=mlflow.get_active_trace_id(),
        )


def to_json(a: Assessment) -> str:
    return json.dumps(a.model_dump(mode="json"), ensure_ascii=False, indent=2)

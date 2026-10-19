"""데이터 구조.

LLM 출력 스키마(`CriteriaExtraction`, `JudgeOutput`)는 Claude 구조화 출력(JSON schema)으로
그대로 쓰이므로 필드 설명이 곧 모델에 대한 지시가 된다.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["pass", "fail", "unknown"]
Verdict = Literal["eligible", "ineligible", "needs_review"]

VERDICT_KO = {"eligible": "지원 가능", "ineligible": "지원 불가", "needs_review": "확인 필요"}


# ---------------------------------------------------------------- 입력


class Announcement(BaseModel):
    id: str
    title: str
    org: str | None = None
    region: str | None = None
    target: str | None = None  # 신청 대상 (설명)
    target_types: str | None = None  # 신청 대상 유형 (예: 대학생,일반인,일반기업)
    excluded: str | None = None  # 신청 제외 대상
    target_age: str | None = None  # 대상 연령
    business_period: str | None = None  # 창업 기간 (업력)
    category: str | None = None  # 지원 분야
    receipt_start: str | None = None
    receipt_end: str | None = None
    content: str | None = None  # 공고 내용
    url: str | None = None
    fetched_at: str | None = None

    def as_text(self) -> str:
        """LLM과 근거 검증에 쓰는 공고 본문. 근거 인용은 이 텍스트 안에 있어야 한다."""
        rows = [
            ("공고명", self.title),
            ("주관기관", self.org),
            ("지원지역", self.region),
            ("신청대상유형", self.target_types),
            ("신청대상", self.target),
            ("신청제외대상", self.excluded),
            ("대상연령", self.target_age),
            ("창업기간", self.business_period),
            ("지원분야", self.category),
            ("접수기간", " ~ ".join(x for x in (self.receipt_start, self.receipt_end) if x)),
            ("공고내용", self.content),
        ]
        return "\n".join(f"[{k}] {v}" for k, v in rows if v)


class CompanyProfile(BaseModel):
    id: str
    name: str
    stage: Literal["예비창업자", "창업기업"]
    founded_date: date | None = None  # 창업일 (예비창업자는 없음)
    ceo_birth_date: date
    region: str  # 시·도 (예: 서울, 경기)
    industry: str
    employees: int = 0
    annual_revenue_krw: int = 0
    traits: list[str] = Field(default_factory=list)  # 예: 여성기업, 장애인기업
    description: str = ""

    def as_text(self, as_of: date) -> str:
        from grantagent.rules import business_years, korean_age

        lines = [
            f"- 회사명: {self.name}",
            f"- 단계: {self.stage}",
            f"- 대표자 만 나이: {korean_age(self.ceo_birth_date, as_of)}세",
            f"- 소재지: {self.region}",
            f"- 업종: {self.industry}",
            f"- 직원 수: {self.employees}명",
            f"- 연매출: {self.annual_revenue_krw:,}원",
        ]
        if self.founded_date:
            lines.insert(2, f"- 업력: {business_years(self.founded_date, as_of):.1f}년")
        if self.traits:
            lines.append(f"- 특성: {', '.join(self.traits)}")
        if self.description:
            lines.append(f"- 설명: {self.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------- LLM 1단계: 자격 요건 추출


class OtherRequirement(BaseModel):
    requirement: str = Field(description="수치로 판단할 수 없는 자격 요건을 한 문장으로 요약")
    quote: str = Field(description="해당 요건이 적힌 공고 원문 구절을 글자 그대로 복사")


def _all_required(schema: dict) -> None:
    """기본값이 있는 필드도 모델이 항상 채우도록 JSON schema에서는 전부 필수로 둔다.
    (기본값은 예전 스키마로 저장된 캐시를 읽기 위한 것)"""
    schema["required"] = list(schema.get("properties", {}))


class CriteriaExtraction(BaseModel):
    model_config = ConfigDict(json_schema_extra=_all_required)

    age_min: int | None = Field(description="대표자 최소 만 나이. 제한이 없으면 null")
    age_max: int | None = Field(
        description="대표자 최대 만 나이 (예: '만 39세 이하' → 39). 없으면 null"
    )
    age_quote: str | None = Field(description="나이 조건 원문 구절 (그대로 복사). 없으면 null")
    allows_prospective: bool | None = Field(
        description="예비창업자 신청 가능 여부. 공고에 언급이 없으면 null"
    )
    prospective_only: bool = Field(
        description="예비창업자만 신청할 수 있는 공고면 true (이미 창업한 기업은 불가)"
    )
    business_years_max: float | None = Field(
        description="창업 후 최대 업력(년). 예: '창업 7년 이내' → 7. 제한이 없으면 null"
    )
    business_quote: str | None = Field(
        description="업력·예비창업자 조건 원문 구절 (그대로 복사). 없으면 null"
    )
    business_exception: str | None = Field(
        default=None,
        description=(
            "업력 제한을 넘어도 신청할 수 있는 대안 자격 원문 구절 (그대로 복사). "
            "예: '신산업 분야는 10년 미만', '「벤처기업 육성에 관한 특별법」에 따른 벤처기업'. "
            "없으면 null"
        ),
    )
    open_to_all: bool = Field(
        default=False,
        description=(
            "신청대상 설명이 '누구나', '관심 있는 분 누구나'처럼 자격 제한이 없음을 명시하면 true"
        ),
    )
    regions: list[str] = Field(
        description=(
            "신청 가능한 시·도 목록 (예: ['서울', '경기']). 전국이거나 제한이 없으면 빈 목록"
        )
    )
    region_quote: str | None = Field(description="지역 조건 원문 구절 (그대로 복사). 없으면 null")
    other_requirements: list[OtherRequirement] = Field(
        description="나이·업력·지역 외의 자격 요건 (업종, 기업 형태, 매출, 인증 등). 없으면 빈 목록"
    )


# ---------------------------------------------------------------- LLM 2단계: 기타 요건 판정


class OtherCheck(BaseModel):
    requirement: str = Field(description="판정한 요건 (입력 목록의 문장 그대로)")
    status: Status = Field(
        description="회사가 요건을 충족하면 pass, 아니면 fail, 정보가 부족하면 unknown"
    )
    reason: str = Field(description="판정 이유를 한 문장으로")


class JudgeOutput(BaseModel):
    checks: list[OtherCheck]


# ---------------------------------------------------------------- 결과


class CriterionCheck(BaseModel):
    criterion: str  # 예: "대표자 나이", "업력", "지역", 또는 기타 요건 문장
    status: Status
    reason: str
    quote: str | None = None
    quote_valid: bool | None = None  # 인용이 공고 원문에 실제로 있는지 (없으면 환각)
    source: Literal["rule", "llm"]


class Assessment(BaseModel):
    announcement_id: str
    profile_id: str
    as_of: date
    verdict: Verdict
    checks: list[CriterionCheck]
    summary: str
    model: str
    prompt_versions: dict[str, str]
    usage: dict[str, int] = Field(default_factory=dict)
    trace_id: str | None = None

    @property
    def verdict_ko(self) -> str:
        return VERDICT_KO[self.verdict]

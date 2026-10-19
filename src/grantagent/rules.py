"""숫자로 판정 가능한 조건은 LLM이 아니라 코드로 판정한다.

LLM은 공고에서 조건(만 39세 이하, 창업 7년 이내, 서울·경기)을 '추출'만 하고,
회사 프로필과 비교하는 계산은 여기서 한다. 나이·날짜 계산을 LLM에 맡기면 틀릴 수 있고,
코드로 하면 같은 입력에 항상 같은 답이 나와 테스트할 수 있다.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from grantagent.schemas import Announcement, CompanyProfile, CriteriaExtraction, CriterionCheck

REGION_ALIASES = {
    "서울특별시": "서울", "서울시": "서울",
    "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원",
    "충청북도": "충북", "충청남도": "충남",
    "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
}  # fmt: skip
NATIONWIDE = {"전국", "전국 대상", "제한없음", "제한 없음", "무관"}


def korean_age(birth: date, as_of: date) -> int:
    """만 나이."""
    return as_of.year - birth.year - ((as_of.month, as_of.day) < (birth.month, birth.day))


def business_years(founded: date, as_of: date) -> float:
    return (as_of - founded).days / 365.25


def normalize_region(name: str) -> str:
    name = name.strip()
    return REGION_ALIASES.get(
        name, name.removesuffix("도").removesuffix("시") if len(name) > 2 else name
    )


def normalize_text(s: str) -> str:
    """근거 비교용: 유니코드 정규화, 공백·구두점 차이 무시."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\s ]+", "", s)
    return re.sub(r"[\"'“”‘’·,.]", "", s)


def quote_in_text(quote: str | None, text: str) -> bool | None:
    """인용이 공고 원문에 있는지. 인용이 없으면 None (검사 대상 아님).

    여러 줄에 걸친 조건을 인용하면 모델이 줄과 줄을 이어 붙이며 사이의 괄호 등을 빠뜨린다.
    (예: '…10년 미만),\n「벤처기업…' → '…10년 미만, 「벤처기업…')
    그래서 줄·쉼표 단위 조각으로 나눠 **모든 조각이 원문에 있으면** 유효로 본다.
    짧은 조각(4자 미만)은 어디에나 있을 수 있어 근거로 치지 않는다.
    """
    if not quote or not quote.strip():
        return None
    body = normalize_text(text)
    if normalize_text(quote) in body:
        return True
    # 줄바꿈·쉼표·세미콜론에서 나눈다 (모델이 줄을 공백이나 쉼표로 이어 붙이기도 한다)
    pieces = [normalize_text(x).rstrip(")") for x in re.split(r"[\n,;]", quote)]
    pieces = [x for x in pieces if len(x) >= 4]
    return len(pieces) > 1 and all(x in body for x in pieces)


# ---------------------------------------------------------------- 공고의 구조화 분류 필드
# K-Startup은 대상 연령·창업 기간을 체크박스 분류로 제공한다.
#   biz_trgt_age: "만 20세 미만,만 20세 이상 ~ 만 39세 이하,만 40세 이상"
#   biz_enyy:     "예비창업자,1년미만,2년미만,...,10년미만"
# 형식이 정해져 있으므로 LLM 없이 코드로 해석한다. 모르는 표현이 섞이면 None (LLM 추출에 맡김).

_AGE_PATTERNS = [
    (re.compile(r"만\s*(\d+)\s*세\s*이상\s*~\s*만\s*(\d+)\s*세\s*이하"), lambda a, b: (a, b)),
    (re.compile(r"만\s*(\d+)\s*세\s*미만"), lambda a: (None, a - 1)),
    (re.compile(r"만\s*(\d+)\s*세\s*이상"), lambda a: (a, None)),
]


def _parts(text: str | None) -> list[str]:
    return [p.strip() for p in re.split(r"[,\n]", text or "") if p.strip()]


def parse_age_categories(text: str | None) -> tuple[int | None, int | None] | None:
    """선택된 연령 구간들을 하나의 (최소, 최대)로. 제한 없으면 (None, None)."""
    parts = _parts(text)
    if not parts:
        return None
    ranges = []
    for part in parts:
        for pattern, to_range in _AGE_PATTERNS:
            if m := pattern.fullmatch(part):
                ranges.append(to_range(*map(int, m.groups())))
                break
        else:
            return None
    lo = None if any(r[0] is None for r in ranges) else min(r[0] for r in ranges)
    hi = None if any(r[1] is None for r in ranges) else max(r[1] for r in ranges)
    return lo, hi


def parse_business_categories(text: str | None) -> dict | None:
    """예: '예비창업자,1년미만,3년미만' → 예비창업자 가능, 업력 3년 미만."""
    parts = _parts(text)
    if not parts:
        return None
    allows, limits, open_ended = False, [], False
    for part in parts:
        if part == "예비창업자":
            allows = True
        elif m := re.fullmatch(r"(\d+)\s*년\s*미만", part):
            limits.append(int(m.group(1)))
        elif re.fullmatch(r"(\d+)\s*년\s*이상", part):
            open_ended = True
        else:
            return None
    return {
        "allows_prospective": allows,
        "prospective_only": allows and not limits and not open_ended,
        "years_lt": None if open_ended or not limits else max(limits),
    }


# 공고명에 이런 말이 있으면 하루짜리 행사일 가능성이 높아, [지원지역]이 개최지일 수 있다
EVENT_KEYWORDS = (
    "매칭데이", "데모데이", "컨퍼런스", "설명회", "세미나", "포럼", "밋업", "meet-up", "meetup",
    "네트워킹", "박람회", "페스티벌", "summit",
)  # fmt: skip
EVENT_CATEGORY = "행사ㆍ네트워크"


def is_event(ann: Announcement) -> bool:
    title = (ann.title or "").lower()
    return any(k in title for k in EVENT_KEYWORDS)


class Criteria(CriteriaExtraction):
    """LLM 추출 결과 + 구조화 분류 필드를 합친 최종 조건 (LLM 스키마에는 쓰지 않음)."""

    business_years_exclusive: bool = False  # True면 '미만'(<), False면 '이내'(<=)
    region_uncertain: bool = False  # 지원지역이 소재지 요건인지 행사 개최지인지 불명확
    sources: dict[str, str] = {}  # 조건별 출처: structured | llm | ignored:<이유>


def merge_structured(ann: Announcement, c: CriteriaExtraction) -> Criteria:
    """구조화 분류(체크박스)와 LLM 추출(본문)을 합친다.

    - 나이·업력: 둘 중 **더 엄격한 조건**. 분류가 '제한 없음'이어도 본문이 더 좁으면 본문을 따른다.
    - 본문이 '누구나'를 명시(open_to_all): 체크박스 분류는 자격 제한으로 보지 않는다.
    - 지역: 본문에 지역 조건이 있으면 본문 우선. 본문이 침묵하면 체크박스를 따르되,
      행사(지원분야 행사ㆍ네트워크)면 개최지로 보고 무시, 공고명만 행사 같으면 '불명확'.
    """
    out = Criteria(**c.model_dump())
    out.sources = {}
    if c.open_to_all:
        out.sources["structured"] = "ignored:open_to_all"
        return out

    age = parse_age_categories(ann.target_age)
    if age and age != (None, None):
        s_min, s_max = age
        llm_stricter = (c.age_max is not None and (s_max is None or c.age_max < s_max)) or (
            c.age_min is not None and (s_min is None or c.age_min > s_min)
        )
        out.age_min = max(x for x in (s_min, c.age_min, -1) if x is not None)
        out.age_min = None if out.age_min == -1 else out.age_min
        out.age_max = min(x for x in (s_max, c.age_max, 999) if x is not None)
        out.age_max = None if out.age_max == 999 else out.age_max
        if not llm_stricter:
            out.age_quote = ann.target_age
            out.sources["age"] = "structured"

    biz = parse_business_categories(ann.business_period)
    if biz:
        out.allows_prospective = biz["allows_prospective"] and c.allows_prospective is not False
        out.prospective_only = biz["prospective_only"] or c.prospective_only
        lt = biz["years_lt"]
        llm_max = c.business_years_max
        if lt is not None and (llm_max is None or lt <= llm_max):
            out.business_years_max = float(lt)
            out.business_years_exclusive = True
            out.business_quote = ann.business_period
            out.sources["business"] = "structured"
        elif lt is None and llm_max is None:
            out.business_quote = c.business_quote or ann.business_period
            out.sources["business"] = "structured"

    regions = [normalize_region(r) for r in _parts(ann.region)]
    if regions and not set(regions) & NATIONWIDE:
        # '[지원지역] 대전'처럼 체크박스 값을 그대로 인용한 경우는 본문 근거로 보지 않는다
        # 본문 근거 = 인용이 신청대상·제외대상·공고내용 문장 안에 실제로 있을 때만.
        # ('[지원지역] 서울'처럼 체크박스 값을 인용한 경우는 본문 근거가 아니다)
        quote = normalize_text(re.sub(r"\[[^\]]*\]", "", c.region_quote or ""))
        body = normalize_text(" ".join(x for x in (ann.target, ann.excluded, ann.content) if x))
        from_text = bool(quote) and quote in body
        if from_text:
            # 본문에 지역 조건이 있음 (예: '대전/세종/충남/충북 소재', '전국 소재') → 본문 우선
            out.sources["region"] = "llm"
        elif ann.category == EVENT_CATEGORY:
            out.regions, out.region_quote = [], None
            out.sources["region"] = "ignored:event_venue"
        elif is_event(ann):
            out.regions, out.region_quote = regions, ann.region
            out.region_uncertain = True
            out.sources["region"] = "structured:uncertain"
        else:
            out.regions, out.region_quote = regions, ann.region
            out.sources["region"] = "structured"
    return out


# ---------------------------------------------------------------- 개별 조건


def check_age(c: CriteriaExtraction, p: CompanyProfile, as_of: date) -> CriterionCheck | None:
    if c.age_min is None and c.age_max is None:
        return None
    age = korean_age(p.ceo_birth_date, as_of)
    ok = (c.age_min is None or age >= c.age_min) and (c.age_max is None or age <= c.age_max)
    rng = f"{c.age_min or ''}~{c.age_max or ''}세"
    return CriterionCheck(
        criterion="대표자 나이",
        status="pass" if ok else "fail",
        reason=f"대표자 만 {age}세, 조건 {rng}",
        quote=c.age_quote,
        source="rule",
    )


def check_business(c: CriteriaExtraction, p: CompanyProfile, as_of: date) -> CriterionCheck | None:
    if p.stage == "예비창업자":
        if c.allows_prospective is None:
            if c.business_years_max is None:
                return None
            return CriterionCheck(
                criterion="업력",
                status="unknown",
                reason="예비창업자이나 공고에 예비창업자 신청 가능 여부가 명시되지 않음",
                quote=c.business_quote,
                source="rule",
            )
        return CriterionCheck(
            criterion="업력",
            status="pass" if c.allows_prospective else "fail",
            reason="예비창업자 신청 " + ("가능" if c.allows_prospective else "불가"),
            quote=c.business_quote,
            source="rule",
        )

    if c.prospective_only:
        return CriterionCheck(
            criterion="업력",
            status="fail",
            reason="예비창업자 전용 공고이나 이미 창업한 기업",
            quote=c.business_quote,
            source="rule",
        )
    if c.business_years_max is None:
        return None
    if p.founded_date is None:
        return CriterionCheck(
            criterion="업력", status="unknown", reason="창업일 정보 없음", quote=c.business_quote,
            source="rule",
        )  # fmt: skip
    years = business_years(p.founded_date, as_of)
    exclusive = getattr(c, "business_years_exclusive", False)
    ok = years < c.business_years_max if exclusive else years <= c.business_years_max
    unit = "미만" if exclusive else "이내"
    reason = f"업력 {years:.1f}년, 조건 {c.business_years_max:g}년 {unit}"
    if not ok and c.business_exception:
        # 업력을 넘었지만 대안 자격(벤처기업 등)이 있으면 단정하지 않는다
        return CriterionCheck(
            criterion="업력",
            status="unknown",
            reason=f"{reason} 초과. 대안 자격 해당 여부 확인 필요: {c.business_exception}",
            quote=c.business_exception,
            source="rule",
        )
    return CriterionCheck(
        criterion="업력",
        status="pass" if ok else "fail",
        reason=reason,
        quote=c.business_quote,
        source="rule",
    )


def check_region(c: CriteriaExtraction, p: CompanyProfile) -> CriterionCheck | None:
    regions = {normalize_region(r) for r in c.regions if r.strip()}
    if not regions or regions & NATIONWIDE:
        return None
    mine = normalize_region(p.region)
    if getattr(c, "region_uncertain", False) and mine not in regions:
        return CriterionCheck(
            criterion="지역",
            status="unknown",
            reason=f"소재지 {mine}, 지원지역 {', '.join(sorted(regions))}이 소재지 요건인지 "
            "행사 개최지인지 불명확",
            quote=c.region_quote,
            source="rule",
        )
    return CriterionCheck(
        criterion="지역",
        status="pass" if mine in regions else "fail",
        reason=f"소재지 {mine}, 대상 지역 {', '.join(sorted(regions))}",
        quote=c.region_quote,
        source="rule",
    )


def rule_checks(c: CriteriaExtraction, p: CompanyProfile, as_of: date) -> list[CriterionCheck]:
    checks = [check_age(c, p, as_of), check_business(c, p, as_of), check_region(c, p)]
    if c.open_to_all:
        # '누구나 신청 가능'은 조건이 없는 게 아니라 '제한 없음'이 확인된 것 → 통과 근거로 남긴다
        # (이게 없으면 확인한 요건 0개 → aggregate가 '확인 필요'로 판정)
        checks.insert(0, CriterionCheck(
            criterion="신청 대상", status="pass", reason="공고에 누구나 신청 가능하다고 명시",
            source="rule",
        ))  # fmt: skip
    return [x for x in checks if x is not None]


# ---------------------------------------------------------------- 최종 판정 (결정 규칙)


def aggregate(checks: list[CriterionCheck]) -> str:
    """
    - 근거가 원문에 있는 fail이 하나라도 있으면 → 지원 불가
    - 근거가 원문에 없는(환각) fail은 믿지 않고 unknown으로 취급
    - unknown이 하나라도 있으면 → 확인 필요
    - 모두 pass면 → 지원 가능
    - 확인한 요건이 하나도 없으면 → 확인 필요 (요건을 못 찾은 것을 '조건 없음'으로 보지 않음)
    """
    if not checks:
        return "needs_review"
    trusted_fail = any(c.status == "fail" and c.quote_valid is not False for c in checks)
    if trusted_fail:
        return "ineligible"
    if any(
        c.status == "unknown" or (c.status == "fail" and c.quote_valid is False) for c in checks
    ):
        return "needs_review"
    return "eligible"

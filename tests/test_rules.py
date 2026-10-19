from datetime import date

from conftest import AS_OF

from grantagent.rules import (
    aggregate,
    business_years,
    korean_age,
    normalize_region,
    quote_in_text,
    rule_checks,
)
from grantagent.schemas import CompanyProfile, CriteriaExtraction, CriterionCheck


def crit(**kw) -> CriteriaExtraction:
    base = dict(
        age_min=None, age_max=None, age_quote=None, allows_prospective=None,
        prospective_only=False, business_years_max=None, business_quote=None,
        regions=[], region_quote=None, other_requirements=[],
    )  # fmt: skip
    return CriteriaExtraction(**{**base, **kw})


def test_korean_age_counts_birthday():
    assert korean_age(date(1987, 10, 19), date(2026, 10, 18)) == 38
    assert korean_age(date(1987, 10, 18), date(2026, 10, 18)) == 39


def test_business_years():
    assert round(business_years(date(2023, 10, 18), date(2026, 10, 18)), 1) == 3.0


def test_region_aliases():
    assert normalize_region("서울특별시") == "서울"
    assert normalize_region("경기도") == "경기"
    assert normalize_region("전북특별자치도") == "전북"


def test_age_and_region_and_years(young_seoul, old_busan):
    c = crit(age_max=39, business_years_max=3, regions=["서울"])
    assert [x.status for x in rule_checks(c, young_seoul, AS_OF)] == ["pass", "pass", "pass"]
    assert [x.status for x in rule_checks(c, old_busan, AS_OF)] == ["fail", "fail", "fail"]


def test_nationwide_region_is_not_a_constraint(old_busan):
    assert rule_checks(crit(regions=["전국"]), old_busan, AS_OF) == []


def test_prospective_founder_rules():
    p = CompanyProfile(
        id="pp", name="예비", stage="예비창업자", ceo_birth_date=date(1999, 1, 1),
        region="서울", industry="교육",
    )  # fmt: skip
    assert rule_checks(crit(allows_prospective=True), p, AS_OF)[0].status == "pass"
    assert rule_checks(crit(allows_prospective=False), p, AS_OF)[0].status == "fail"
    # 업력 제한은 있는데 예비창업자 언급이 없으면 판단 보류
    assert rule_checks(crit(business_years_max=3), p, AS_OF)[0].status == "unknown"


def test_prospective_only_rejects_existing_company(young_seoul):
    out = rule_checks(crit(prospective_only=True, allows_prospective=True), young_seoul, AS_OF)
    assert out[0].status == "fail"


def test_quote_matching_ignores_whitespace_and_punctuation():
    text = "[신청대상] 공고일 기준 창업 3년 이내 기업, 또는 예비창업자"
    assert quote_in_text("창업 3년 이내 기업 또는 예비창업자", text) is True
    assert quote_in_text("창업 7년 이내", text) is False
    assert quote_in_text(None, text) is None


def check(status, quote_valid=True):
    return CriterionCheck(
        criterion="x", status=status, reason="", quote_valid=quote_valid, source="llm"
    )


def test_aggregate_rules():
    assert aggregate([check("pass"), check("pass")]) == "eligible"
    assert aggregate([check("pass"), check("unknown")]) == "needs_review"
    assert aggregate([check("unknown"), check("fail")]) == "ineligible"
    # 원문에 없는 근거로 내린 fail은 믿지 않는다
    assert aggregate([check("pass"), check("fail", quote_valid=False)]) == "needs_review"
    assert aggregate([]) == "needs_review"


def test_quote_spanning_lines_with_dropped_paren():
    from grantagent.rules import quote_in_text

    text = (
        "업력 7년 이내 스타트업(신산업 창업 분야 중소기업*은 10년 미만),\n"
        "「벤처기업 육성에 관한 특별법」에 따른 벤처기업"
    )
    stitched = (
        "신산업 창업 분야 중소기업*은 10년 미만,\n「벤처기업 육성에 관한 특별법」에 따른 벤처기업"
    )
    assert quote_in_text(stitched, text) is True
    # 실제 평가에서 나온 형태: 줄바꿈 대신 공백으로 이어 붙임
    assert quote_in_text(stitched.replace("\n", " "), text) is True
    # 조각 하나라도 원문에 없으면 환각
    assert quote_in_text("신산업 분야는 15년 미만,\n「벤처기업 육성에 관한 특별법」", text) is False
    assert quote_in_text("수출 실적 10만불 이상", text) is False


def test_age_reason_reads_naturally(young_seoul):
    from grantagent.rules import check_age

    assert check_age(crit(age_min=20, age_max=None), young_seoul, AS_OF).reason.endswith(
        "만 20세 이상"
    )
    assert check_age(crit(age_min=None, age_max=39), young_seoul, AS_OF).reason.endswith(
        "만 39세 이하"
    )
    assert check_age(crit(age_min=20, age_max=39), young_seoul, AS_OF).reason.endswith("만 20~39세")

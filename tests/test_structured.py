"""K-Startup 구조화 분류 필드 해석 + LLM 추출과의 병합. RAW는 실제 API 응답 (2026-10-18)."""

from datetime import date

from conftest import AS_OF

from grantagent.kstartup import normalize
from grantagent.rules import (
    aggregate,
    merge_structured,
    parse_age_categories,
    parse_business_categories,
    rule_checks,
)
from grantagent.schemas import CompanyProfile, CriteriaExtraction

RAW = {
    "aply_excl_trgt_ctnt": None,
    "aply_trgt": "청소년,대학생,일반인,대학,연구기관,일반기업,1인 창조기업",
    "aply_trgt_ctnt": "예비창업자 또는 창업 기업\r\nAI 기반으로 플랫폼 창업을 준비중이거나 "
    "쇼핑몰 이커머스 사업중인 사업가\r\nAI·플랫폼·SaaS·데이터 기반 서비스 아이템 보유자",
    "biz_enyy": "예비창업자,1년미만,2년미만,3년미만,5년미만,7년미만,10년미만",
    "biz_pbanc_nm": "[ 모 집 ] GVC AI Platform Startup Enabling Program",
    "biz_trgt_age": "만 20세 미만,만 20세 이상 ~ 만 39세 이하,만 40세 이상",
    "detl_pg_url": "https://www.k-startup.go.kr/web/contents/bizpbanc-ongoing.do?pbancSn=179540",
    "id": 1,
    "pbanc_ctnt": "AI 기반 플랫폼 서비스를 만들고 싶은 예비창업자와 초기 스타트업을 모집합니다.",
    "pbanc_ntrp_nm": "글로벌벤처센터",
    "pbanc_rcpt_bgng_dt": "20261016",
    "pbanc_rcpt_end_dt": "20261022",
    "pbanc_sn": 179540,
    "supt_biz_clsfc": "사업화",
    "supt_regin": "전국",
}


def empty(**kw) -> CriteriaExtraction:
    base = dict(
        age_min=None, age_max=None, age_quote=None, allows_prospective=None,
        prospective_only=False, business_years_max=None, business_quote=None,
        regions=[], region_quote=None, other_requirements=[],
    )  # fmt: skip
    return CriteriaExtraction(**{**base, **kw})


def company(founded, born="1990-01-01", region="서울", stage="창업기업"):
    return CompanyProfile(
        id="c", name="c", stage=stage, region=region, industry="AI 플랫폼",
        founded_date=date.fromisoformat(founded) if founded else None,
        ceo_birth_date=date.fromisoformat(born),
    )  # fmt: skip


def test_real_record_normalizes_ids_and_line_breaks():
    a = normalize(RAW, "t")
    assert a.id == "179540"  # 'id': 1 이 아니라 공고번호
    assert "\r" not in a.target and "\n" in a.target
    assert a.target_types.startswith("청소년")


def test_parse_age_categories():
    assert parse_age_categories(RAW["biz_trgt_age"]) == (None, None)  # 전 구간 → 제한 없음
    assert parse_age_categories("만 20세 이상 ~ 만 39세 이하") == (20, 39)
    assert parse_age_categories("만 20세 미만,만 20세 이상 ~ 만 39세 이하") == (None, 39)
    assert parse_age_categories("만 40세 이상") == (40, None)
    assert parse_age_categories("청년") is None  # 모르는 표현 → LLM에 맡김
    assert parse_age_categories(None) is None


def test_parse_business_categories():
    b = parse_business_categories(RAW["biz_enyy"])
    assert b == {"allows_prospective": True, "prospective_only": False, "years_lt": 10}
    assert parse_business_categories("예비창업자")["prospective_only"] is True
    assert parse_business_categories("3년미만,5년미만")["allows_prospective"] is False
    assert parse_business_categories("7년미만,10년이상")["years_lt"] is None


def test_merge_uses_structured_business_limit_with_quote():
    ann = normalize(RAW, "t")
    c = merge_structured(ann, empty())
    assert c.business_years_max == 10 and c.business_years_exclusive
    assert c.allows_prospective is True
    assert c.age_min is None and c.age_max is None  # 연령 제한 없음
    assert c.regions == []  # 전국
    assert c.sources == {"business": "structured"}


def test_merge_keeps_stricter_llm_condition():
    ann = normalize(RAW, "t")
    # 분류는 10년 미만이지만 본문에 '창업 3년 이내'가 있다면 LLM 쪽을 따른다
    c = merge_structured(ann, empty(business_years_max=3, business_quote="창업 3년 이내"))
    assert c.business_years_max == 3 and not c.business_years_exclusive
    assert c.business_quote == "창업 3년 이내"
    # 연령 분류는 제한 없음이어도 본문의 '만 39세 이하'는 유지
    c = merge_structured(ann, empty(age_max=39, age_quote="만 39세 이하"))
    assert c.age_max == 39


def test_structured_region_overrides():
    ann = normalize({**RAW, "supt_regin": "부산"}, "t")
    c = merge_structured(ann, empty())
    assert c.regions == ["부산"] and c.region_quote == "부산"


def test_rule_checks_on_real_record():
    ann = normalize(RAW, "t")
    c = merge_structured(ann, empty())
    # 업력 10년 미만: 5년차는 통과, 10.1년차는 탈락, 예비창업자는 통과
    assert rule_checks(c, company("2021-04-20"), AS_OF)[0].status == "pass"
    old = rule_checks(c, company("2016-09-05"), AS_OF)[0]
    assert old.status == "fail" and "10년 미만" in old.reason
    assert rule_checks(c, company(None, stage="예비창업자"), AS_OF)[0].status == "pass"


# ---------------------------------------------------------------- v2: 기준 평가에서 틀린 유형들


def test_open_to_all_ignores_checkbox_limits():
    # 딥테크 컨퍼런스: 신청대상 '관심 있는 누구나', 체크박스는 서울·10년미만
    ann = normalize({**RAW, "supt_regin": "서울", "supt_biz_clsfc": "행사ㆍ네트워크"}, "t")
    c = merge_structured(ann, empty(open_to_all=True))
    checks = rule_checks(c, company("2016-09-05", region="대전"), AS_OF)
    assert [(x.criterion, x.status) for x in checks] == [("신청 대상", "pass")]
    assert aggregate(checks) == "eligible"  # 확인한 요건 0개 → '확인 필요'가 되면 안 됨


def test_event_category_region_is_venue():
    ann = normalize({**RAW, "supt_regin": "서울", "supt_biz_clsfc": "행사ㆍ네트워크"}, "t")
    for quote in ("[지원지역] 서울", "지원지역: 서울", "서울"):  # 체크박스 값을 어떻게 인용하든
        c = merge_structured(ann, empty(regions=["서울"], region_quote=quote))
        assert c.regions == [] and c.sources["region"] == "ignored:event_venue"


def test_event_like_title_makes_region_uncertain():
    # 대전 제조파트너스 매칭데이: 지원분야는 멘토링, 본문에 지역 조건 없음
    ann = normalize(
        {**RAW, "supt_regin": "대전", "supt_biz_clsfc": "멘토링ㆍ컨설팅ㆍ교육",
         "biz_pbanc_nm": "2026 대전 제조파트너스 매칭데이 참가자 모집"}, "t",
    )  # fmt: skip
    c = merge_structured(ann, empty(regions=["대전"], region_quote="대전"))
    region = next(x for x in rule_checks(c, company("2024-06-01", region="경기"), AS_OF)
                  if x.criterion == "지역")  # fmt: skip
    assert region.status == "unknown"
    in_daejeon = rule_checks(c, company("2024-06-01", region="대전"), AS_OF)
    assert next(x for x in in_daejeon if x.criterion == "지역").status == "pass"


def test_free_text_region_beats_checkbox():
    # 충청권 피칭룸: 체크박스는 충북, 본문은 대전/세종/충남/충북
    ann = normalize({**RAW, "supt_regin": "충북",
                     "aply_trgt_ctnt": "대전/세종/충남/충북에 소재하는 기업"}, "t")  # fmt: skip
    c = merge_structured(
        ann,
        empty(regions=["대전", "세종", "충남", "충북"], region_quote="대전/세종/충남/충북에 소재"),
    )
    assert set(c.regions) == {"대전", "세종", "충남", "충북"}
    # 본문이 '전국 소재'를 명시하면 체크박스 지역은 무시
    ann = normalize(
        {**RAW, "supt_regin": "서울", "aply_trgt_ctnt": "전국 소재 업력 7년 이내 스타트업"}, "t"
    )
    c = merge_structured(ann, empty(regions=[], region_quote="전국 소재 업력 7년 이내 스타트업"))
    assert c.regions == []


def test_business_exception_turns_fail_into_unknown():
    ann = normalize(RAW, "t")
    c = merge_structured(
        ann,
        empty(business_years_max=7, business_quote="업력 7년 이내",
              business_exception="「벤처기업 육성에 관한 특별법」에 따른 벤처기업"),
    )  # fmt: skip
    old = rule_checks(c, company("2016-09-05"), AS_OF)[0]
    assert old.status == "unknown" and "대안 자격" in old.reason
    assert rule_checks(c, company("2021-04-20"), AS_OF)[0].status == "pass"

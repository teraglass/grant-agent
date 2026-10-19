from grantagent.kstartup import _items, has_eligibility_info, normalize

RAW = {
    "pbanc_sn": 176543,
    "biz_pbanc_nm": "2026년 예비창업패키지",
    "pbanc_ntrp_nm": "창업진흥원",
    "supt_regin": "전국",
    "aply_trgt_ctnt": "공고일 기준 창업 경험이 없는 예비창업자",
    "biz_trgt_age": "만 39세 이하",
    "biz_enyy": "예비창업자",
    "pbanc_rcpt_bgng_dt": "20261001",
    "pbanc_rcpt_end_dt": "20261031",
    "detl_pg_url": "https://www.k-startup.go.kr/...",
}


def test_items_handles_both_response_shapes():
    assert _items({"data": [RAW]}) == [RAW]
    std = {"response": {"body": {"items": {"item": RAW}}}}
    assert _items(std) == [RAW]
    assert _items({"response": {"body": {"items": ""}}}) == []


def test_normalize_maps_fields_and_text():
    a = normalize(RAW, "2026-10-18T21:00:00")
    assert a.id == "176543" and a.title == "2026년 예비창업패키지"
    assert a.target_age == "만 39세 이하"
    assert "[신청대상] 공고일 기준 창업 경험이 없는 예비창업자" in a.as_text()
    assert has_eligibility_info(a)


def test_normalize_skips_records_without_id_or_title():
    assert normalize({"biz_pbanc_nm": "x"}, "t") is None

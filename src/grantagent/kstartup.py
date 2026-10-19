"""K-Startup 사업공고 수집 (공공데이터포털 15125364, 창업진흥원).

공고는 매일 바뀌므로 평가에는 실시간 API가 아니라 **수집 시점의 스냅샷**
(data/announcements.jsonl)을 쓴다. 그래야 프롬프트·모델 버전끼리 같은 입력으로 비교할 수 있다.

응답 필드명은 공공데이터포털 명세 문서(로그인 후 열람)에 있어 후보 이름을 여러 개 두고,
`--inspect`로 실제 필드를 확인할 수 있게 했다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import httpx

from grantagent.config import ANNOUNCEMENTS_PATH
from grantagent.schemas import Announcement

ENDPOINT = os.getenv(
    "KSTARTUP_ENDPOINT",
    "https://apis.data.go.kr/B552735/kisedKstartupService01/getAnnouncementInformation01",
)

# Announcement 필드 → API 응답 필드 후보 (앞에서부터 먼저 찾은 값 사용)
FIELD_MAP: dict[str, list[str]] = {
    "id": ["pbanc_sn", "pbancSn", "id"],
    "title": ["biz_pbanc_nm", "intg_pbanc_biz_nm", "pbanc_nm", "bizPbancNm"],
    "org": ["pbanc_ntrp_nm", "sprv_inst", "biz_prch_dprt_nm"],
    "region": ["supt_regin", "suptRegin"],
    "target": ["aply_trgt_ctnt", "aply_trgt", "aplyTrgtCtnt"],
    "target_types": ["aply_trgt", "aplyTrgt"],
    "excluded": ["aply_excl_trgt_ctnt", "aplyExclTrgtCtnt"],
    "target_age": ["biz_trgt_age", "bizTrgtAge"],
    "business_period": ["biz_enyy", "bizEnyy"],
    "category": ["supt_biz_clsfc", "suptBizClsfc"],
    "receipt_start": ["pbanc_rcpt_bgng_dt", "pbancRcptBgngDt"],
    "receipt_end": ["pbanc_rcpt_end_dt", "pbancRcptEndDt"],
    "content": ["pbanc_ctnt", "pbancCtnt"],
    "url": ["detl_pg_url", "detlPgUrl"],
}


def _items(payload: dict) -> list[dict]:
    """두 가지 응답 형식 모두 처리: odcloud({data: [...]}), 표준({response:{body:{items:...}}})."""
    if isinstance(payload.get("data"), list):
        return payload["data"]
    body = payload.get("response", {}).get("body", {})
    items = body.get("items") or {}
    item = items.get("item", []) if isinstance(items, dict) else items
    return item if isinstance(item, list) else [item]


def _pick(raw: dict, keys: list[str]) -> str | None:
    for k in keys:
        v = raw.get(k)
        if v not in (None, "", " "):
            return str(v).replace("\r\n", "\n").replace("\r", "\n").strip()
    return None


def normalize(raw: dict, fetched_at: str) -> Announcement | None:
    values = {field: _pick(raw, keys) for field, keys in FIELD_MAP.items()}
    if not values["id"] or not values["title"]:
        return None
    return Announcement(**values, fetched_at=fetched_at)


def fetch_page(api_key: str, page: int, per_page: int, recruiting_only: bool) -> dict:
    params = {"serviceKey": api_key, "page": page, "perPage": per_page, "returnType": "json"}
    if recruiting_only:
        params["cond[rcrt_prgs_yn::EQ]"] = "Y"
    r = httpx.get(ENDPOINT, params=params, timeout=30)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(
            f"API 오류 HTTP {r.status_code}: {r.text[:300]}\n"
            "- 인증키는 'Decoding' 키를 쓰세요\n"
            "- 활용신청 직후에는 키 활성화까지 시간이 걸릴 수 있습니다 (보통 1시간 이내)"
        )
    return r.json()


def has_eligibility_info(a: Announcement) -> bool:
    return any([a.target, a.target_age, a.business_period, a.excluded])


def load_snapshot(path=ANNOUNCEMENTS_PATH) -> dict[str, Announcement]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = Announcement.model_validate_json(line)
            out[a.id] = a
    return out


def save_snapshot(items: dict[str, Announcement], path=ANNOUNCEMENTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in items.values():
            f.write(a.model_dump_json(exclude_none=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Startup 공고 수집 → data/announcements.jsonl")
    parser.add_argument("--limit", type=int, default=60, help="새로 추가할 최대 공고 수")
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument("--all", action="store_true", help="마감된 공고도 포함")
    parser.add_argument("--inspect", action="store_true", help="첫 공고의 원본 필드만 출력")
    args = parser.parse_args()

    api_key = os.getenv("DATA_GO_KR_API_KEY")
    if not api_key:
        sys.exit(
            "DATA_GO_KR_API_KEY가 없습니다. .env에 공공데이터포털 인증키(Decoding)를 넣어 주세요."
        )

    if args.inspect:
        items = _items(fetch_page(api_key, 1, 1, not args.all))
        if not items:
            sys.exit("응답에 공고가 없습니다.")
        raw = items[0]
        print(json.dumps(raw, ensure_ascii=False, indent=2)[:4000])
        mapped = {f: next((k for k in keys if k in raw), None) for f, keys in FIELD_MAP.items()}
        print("\n[매핑 결과]")
        for f, k in mapped.items():
            print(f"  {f:<16} ← {k or '(못 찾음: FIELD_MAP에 실제 필드명 추가 필요)'}")
        return

    snapshot = load_snapshot()
    fetched_at = datetime.now().isoformat(timespec="seconds")
    added, skipped, page = 0, 0, 1
    while added < args.limit:
        items = _items(fetch_page(api_key, page, args.per_page, not args.all))
        if not items:
            break
        for raw in items:
            a = normalize(raw, fetched_at)
            if a is None or a.id in snapshot:
                continue
            if not has_eligibility_info(a):
                skipped += 1
                continue
            snapshot[a.id] = a
            added += 1
            if added >= args.limit:
                break
        page += 1
    save_snapshot(snapshot)
    print(
        f"[fetch] 새 공고 {added}건 추가 (자격 정보 없어 제외 {skipped}건), "
        f"스냅샷 총 {len(snapshot)}건 → {ANNOUNCEMENTS_PATH}"
    )


if __name__ == "__main__":
    main()

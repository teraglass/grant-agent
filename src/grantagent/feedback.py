"""사용자 피드백(data/feedback.jsonl)을 검토해 평가셋에 추가한다.

    데모에서 피드백 ──▶ 트레이스에 사람 평가 + data/feedback.jsonl
                              │
                    make feedback (검토만)         새 사례 / 기존 라벨과 충돌 / 이미 있음
                    make feedback APPLY=1          새 사례만 평가셋에 추가
                              │
                    make eval ──▶ 다음 프롬프트는 늘어난 평가셋으로 게이트 통과해야 함

피드백도 틀릴 수 있으므로:
- 기존 라벨과 다르면 덮어쓰지 않고 '충돌'로 보여 준다 (사람이 eval_set.jsonl을 직접 고친다)
- 같은 사례에 피드백이 여러 번이면 마지막 것을 쓰되, 서로 다르면 충돌로 본다
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime

from grantagent.api import FEEDBACK_PATH
from grantagent.config import EVAL_SET_PATH
from grantagent.data import EvalCase, append_eval_case, load_eval_set
from grantagent.schemas import VERDICT_KO


def load_feedback(path=FEEDBACK_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def review(feedback: list[dict], existing: list[EvalCase]) -> dict[str, list]:
    """피드백을 사례별로 묶어 new / conflict / known으로 나눈다."""
    labels = {c.case_id: c.expected for c in existing}
    by_case: dict[str, list[dict]] = {}
    for fb in feedback:
        case_id = f"{fb['announcement_id']}-{fb['profile_id']}"
        by_case.setdefault(case_id, []).append(fb)

    out: dict[str, list] = {"new": [], "conflict": [], "known": []}
    for case_id, items in by_case.items():
        answers = {fb["expected"] for fb in items}
        last = items[-1]
        if len(answers) > 1:
            out["conflict"].append((case_id, f"피드백끼리 다름: {sorted(answers)}", last))
        elif case_id in labels and labels[case_id] != last["expected"]:
            out["conflict"].append(
                (case_id, f"평가셋 {labels[case_id]} ≠ 피드백 {last['expected']}", last)
            )
        elif case_id in labels:
            out["known"].append((case_id, "평가셋과 같음", last))
        else:
            out["new"].append((case_id, "", last))
    return out


def to_case(case_id: str, fb: dict) -> EvalCase:
    note = "사용자 피드백" + (f": {fb['comment']}" if fb.get("comment") else "")
    if not fb["correct"]:
        note += f" (에이전트 판정 {fb['predicted']} → 정정)"
    return EvalCase(
        case_id=case_id,
        announcement_id=fb["announcement_id"],
        profile_id=fb["profile_id"],
        as_of=date.fromisoformat(fb["as_of"]),
        expected=fb["expected"],
        note=note,
        labeled_at=datetime.now().isoformat(timespec="seconds"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="사용자 피드백 검토 → 평가셋 추가")
    parser.add_argument("--apply", action="store_true", help="새 사례를 평가셋에 추가")
    args = parser.parse_args()

    feedback = load_feedback()
    if not feedback:
        print(f"피드백이 없습니다 ({FEEDBACK_PATH})")
        return
    result = review(feedback, load_eval_set())
    wrong = sum(not fb["correct"] for fb in feedback)
    print(f"[feedback] {len(feedback)}건 (에이전트가 틀렸다는 피드백 {wrong}건)\n")
    titles = {"new": "새 사례", "conflict": "충돌 (직접 확인 필요)", "known": "이미 평가셋에 있음"}
    for kind in ("new", "conflict", "known"):
        if not result[kind]:
            continue
        print(f"■ {titles[kind]} {len(result[kind])}건")
        for case_id, why, fb in result[kind]:
            mark = "✅" if fb["correct"] else "❌"
            print(
                f"  {mark} {case_id:<14} 정답 {VERDICT_KO[fb['expected']]:<6}"
                f" (에이전트 {VERDICT_KO[fb['predicted']]}) {why} {fb.get('comment', '')}"
            )
        print()

    if args.apply and result["new"]:
        for case_id, _, fb in result["new"]:
            append_eval_case(to_case(case_id, fb))
        print(f"[feedback] 새 사례 {len(result['new'])}건을 {EVAL_SET_PATH}에 추가했습니다.")
        print("  평가셋이 바뀌었으니 make eval로 현재 @production의 기준 run을 다시 만드세요.")
    elif result["new"]:
        print("평가셋에 추가하려면: make feedback APPLY=1")


if __name__ == "__main__":
    main()

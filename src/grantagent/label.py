"""평가셋 라벨링 도구: 공고 × 회사 프로필 쌍에 정답 판정을 붙인다.

LLM 평가의 핵심은 '사람이 확인한 정답'이다. 에이전트 초안(--draft)을 보면 빠르지만
초안에 끌려가는 편향(anchoring)이 생기므로, 기본값은 초안 없이 직접 판단하는 방식이다.
초안을 켜더라도 Enter로 그대로 수락하는 기능은 일부러 넣지 않았다.

  e = 지원 가능 (eligible)   i = 지원 불가 (ineligible)   r = 확인 필요 (needs_review)
  s = 건너뛰기               q = 저장하고 종료
"""

from __future__ import annotations

import argparse
import random
import textwrap
from datetime import date, datetime

from grantagent.data import (
    EvalCase,
    append_eval_case,
    load_announcements,
    load_eval_set,
    load_profiles,
)

KEYS = {"e": "eligible", "i": "ineligible", "r": "needs_review"}


def pick_pairs(n_announcements: int, seed: int, profile_ids: list[str]):
    anns = sorted(load_announcements().values(), key=lambda a: a.id)
    random.Random(seed).shuffle(anns)
    return [(a, pid) for a in anns[:n_announcements] for pid in profile_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="평가셋 정답 라벨링")
    parser.add_argument("--n-announcements", type=int, default=8)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--draft", action="store_true", help="에이전트 판정 초안을 함께 표시")
    args = parser.parse_args()

    profiles = load_profiles()
    done = {c.case_id for c in load_eval_set()}
    pairs = [
        (a, pid)
        for a, pid in pick_pairs(args.n_announcements, args.seed, list(profiles))
        if f"{a.id}-{pid}" not in done
    ]
    print(f"라벨링할 쌍 {len(pairs)}개 (이미 완료 {len(done)}개)\n")

    agent = None
    if args.draft:
        from grantagent import tracing
        from grantagent.agent import EligibilityAgent
        from grantagent.config import agent_model
        from grantagent.llm import ClaudeLLM

        tracing.setup("grant-agent-labeling")
        agent = EligibilityAgent(ClaudeLLM(agent_model()))

    last_ann = None
    for n, (ann, pid) in enumerate(pairs, 1):
        profile = profiles[pid]
        print("=" * 80)
        print(f"[{n}/{len(pairs)}] {ann.id} × {pid}")
        if ann.id != last_ann:
            print(textwrap.indent(ann.as_text()[:2500], "  "))
            last_ann = ann.id
        else:
            print(f"  (같은 공고) {ann.title}")
        print("\n  ▶ 회사")
        print(textwrap.indent(profile.as_text(args.as_of), "    "))
        if agent:
            a = agent.assess(ann, profile, args.as_of)
            print(f"\n  ▶ 에이전트 초안: {a.verdict} — {a.summary}")

        while True:
            key = input("\n  정답 [e/i/r] (s=건너뛰기, q=종료): ").strip().lower()
            if key in KEYS or key in ("s", "q"):
                break
        if key == "q":
            break
        if key == "s":
            continue
        note = input("  메모 (근거가 된 조건, 선택): ").strip()
        append_eval_case(
            EvalCase(
                case_id=f"{ann.id}-{pid}",
                announcement_id=ann.id,
                profile_id=pid,
                as_of=args.as_of,
                expected=KEYS[key],
                note=note,
                labeled_at=datetime.now().isoformat(timespec="seconds"),
            )
        )
    total = load_eval_set()
    dist = {v: sum(c.expected == v for c in total) for v in KEYS.values()}
    print(f"\n평가셋 {len(total)}건 저장됨 · 분포 {dist}")


if __name__ == "__main__":
    main()

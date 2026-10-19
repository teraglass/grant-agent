"""grantagent assess / list 명령."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from grantagent.config import agent_model
from grantagent.schemas import VERDICT_KO

ICON = {"pass": "✅", "fail": "❌", "unknown": "❔"}


def print_assessment(a, ann) -> None:
    print(f"\n■ {ann.title}")
    print(f"  판정: {VERDICT_KO[a.verdict]}  ({a.summary})")
    for c in a.checks:
        q = ""
        if c.quote:
            mark = "" if c.quote_valid else "  ⚠ 원문에 없는 인용"
            q = f'\n       근거: "{c.quote[:80]}"{mark}'
        print(f"  {ICON[c.status]} [{c.source}] {c.criterion} — {c.reason}{q}")
    tokens = a.usage.get("input_tokens", 0) + a.usage.get("output_tokens", 0)
    print(f"  모델 {a.model} · 프롬프트 {a.prompt_versions} · 토큰 {tokens:,}")
    if a.trace_id:
        print(f"  trace: {a.trace_id}")


def cmd_list(_args) -> None:
    from grantagent.data import load_announcements, load_profiles

    anns = load_announcements()
    print(f"[공고 {len(anns)}건]")
    for a in list(anns.values())[:50]:
        print(f"  {a.id:<10} {a.title[:60]}")
    print("\n[회사 프로필]")
    for p in load_profiles().values():
        print(f"  {p.id}  {p.name} · {p.stage} · {p.region} · {p.industry}")


def cmd_assess(args) -> None:
    from grantagent import tracing
    from grantagent.agent import EligibilityAgent
    from grantagent.data import load_announcements, load_profiles
    from grantagent.llm import ClaudeLLM

    anns, profiles = load_announcements(), load_profiles()
    if args.announcement not in anns:
        sys.exit(f"공고 {args.announcement}가 스냅샷에 없습니다 (grantagent list로 확인)")
    if args.profile not in profiles:
        sys.exit(f"프로필 {args.profile}가 없습니다")
    tracing.setup()
    agent = EligibilityAgent(ClaudeLLM(args.model), use_cache=not args.no_cache)
    ann = anns[args.announcement]
    a = agent.assess(ann, profiles[args.profile], args.as_of)
    print_assessment(a, ann)


def main() -> None:
    parser = argparse.ArgumentParser(prog="grantagent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="스냅샷 공고와 프로필 목록")

    p = sub.add_parser("assess", help="공고 하나 × 프로필 하나 판정")
    p.add_argument("announcement")
    p.add_argument("profile")
    p.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    p.add_argument("--model", default=agent_model())
    p.add_argument("--no-cache", action="store_true", help="요건 추출 캐시를 쓰지 않음")

    args = parser.parse_args()
    {"list": cmd_list, "assess": cmd_assess}[args.cmd](args)


if __name__ == "__main__":
    main()

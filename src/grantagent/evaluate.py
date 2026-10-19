"""평가셋으로 에이전트를 평가하고 결과를 MLflow run 하나에 기록한다.

    data/eval_set.jsonl (사람이 붙인 정답 40건)
        │
        ▼
    mlflow.genai.evaluate(predict_fn=에이전트, scorers=[...])
        │  사례마다 트레이스 + 평가 기준별 통과/실패
        ▼
    집계 지표 (정확도, macro-F1, 잘못된 탈락 비율, 근거 유효율, 비용, 지연)
        │
        ▼
    MLflow run (tags: 모델, 프롬프트 버전, 평가셋 해시) → 승격 게이트(gate.py)가 비교

평가 조건(모델·프롬프트 버전)은 run 태그로 남기므로, 같은 평가셋에서 조건만 바꿔 여러 번 돌리고
MLflow UI에서 run끼리 비교한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from collections import Counter
from datetime import date

import mlflow

from grantagent import prompts
from grantagent.config import EVAL_SET_PATH, ROOT, agent_model, tracking_uri
from grantagent.data import EvalCase, load_announcements, load_eval_set, load_profiles
from grantagent.scorers import CODE_SCORERS, reasons_faithful_judge

VERDICTS = ["eligible", "ineligible", "needs_review"]
REPORT_DIR = ROOT / "reports"

# USD / 100만 토큰 (입력, 출력). 2026-10 기준 Anthropic 공개 가격
# (Haiku는 프롬프트 10만 토큰 이하 구간)
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, usage: dict) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out) / 1e6


def dataset_hash(path=EVAL_SET_PATH) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- 집계 지표


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    """rows: [{expected, verdict, checks, latency_s, cost_usd, usage}]"""
    n = len(rows)
    y_true = [r["expected"] for r in rows]
    y_pred = [r["verdict"] for r in rows]
    m: dict[str, float] = {"n_cases": n, "n_errors": sum(p == "error" for p in y_pred)}
    m["accuracy"] = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / n

    f1s = []
    for v in VERDICTS:
        tp = sum(t == v and p == v for t, p in zip(y_true, y_pred, strict=True))
        support = y_true.count(v)
        predicted = y_pred.count(v)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        m[f"recall_{v}"] = recall
        m[f"precision_{v}"] = precision
        if support:
            f1s.append(f1)
    m["macro_f1"] = sum(f1s) / len(f1s)

    # 가장 치명적인 실수: 지원할 수 있거나 확인이 필요한 회사에 "지원 불가"
    not_ineligible = [r for r in rows if r["expected"] != "ineligible"]
    m["false_rejection_rate"] = (
        sum(r["verdict"] == "ineligible" for r in not_ineligible) / len(not_ineligible)
        if not_ineligible
        else 0.0
    )
    # 지원할 수 없는 회사에 "지원 가능"
    not_eligible = [r for r in rows if r["expected"] != "eligible"]
    m["false_accept_rate"] = (
        sum(r["verdict"] == "eligible" for r in not_eligible) / len(not_eligible)
        if not_eligible
        else 0.0
    )
    # 항상 가장 많은 정답 하나로만 답했을 때의 정확도 (이보다 높아야 의미가 있다)
    m["baseline_majority_accuracy"] = Counter(y_true).most_common(1)[0][1] / n

    quoted = [c for r in rows for c in r["checks"] if c.get("quote_valid") is not None]
    m["evidence_valid_rate"] = (
        sum(c["quote_valid"] for c in quoted) / len(quoted) if quoted else 1.0
    )
    m["checks_per_case"] = sum(len(r["checks"]) for r in rows) / n
    m["llm_checks_per_case"] = (
        sum(sum(c.get("source") == "llm" for c in r["checks"]) for r in rows) / n
    )

    latencies = sorted(r["latency_s"] for r in rows)
    m["latency_p50_s"] = statistics.median(latencies)
    m["latency_p95_s"] = latencies[min(n - 1, int(round(0.95 * (n - 1))))]
    m["input_tokens"] = sum(r["usage"].get("input_tokens", 0) for r in rows)
    m["output_tokens"] = sum(r["usage"].get("output_tokens", 0) for r in rows)
    costs = [r["cost_usd"] for r in rows]
    if all(c is not None for c in costs):
        m["cost_usd_total"] = sum(costs)
        m["cost_usd_per_case"] = sum(costs) / n
    return m


def confusion_matrix_text(rows: list[dict]) -> str:
    cols = VERDICTS + (["error"] if any(r["verdict"] == "error" for r in rows) else [])
    counts = Counter((r["expected"], r["verdict"]) for r in rows)
    lines = ["정답 \\ 예측 | " + " | ".join(cols), "---|" + "---|" * len(cols)]
    for t in VERDICTS:
        lines.append(f"{t} | " + " | ".join(str(counts[(t, p)]) for p in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------- 실행


def build_agent(model: str, extract_version: int | None, judge_version: int | None, use_cache):
    from grantagent.agent import EligibilityAgent
    from grantagent.llm import ClaudeLLM

    return EligibilityAgent(
        ClaudeLLM(model),
        extract_prompt=prompts.load("extract", version=extract_version),
        judge_prompt=prompts.load("judge", version=judge_version),
        use_cache=use_cache,
    )


def run_eval(
    agent,
    cases: list[EvalCase],
    run_name: str | None = None,
    llm_judge: bool = False,
    extra_tags: dict | None = None,
    anns: dict | None = None,
    profiles: dict | None = None,
) -> tuple[str, dict]:
    """평가 실행. (run_id, 집계 지표) 반환. MLflow 연결·실험 설정은 호출하는 쪽에서 한다."""
    anns = anns if anns is not None else load_announcements()
    profiles = profiles if profiles is not None else load_profiles()
    model = agent.llm.model
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def predict_fn(case_id: str, announcement_id: str, profile_id: str, as_of: str) -> dict:
        start = time.perf_counter()
        profile = profiles[profile_id]
        try:
            a = agent.assess(anns[announcement_id], profile, date.fromisoformat(as_of))
            out = {
                "verdict": a.verdict,
                "summary": a.summary,
                "checks": [c.model_dump() for c in a.checks],
                "usage": a.usage,
            }
        except Exception as e:  # 한 건 실패로 전체 평가를 멈추지 않는다 (오답으로 집계)
            out = {"verdict": "error", "summary": f"{type(e).__name__}: {e}", "checks": [],
                   "usage": {}}  # fmt: skip
        out["latency_s"] = round(time.perf_counter() - start, 3)
        out["cost_usd"] = cost_usd(model, out["usage"])
        out["company"] = profile.as_text(date.fromisoformat(as_of))
        out["llm_checks"] = [
            {k: c[k] for k in ("criterion", "status", "reason")}
            for c in out["checks"]
            if c["source"] == "llm"
        ]
        with lock:
            results[case_id] = out
        return out

    data = [
        {
            "inputs": {
                "case_id": c.case_id,
                "announcement_id": c.announcement_id,
                "profile_id": c.profile_id,
                "as_of": c.as_of.isoformat(),
            },
            "expectations": {"expected_verdict": c.expected, "note": c.note},
        }
        for c in cases
    ]
    scorers = list(CODE_SCORERS) + ([reasons_faithful_judge()] if llm_judge else [])
    tags = {
        "eval.kind": "grant-eval",
        "eval.model": model,
        "eval.prompt.extract": agent.extract_prompt.version,
        "eval.prompt.judge": agent.judge_prompt.version,
        "eval.dataset_md5": dataset_hash(),
        **(extra_tags or {}),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(
            {"model": model, "prompt_extract": agent.extract_prompt.ref,
             "prompt_judge": agent.judge_prompt.ref, "n_cases": len(cases),
             "llm_judge": llm_judge}
        )  # fmt: skip
        mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)

        rows = [{"case_id": c.case_id, "expected": c.expected, "note": c.note,
                 **results[c.case_id]} for c in cases]  # fmt: skip
        metrics = compute_metrics(rows)
        mlflow.log_metrics(metrics)
        cm = confusion_matrix_text(rows)
        mlflow.log_text(cm, "confusion_matrix.md")
        mlflow.log_table(
            {
                "case_id": [r["case_id"] for r in rows],
                "expected": [r["expected"] for r in rows],
                "verdict": [r["verdict"] for r in rows],
                "correct": [r["expected"] == r["verdict"] for r in rows],
                "summary": [r["summary"] for r in rows],
                "label_note": [r["note"] for r in rows],
                "latency_s": [r["latency_s"] for r in rows],
            },
            artifact_file="cases.json",
        )
        mlflow.log_artifact(str(EVAL_SET_PATH))

        REPORT_DIR.mkdir(exist_ok=True)
        report = {"run_id": run.info.run_id, "run_name": run.info.run_name, "tags": tags,
                  "metrics": metrics, "cases": rows}  # fmt: skip
        path = REPORT_DIR / f"eval-{run.info.run_id[:8]}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if metrics["n_errors"]:
            # API 오류(크레딧 소진, 인증 실패 등)가 섞인 run은 성능 비교에 쓰면 안 된다.
            # FAILED로 끝내 게이트가 기준 run으로 고르지 않게 한다.
            first_error = next(r["summary"] for r in rows if r["verdict"] == "error")
            mlflow.set_tag("eval.error", first_error[:500])
            mlflow.end_run(status="FAILED")
        return run.info.run_id, metrics


def print_report(run_id: str, metrics: dict, rows_path=None) -> None:
    def pct(k):
        return f"{metrics[k]:.1%}"

    print(f"\n[eval] run {run_id}")
    print(f"  정확도        {pct('accuracy')}  (항상 다수 정답만 답하면 "
          f"{pct('baseline_majority_accuracy')})")  # fmt: skip
    print(f"  macro-F1      {metrics['macro_f1']:.3f}")
    print(
        f"  재현율        지원가능 {pct('recall_eligible')} · 지원불가 {pct('recall_ineligible')}"
        f" · 확인필요 {pct('recall_needs_review')}"
    )
    print(f"  잘못된 탈락    {pct('false_rejection_rate')}  (지원 가능·확인 필요 → 지원 불가)")
    print(f"  근거 유효율    {pct('evidence_valid_rate')}")
    cost = metrics.get("cost_usd_total")
    cost_s = f"${cost:.3f} (건당 ${metrics['cost_usd_per_case']:.4f})" if cost is not None else "-"
    print(f"  비용          {cost_s} · 토큰 {metrics['input_tokens']:,.0f}/"
          f"{metrics['output_tokens']:,.0f}")  # fmt: skip
    print(
        f"  지연          p50 {metrics['latency_p50_s']:.1f}s · p95 {metrics['latency_p95_s']:.1f}s"
    )
    if metrics["n_errors"]:
        print(f"  ⚠ 실행 오류 {metrics['n_errors']:.0f}건 → run을 FAILED로 표시 (게이트 제외)")
    print(f"  리포트        reports/eval-{run_id[:8]}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="평가셋으로 에이전트 평가 → MLflow run")
    parser.add_argument("--model", default=agent_model())
    parser.add_argument("--extract-version", type=int, help="기본: @production")
    parser.add_argument("--judge-version", type=int, help="기본: @production")
    parser.add_argument("--llm-judge", action="store_true", help="LLM 심사위원 추가 (비용 발생)")
    parser.add_argument("--use-cache", action="store_true", help="요건 추출 디스크 캐시 사용")
    parser.add_argument("--limit", type=int, help="앞에서 N건만 (점검용)")
    args = parser.parse_args()

    from grantagent import tracing

    if not prompts.server_reachable(tracking_uri()):
        raise SystemExit(
            f"MLflow 서버({tracking_uri()})에 연결할 수 없습니다. 평가 결과를 기록하려면 서버가 "
            "필요합니다.\n  - mlops-eurosat 폴더에서 `make up`"
        )
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "4")  # API 동시 호출 수
    tracing.setup()
    cases = load_eval_set()[: args.limit]
    if not cases:
        raise SystemExit("평가셋이 비어 있습니다. make label로 먼저 라벨링하세요.")
    agent = build_agent(args.model, args.extract_version, args.judge_version, args.use_cache)
    if "local" in agent.prompt_versions.values():
        raise SystemExit(
            "프롬프트를 레지스트리에서 불러오지 못했습니다. make prompts를 먼저 실행하세요."
        )
    name = (
        f"eval {args.model.removeprefix('claude-')} "
        f"extract@v{agent.extract_prompt.version} judge@v{agent.judge_prompt.version}"
    )
    run_id, metrics = run_eval(agent, cases, run_name=name, llm_judge=args.llm_judge)
    print_report(run_id, metrics)


if __name__ == "__main__":
    main()

"""승격 게이트: 후보 프롬프트의 평가 run이 현재 @production보다 나빠지지 않았을 때만 승격한다.

    후보 평가 run (예: extract@v2 judge@v2)
        │  같은 모델 · 같은 평가셋으로 평가한 현재 @production run을 찾아 비교
        ▼
    조건 (GATES)을 모두 통과 → 두 프롬프트의 @production alias를 후보 버전으로 이동
                     실패 → 이유를 출력하고 run에 gate=failed 태그

롤백도 같은 명령이다. 예전 버전으로 평가한 run을 `--force`로 지정하면 alias가 되돌아간다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

from grantagent import prompts
from grantagent.config import EXPERIMENT, tracking_uri


@dataclass(frozen=True)
class Gate:
    metric: str
    label: str
    higher_is_better: bool
    tolerance: float = 0.0  # 기준 대비 허용 오차 (절대값)
    ratio: float | None = None  # 기준의 몇 배까지 허용 (비용 등)
    floor: float | None = None  # 기준과 상관없이 넘어야 하는 최소값


GATES = [
    Gate("accuracy", "정확도", higher_is_better=True),
    Gate("false_rejection_rate", "잘못된 탈락 비율", higher_is_better=False),
    Gate("evidence_valid_rate", "근거 유효율", higher_is_better=True, floor=0.95),
    Gate("cost_usd_per_case", "건당 비용", higher_is_better=False, ratio=1.5),
    Gate("n_errors", "실행 오류", higher_is_better=False),
]


def check(candidate: dict, baseline: dict) -> list[tuple[Gate, bool, str]]:
    results = []
    for g in GATES:
        c, b = candidate.get(g.metric), baseline.get(g.metric)
        if c is None or b is None:
            results.append((g, True, "지표 없음 (건너뜀)"))
            continue
        if g.ratio is not None:
            ok = c <= b * g.ratio
            rule = f"기준의 {g.ratio:g}배 이하"
        elif g.higher_is_better:
            ok = c >= b - g.tolerance
            rule = "기준 이상"
        else:
            ok = c <= b + g.tolerance
            rule = "기준 이하"
        if g.floor is not None and c < g.floor:
            ok, rule = False, f"{rule}, 최소 {g.floor:g}"
        results.append((g, ok, f"후보 {c:.4g} vs 기준 {b:.4g} ({rule})"))
    return results


def production_versions(client: MlflowClient) -> dict[str, str]:
    return {
        key: str(client.get_prompt_version_by_alias(name, prompts.PRODUCTION).version)
        for key, name in prompts.PROMPTS.items()
    }


def find_baseline(client: MlflowClient, exp_id: str, cand, prod: dict) -> object | None:
    t = cand.data.tags
    flt = (
        "tags.`eval.kind` = 'grant-eval'"
        f" and tags.`eval.model` = '{t['eval.model']}'"
        f" and tags.`eval.dataset_md5` = '{t['eval.dataset_md5']}'"
        f" and tags.`eval.prompt.extract` = '{prod['extract']}'"
        f" and tags.`eval.prompt.judge` = '{prod['judge']}'"
        " and attributes.status = 'FINISHED'"
        " and metrics.n_errors = 0"  # 오류가 섞인 run은 기준으로 쓰지 않음
    )
    runs = client.search_runs([exp_id], flt, order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None


def promote(run_id: str, dry_run: bool = False, force: bool = False) -> bool:
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    cand = client.get_run(run_id)
    t = cand.data.tags
    if t.get("eval.kind") != "grant-eval":
        raise SystemExit(f"{run_id}는 평가 run이 아닙니다 (make eval로 만든 run을 지정하세요)")
    cand_versions = {"extract": t["eval.prompt.extract"], "judge": t["eval.prompt.judge"]}
    prod = production_versions(client)
    print(f"[gate] 후보 {cand.info.run_name}  ({run_id[:8]})")
    print(f"       현재 @production: extract v{prod['extract']} · judge v{prod['judge']}")

    if cand_versions == prod and not force:
        print("       후보가 이미 @production 버전입니다. 승격할 것이 없습니다.")
        return False

    passed = True
    if force:
        print("       --force: 게이트 비교 없이 alias를 옮깁니다 (롤백용)")
    else:
        exp_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
        base = find_baseline(client, exp_id, cand, prod)
        if base is None:
            raise SystemExit(
                "비교할 기준 run이 없습니다. 같은 모델·평가셋으로 "
                "현재 @production을 먼저 평가하세요:\n"
                f"  make eval MODEL={t['eval.model']}"
            )
        print(f"       기준 run: {base.info.run_name}  ({base.info.run_id[:8]})\n")
        for g, ok, detail in check(cand.data.metrics, base.data.metrics):
            print(f"  {'✅' if ok else '❌'} {g.label:<12} {detail}")
            passed &= ok
        client.set_tag(run_id, "gate.baseline_run", base.info.run_id)

    client.set_tag(run_id, "gate", "passed" if passed else "failed")
    if not passed:
        print("\n[gate] 실패: @production 유지")
        return False
    if dry_run:
        print("\n[gate] 통과 (--dry-run이라 alias는 그대로)")
        return True

    for key, name in prompts.PROMPTS.items():
        mlflow.genai.set_prompt_alias(
            name, alias=prompts.PRODUCTION, version=int(cand_versions[key])
        )
    client.set_tag(run_id, "promoted_from", f"extract v{prod['extract']} judge v{prod['judge']}")
    print(
        f"\n[gate] 통과 → @production = extract v{cand_versions['extract']} · "
        f"judge v{cand_versions['judge']}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="평가 run을 기준과 비교해 프롬프트 @production 승격"
    )
    parser.add_argument("run_id")
    parser.add_argument("--dry-run", action="store_true", help="비교만 하고 alias는 옮기지 않음")
    parser.add_argument("--force", action="store_true", help="게이트 없이 승격 (롤백용)")
    args = parser.parse_args()
    if not prompts.server_reachable(tracking_uri()):
        raise SystemExit(f"MLflow 서버({tracking_uri()})에 연결할 수 없습니다.")
    promote(args.run_id, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()

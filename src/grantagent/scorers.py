"""평가 기준 (MLflow scorer).

사례별로 통과/실패(bool)를 남겨 MLflow UI에서 어떤 사례가 왜 틀렸는지 바로 볼 수 있게 한다.
비율·F1 같은 집계 지표는 evaluate.py에서 따로 계산한다.

| 평가 기준            | 구현     | 이유 |
|---------------------|----------|------|
| verdict_correct      | 코드     | 정답 라벨과 비교만 하면 됨 |
| no_false_rejection   | 코드     | 가장 치명적인 실수: 지원할 수 있는 회사에 "지원 불가"라고 답함 |
| evidence_grounded    | 코드     | 인용이 원문에 있는지는 문자열 비교로 확인 가능 |
| reasons_faithful     | LLM 심사 | 이유가 회사 정보와 모순되는지는 의미 판단 필요 (선택) |
"""

from __future__ import annotations

import os
from typing import Literal

from mlflow.genai.scorers import scorer


@scorer
def verdict_correct(outputs: dict, expectations: dict) -> bool:
    return outputs.get("verdict") == expectations["expected_verdict"]


@scorer
def no_false_rejection(outputs: dict, expectations: dict) -> bool:
    """정답이 '지원 가능'이나 '확인 필요'인데 '지원 불가'로 단정하면 실패."""
    return not (
        outputs.get("verdict") == "ineligible" and expectations["expected_verdict"] != "ineligible"
    )


@scorer
def evidence_grounded(outputs: dict) -> bool:
    """판정에 쓴 모든 인용이 공고 원문에 실제로 있으면 통과."""
    return all(c.get("quote_valid") is not False for c in outputs.get("checks", []))


JUDGE_INSTRUCTIONS = """\
당신은 창업지원사업 자격 판정 결과를 검수하는 심사위원입니다.

{{ outputs }}

위 데이터의 company는 회사 정보이고, llm_checks는 AI가 자격 요건별로 내린 판정(status)과
그 이유(reason)입니다.

각 이유가 회사 정보의 사실과 모순되거나, 회사 정보에 없는 사실을 지어냈는지 확인하세요.
- 회사 정보에 근거해 판단했거나, 정보가 없어 unknown으로 둔 경우는 문제없습니다.
- 회사 정보와 다른 사실(업종, 직원 수, 소재지, 인증 등)을 근거로 삼았다면 문제입니다.

모든 이유가 회사 정보와 일치하면 yes, 하나라도 모순되거나 지어낸 사실이 있으면 no로 답하세요.
"""


def reasons_faithful_judge():
    """LLM 심사위원. 비용이 들어 `--llm-judge`를 줄 때만 사용한다."""
    from mlflow.genai.judges import make_judge

    model = os.getenv("JUDGE_MODEL", "claude-haiku-5-5")
    return make_judge(
        name="reasons_faithful",
        instructions=JUDGE_INSTRUCTIONS,
        model=f"anthropic:/{model}",
        feedback_value_type=Literal["yes", "no"],
    )


CODE_SCORERS = [verdict_correct, no_false_rejection, evidence_grounded]

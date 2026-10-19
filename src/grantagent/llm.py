"""Claude 구조화 출력 호출.

`client.messages.parse()`가 더 간단하지만, MLflow anthropic autolog는 `messages.create`만
감시하므로 parse를 쓰면 LLM 호출이 트레이스에 남지 않는다. 그래서 create + output_config로
JSON schema를 직접 넘기고, 응답을 pydantic으로 검증한다.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    model: str

    def structured(
        self, prompt: str, schema: type[T], max_tokens: int = 4096
    ) -> tuple[T, dict]: ...


class ClaudeLLM:
    def __init__(self, model: str, client=None):
        import anthropic

        self.model = model
        # SDK 기본 타임아웃은 10분이라, 응답이 멈춘 요청 하나가 평가 전체를 수십 분 붙잡는다.
        # 판정 한 번은 보통 수 초 안에 끝나므로 짧게 끊고 재시도한다.
        self.client = client or anthropic.Anthropic(
            timeout=float(os.getenv("LLM_TIMEOUT_S", "60")), max_retries=3
        )
        self._schema_cache: dict[type, dict] = {}

    def _schema(self, schema: type[BaseModel]) -> dict:
        if schema not in self._schema_cache:
            from anthropic import transform_schema

            self._schema_cache[schema] = transform_schema(schema)
        return self._schema_cache[schema]

    def structured(self, prompt: str, schema: type[T], max_tokens: int = 4096) -> tuple[T, dict]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": self._schema(schema)}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return schema.model_validate_json(text), usage

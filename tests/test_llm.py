"""ClaudeLLM이 구조화 출력 요청을 올바르게 만들고 응답을 검증하는지 (API 호출 없이)."""

from types import SimpleNamespace

from grantagent.llm import ClaudeLLM
from grantagent.schemas import JudgeOutput


class FakeMessages:
    def __init__(self, text):
        self.text, self.kwargs = text, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text=self.text),
            ],
            usage=SimpleNamespace(input_tokens=321, output_tokens=45),
        )


def test_structured_request_and_parse():
    body = '{"checks": [{"requirement": "제조업", "status": "pass", "reason": "제조"}]}'
    messages = FakeMessages(body)
    llm = ClaudeLLM("claude-sonnet-5-5", client=SimpleNamespace(messages=messages))

    out, usage = llm.structured("프롬프트", JudgeOutput)

    assert out.checks[0].status == "pass"
    assert usage == {"input_tokens": 321, "output_tokens": 45}
    fmt = messages.kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert messages.kwargs["model"] == "claude-sonnet-5-5"

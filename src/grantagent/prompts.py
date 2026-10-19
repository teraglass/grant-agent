"""프롬프트를 코드에서 분리해 MLflow 프롬프트 레지스트리로 버전 관리한다.

- `prompts/*.txt`가 원본. `make prompts`로 레지스트리에 등록 (내용이 바뀐 경우에만 새 버전)
- 실행 시에는 `prompts:/grant-extract@production`처럼 alias로 불러온다
- 레지스트리에 연결할 수 없으면 로컬 파일로 동작 (버전은 "local")
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from grantagent.config import PROMPT_DIR, tracking_uri

log = logging.getLogger(__name__)

PROMPTS = {"extract": "grant-extract", "judge": "grant-judge"}
PRODUCTION = "production"


@dataclass(frozen=True)
class Prompt:
    key: str
    name: str
    version: str
    template: str

    def format(self, **kwargs) -> str:
        text = self.template
        for k, v in kwargs.items():
            text = text.replace("{{" + k + "}}", str(v))
        return text

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def local_template(key: str) -> str:
    return (PROMPT_DIR / f"{key}.txt").read_text(encoding="utf-8")


def server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """MLflow 클라이언트는 연결 실패 시 수 분간 재시도하므로, 먼저 짧게 확인한다."""
    if not uri.startswith(("http://", "https://")):
        return True  # sqlite 등 로컬 저장소
    import httpx

    try:
        return httpx.get(uri.rstrip("/") + "/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def load(key: str, alias: str = PRODUCTION, version: int | None = None) -> Prompt:
    """레지스트리에서 alias(기본 production) 또는 특정 버전을 불러온다."""
    name = PROMPTS[key]
    if not server_reachable(tracking_uri()):
        log.warning("MLflow 서버(%s)에 연결할 수 없어 로컬 프롬프트 사용: %s", tracking_uri(), name)
        return Prompt(key, name, "local", local_template(key))
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        uri = f"prompts:/{name}/{version}" if version else f"prompts:/{name}@{alias}"
        pv = mlflow.genai.load_prompt(uri)
        return Prompt(key, name, str(pv.version), pv.template)
    except Exception as e:  # 서버 미실행, 미등록 등
        log.warning("prompt registry unavailable for %s (%s) → local file", name, type(e).__name__)
        return Prompt(key, name, "local", local_template(key))


def register(commit_message: str, set_production_if_missing: bool = True) -> dict[str, str]:
    """로컬 파일 내용이 최신 버전과 다를 때만 새 버전 등록."""
    if not server_reachable(tracking_uri()):
        raise SystemExit(
            f"MLflow 서버({tracking_uri()})에 연결할 수 없습니다.\n"
            "  - mlops-eurosat 폴더에서 `make up`, 또는 이 폴더에서 `make mlflow-up`\n"
            "  - Docker(OrbStack)가 켜져 있는지 확인"
        )
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    result = {}
    for key, name in PROMPTS.items():
        template = local_template(key)
        try:
            latest = mlflow.genai.load_prompt(f"prompts:/{name}@latest")
        except Exception:
            latest = None
        if latest is not None and latest.template == template:
            result[name] = f"v{latest.version} (변경 없음)"
            version = latest.version
        else:
            pv = mlflow.genai.register_prompt(
                name=name, template=template, commit_message=commit_message
            )
            version = pv.version
            result[name] = f"v{version} 등록"
        if set_production_if_missing:
            try:
                client.get_prompt_version_by_alias(name, PRODUCTION)
            except Exception:
                mlflow.genai.set_prompt_alias(name, alias=PRODUCTION, version=int(version))
                result[name] += f" → @{PRODUCTION}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="prompts/*.txt를 MLflow 프롬프트 레지스트리에 등록"
    )
    parser.add_argument("-m", "--message", default="update prompts")
    args = parser.parse_args()
    for name, status in register(args.message).items():
        print(f"[prompts] {name}: {status}")


if __name__ == "__main__":
    main()

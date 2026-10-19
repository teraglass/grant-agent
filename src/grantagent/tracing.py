from __future__ import annotations

import os

import mlflow

from grantagent.config import EXPERIMENT, tracking_uri

# S3 저장소 사용 시 presigned URL(Docker 내부 주소) 대신 MLflow 서버 경유
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")


def setup(experiment: str = EXPERIMENT) -> None:
    """MLflow 연결 + Claude 호출 자동 트레이싱."""
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    mlflow.anthropic.autolog()

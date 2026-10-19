ENV_FILE := $(if $(wildcard .env),--env-file .env,)
RUN := uv run $(ENV_FILE)

.PHONY: help setup mlflow-up mlflow-down inspect fetch prompts list assess label label-draft eval promote test lint format

help:
	@echo "setup         의존성 설치 + .env 생성"
	@echo "mlflow-up     MLflow 서버 실행 (eurosat 프로젝트 MLflow가 떠 있으면 생략)"
	@echo "inspect       K-Startup API 첫 공고의 실제 필드 확인"
	@echo "fetch         공고 수집 → data/announcements.jsonl 스냅샷"
	@echo "prompts       prompts/*.txt를 MLflow 프롬프트 레지스트리에 등록"
	@echo "list          스냅샷 공고 · 프로필 목록"
	@echo "assess        한 건 판정  예) make assess A=<공고ID> P=p02"
	@echo "label         평가셋 정답 라벨링 (label-draft: 에이전트 초안 함께 보기)"
	@echo "eval          평가셋 평가 → MLflow run  예) make eval MODEL=claude-haiku-5-5 EX=2 JU=2 JUDGE=1"
	@echo "promote       승격 게이트  예) make promote R=<run_id> (DRY=1 비교만, FORCE=1 롤백)"
	@echo "test / lint   테스트 / 린트"

setup:
	uv sync
	@test -f .env || cp .env.example .env
	@echo ".env에 ANTHROPIC_API_KEY, DATA_GO_KR_API_KEY를 넣어 주세요"

mlflow-up:
	docker compose up -d --build
	@echo "MLflow UI: http://127.0.0.1:5001"

mlflow-down:
	docker compose down

inspect:
	$(RUN) python -m grantagent.kstartup --inspect

fetch:
	$(RUN) python -m grantagent.kstartup --limit 60

prompts:
	$(RUN) python -m grantagent.prompts -m "$(or $(M),update prompts)"

list:
	$(RUN) grantagent list

assess:
	$(RUN) grantagent assess $(A) $(P)

label:
	$(RUN) python -m grantagent.label --n-announcements 8

label-draft:
	$(RUN) python -m grantagent.label --n-announcements 8 --draft

eval:
	$(RUN) python -m grantagent.evaluate \
		$(if $(MODEL),--model $(MODEL)) $(if $(EX),--extract-version $(EX)) \
		$(if $(JU),--judge-version $(JU)) $(if $(JUDGE),--llm-judge) $(if $(LIMIT),--limit $(LIMIT))

promote:
	$(RUN) python -m grantagent.gate $(R) $(if $(DRY),--dry-run) $(if $(FORCE),--force)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

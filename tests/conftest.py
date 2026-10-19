import os
from datetime import date

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from grantagent.schemas import Announcement, CompanyProfile  # noqa: E402

AS_OF = date(2026, 10, 18)


@pytest.fixture
def ann() -> Announcement:
    return Announcement(
        id="T001",
        title="2026년 청년창업 지원사업 (테스트)",
        org="테스트진흥원",
        region="서울",
        target="공고일 기준 창업 3년 이내 기업 또는 예비창업자",
        excluded="휴·폐업 중인 기업, 국세 체납 기업",
        target_age="만 39세 이하",
        business_period="예비창업자, 3년미만",
        content="서울 소재 제조·ICT 분야 창업기업의 시제품 제작을 지원합니다.",
    )


@pytest.fixture
def young_seoul() -> CompanyProfile:
    return CompanyProfile(
        id="y1",
        name="청년서울",
        stage="창업기업",
        founded_date=date(2025, 1, 1),
        ceo_birth_date=date(1995, 5, 1),
        region="서울특별시",
        industry="ICT 소프트웨어",
    )


@pytest.fixture
def old_busan() -> CompanyProfile:
    return CompanyProfile(
        id="o1",
        name="부산오래",
        stage="창업기업",
        founded_date=date(2015, 1, 1),
        ceo_birth_date=date(1970, 1, 1),
        region="부산",
        industry="수산 유통",
    )

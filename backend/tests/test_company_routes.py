import copy
import json
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.dependencies import (
    get_benchmark_catalog_service,
    get_company_analysis_provider,
    get_gcis_client,
)
from app.schemas.company import CompanyCapital, CompanyData, CompanyStatus
from app.services.benchmark_catalog import (
    BenchmarkCatalogInvalidError,
    BenchmarkCatalogService,
)
from app.services.benchmark_snapshot import BenchmarkSnapshotStore
from app.services.gcis import (
    GCISConnectionError,
    GCISInvalidResponseError,
    GCISNotFoundError,
    GCISTimeoutError,
    GCISUpstreamHTTPError,
)
from app.services.gcis_response_cache import GCISResponseCache
from app.services.llm_prompt import CompanyAnalysisPromptPackage
from app.services.llm_provider import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMIncompleteResponseError,
    LLMProviderResult,
    LLMProviderUsage,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUpstreamError,
)


pytestmark = pytest.mark.anyio

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = WORKSPACE_ROOT / "samples" / "gcis"
LLM_SAMPLE_DIR = WORKSPACE_ROOT / "samples" / "llm"


def load_sample(filename: str) -> list[dict[str, object]]:
    with (SAMPLE_DIR / filename).open(encoding="utf-8") as sample_file:
        payload = json.load(sample_file)
    assert isinstance(payload, list)
    return payload


def load_route_analysis_output() -> dict[str, object]:
    payload = json.loads(
        (LLM_SAMPLE_DIR / "company-analysis-output-v1.example.json").read_text(
            encoding="utf-8"
        )
    )
    # The route fixture uses a 30-company synthetic benchmark whose Acer PR is 100,
    # while the saved LLM example uses the production F-category benchmark (PR 99.2).
    payload["findings"][3]["observation"] = (
        "年資與登記資本的同業綜合相對位置為 PR 100.0。"
    )
    return payload


def build_test_benchmark_service(tmp_path: Path) -> BenchmarkCatalogService:
    as_of = date(2026, 8, 1)
    version = "benchmark-2026-08-01-F-v1"
    source_version = "gcis-benchmark-F-2026-08-v1"
    database_path = tmp_path / "benchmark.sqlite3"
    companies = [
        CompanyData(
            tax_id=f"{number:08d}",
            name=f"同業樣本 {number}",
            status=CompanyStatus(code="01", description="核准設立"),
            capital=CompanyCapital(registered=number * 100_000),
            established_at=as_of - timedelta(days=365 * number),
        )
        for number in range(1, 31)
    ]
    metadata = BenchmarkSnapshotStore(database_path).create_snapshot(
        companies,
        version=version,
        as_of=as_of,
        source="GCIS nationwide F-category batch manifest",
        source_version=source_version,
        created_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        industry_code_override="F",
        industry_mapping_version="gcis-regional-category-membership-v1",
    )
    entry = {
        "snapshot_version": version,
        "source_version": source_version,
        "sample_count": 30,
        "checksum_sha256": metadata.checksum_sha256,
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": "benchmark-catalog-2026-08-01-v1",
                "as_of": "2026-08-01",
                "industry_mapping_version": (
                    "gcis-regional-category-membership-v1"
                ),
                "membership_sample_count": 300,
                "membership_count_note": "Test membership count.",
                "sqlite_integrity_check": "ok",
                "categories": {
                    code: (
                        dict(entry)
                        if code == "F"
                        else {
                            **entry,
                            "snapshot_version": f"unused-{code}-snapshot",
                            "source_version": f"unused-{code}-source",
                        }
                    )
                    for code in "ABCDEFGHIJ"
                },
            }
        ),
        encoding="utf-8",
    )
    return BenchmarkCatalogService(
        catalog_path,
        database_path,
        expected_catalog_version="benchmark-catalog-2026-08-01-v1",
    )


@dataclass
class FakeGCISClient:
    basic_outcome: object = field(
        default_factory=lambda: load_sample("company-basic-20828393.json")
    )
    business_outcome: object = field(
        default_factory=lambda: load_sample("company-business-20828393.json")
    )
    search_outcome: object = field(default_factory=list)
    basic_calls: list[str] = field(default_factory=list)
    business_calls: list[str] = field(default_factory=list)
    search_calls: list[tuple[str, str, int]] = field(default_factory=list)

    async def get_company_basic(self, tax_id: str) -> list[dict[str, object]]:
        self.basic_calls.append(tax_id)
        return self._resolve(self.basic_outcome)

    async def get_company_business_items(
        self,
        tax_id: str,
    ) -> list[dict[str, object]]:
        self.business_calls.append(tax_id)
        return self._resolve(self.business_outcome)

    async def search_companies(
        self,
        keyword: str,
        company_status: str = "01",
        limit: int = 10,
    ) -> list[dict[str, object]]:
        self.search_calls.append((keyword, company_status, limit))
        return self._resolve(self.search_outcome)

    @staticmethod
    def _resolve(outcome: object) -> list[dict[str, object]]:
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, list)
        return outcome


@dataclass
class FakeCompanyAnalysisProvider:
    outcome: object = field(default_factory=load_route_analysis_output)
    calls: list[CompanyAnalysisPromptPackage] = field(default_factory=list)
    availability_checks: int = 0
    provider_name: str = "fake"
    model_name: str = "fake-company-analysis-v1"

    def ensure_available(self) -> None:
        self.availability_checks += 1

    async def generate(
        self,
        prompt: CompanyAnalysisPromptPackage,
    ) -> LLMProviderResult:
        self.calls.append(prompt)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        assert isinstance(self.outcome, dict)
        return LLMProviderResult(
            output=copy.deepcopy(self.outcome),
            provider=self.provider_name,
            model=self.model_name,
            response_id="resp_test_123",
            request_id="req_test_123",
            created_at=datetime(2026, 8, 23, 4, 30, tzinfo=timezone.utc),
            service_tier="default",
            usage=LLMProviderUsage(
                input_tokens=1200,
                output_tokens=450,
                total_tokens=1650,
            ),
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_gcis_client(app: FastAPI) -> FakeGCISClient:
    fake = FakeGCISClient()
    app.dependency_overrides[get_gcis_client] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def benchmark_service(app: FastAPI, tmp_path: Path) -> BenchmarkCatalogService:
    service = build_test_benchmark_service(tmp_path)
    app.dependency_overrides[get_benchmark_catalog_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_benchmark_catalog_service, None)


@pytest.fixture
def analysis_provider(app: FastAPI) -> FakeCompanyAnalysisProvider:
    provider = FakeCompanyAnalysisProvider()
    app.dependency_overrides[get_company_analysis_provider] = lambda: provider
    yield provider
    app.dependency_overrides.pop(get_company_analysis_provider, None)


@pytest.mark.parametrize(
    "tax_id",
    [
        "123",
        "123456789",
        "abcdefgh",
        "２０８２８３９３",
        "٢٠٨٢٨٣٩٣",
        "१२३४५६७८",
        "2082839８",
    ],
)
async def test_company_route_returns_public_error_for_invalid_tax_id(
    tax_id: str,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    response = await client.get(f"/api/v1/companies/{tax_id}")

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "INVALID_TAX_ID",
            "message": "請輸入 8 碼統一編號。",
            "retryable": False,
        }
    }
    assert fake_gcis_client.basic_calls == []
    assert fake_gcis_client.business_calls == []


@pytest.mark.parametrize(
    ("method", "path_suffix"),
    [
        ("GET", ""),
        ("GET", "/bizscore"),
        ("POST", "/analysis"),
    ],
)
@pytest.mark.parametrize(
    "tax_id",
    ["２０８２８３９３", "٢٠٨٢٨٣٩٣", "१२३४५६७८", "2082839８"],
)
async def test_company_routes_reject_non_ascii_tax_ids_before_dependencies(
    method: str,
    path_suffix: str,
    tax_id: str,
    app: FastAPI,
    client: AsyncClient,
) -> None:
    dependency_calls: list[str] = []

    def unexpected_dependency(name: str):
        def dependency():
            dependency_calls.append(name)
            raise AssertionError(f"{name} dependency must not be resolved")

        return dependency

    app.dependency_overrides[get_gcis_client] = unexpected_dependency("GCIS")
    app.dependency_overrides[get_benchmark_catalog_service] = (
        unexpected_dependency("Benchmark")
    )
    app.dependency_overrides[get_company_analysis_provider] = (
        unexpected_dependency("LLM")
    )

    response = await client.request(
        method,
        f"/api/v1/companies/{tax_id}{path_suffix}",
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "INVALID_TAX_ID",
            "message": "請輸入 8 碼統一編號。",
            "retryable": False,
        }
    }
    assert dependency_calls == []


async def test_bizscore_route_returns_public_error_for_invalid_tax_id(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    response = await client.get("/api/v1/companies/123/bizscore")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_TAX_ID"
    assert fake_gcis_client.basic_calls == []
    assert fake_gcis_client.business_calls == []


async def test_company_route_merges_and_normalizes_both_gcis_records(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    response = await client.get("/api/v1/companies/20828393")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["tax_id"] == "20828393"
    assert payload["data"]["name"] == "宏碁股份有限公司"
    assert payload["data"]["status"] == {
        "code": "01",
        "description": "核准設立",
    }
    assert payload["data"]["capital"] == {
        "registered": 40_000_000_000,
        "paid_in": 30_478_538_280,
        "currency": "TWD",
    }
    assert payload["data"]["established_at"] == "1979-07-18"
    assert payload["data"]["last_changed_at"] == "2026-06-22"
    assert len(payload["data"]["business_items"]) == 22
    assert payload["meta"]["source"] == "GCIS"
    assert payload["meta"]["provider"] == "經濟部商業發展署"
    assert payload["meta"]["fetched_at"].endswith("+08:00")
    assert payload["meta"]["partial"] is False
    assert payload["meta"]["warnings"] == []
    assert payload["meta"]["data_freshness"] == "live"
    assert payload["meta"]["fallback_reason"] is None
    assert fake_gcis_client.basic_calls == ["20828393"]
    assert fake_gcis_client.business_calls == ["20828393"]


async def test_company_route_serves_recent_snapshot_on_transient_gcis_failure(
    app: FastAPI,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    clock_value = 0.0
    epoch = datetime(2026, 8, 26, tzinfo=timezone.utc)

    def monotonic() -> float:
        return clock_value

    def utcnow() -> datetime:
        return epoch + timedelta(seconds=clock_value)

    app.state.gcis_response_cache = GCISResponseCache(
        max_entries=4,
        ttl_seconds=10,
        stale_if_error_seconds=20,
        monotonic=monotonic,
        utcnow=utcnow,
    )
    live_response = await client.get("/api/v1/companies/20828393")
    live_payload = live_response.json()
    clock_value = 11
    fake_gcis_client.basic_outcome = GCISConnectionError("connection refused")
    fake_gcis_client.business_outcome = GCISConnectionError("connection refused")

    fallback_response = await client.get("/api/v1/companies/20828393")

    assert live_response.status_code == 200
    assert fallback_response.status_code == 200
    fallback_payload = fallback_response.json()
    assert fallback_payload["data"] == live_payload["data"]
    assert fallback_payload["meta"]["fetched_at"] == (
        live_payload["meta"]["fetched_at"]
    )
    assert fallback_payload["meta"]["data_freshness"] == "stale_cache"
    assert fallback_payload["meta"]["fallback_reason"] == "GCIS_CONNECTION"
    assert fake_gcis_client.basic_calls == ["20828393", "20828393"]
    assert fake_gcis_client.business_calls == ["20828393", "20828393"]


async def test_business_endpoint_failure_returns_partial_company_data(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    fake_gcis_client.business_outcome = GCISTimeoutError("timed out")

    response = await client.get("/api/v1/companies/20828393")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["name"] == "宏碁股份有限公司"
    assert payload["data"]["business_items"] == []
    assert payload["meta"]["partial"] is True
    assert any("response is partial" in item for item in payload["meta"]["warnings"])
    assert any("timed out" in item for item in payload["meta"]["warnings"])


async def test_bizscore_route_uses_catalog_snapshot_and_returns_five_dimensions(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
) -> None:
    response = await client.get("/api/v1/companies/20828393/bizscore")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["company"]["tax_id"] == "20828393"
    result = payload["data"]["bizscore"]
    assert result["version"] == "1.0"
    assert result["score"] == 88
    assert result["band"] == "公開資料呈現較穩健"
    assert result["coverage"] == 1
    assert result["provisional"] is False
    assert result["as_of"] == "2026-08-01"
    assert len(result["dimensions"]) == 5
    assert result["benchmark"]["catalog_version"] == (
        "benchmark-catalog-2026-08-01-v1"
    )
    assert result["benchmark"]["industry_code"] == "F"
    assert result["benchmark"]["snapshot_version"] == (
        "benchmark-2026-08-01-F-v1"
    )
    assert result["benchmark"]["sample_count"] == 30
    assert result["peer_benchmark"]["industry_code"] == "F"
    assert result["peer_benchmark"]["sample_size"] == 30
    assert result["peer_benchmark"]["company_age_median"] == 15.5
    assert result["peer_benchmark"]["registered_capital_median"] == 1_550_000
    assert isinstance(result["peer_benchmark"]["age_percentile"], float)
    assert isinstance(result["peer_benchmark"]["capital_percentile"], float)
    assert isinstance(result["peer_benchmark"]["peer_index"], float)
    assert fake_gcis_client.basic_calls == ["20828393"]
    assert fake_gcis_client.business_calls == ["20828393"]


async def test_bizscore_route_does_not_score_partial_a3_without_status_code(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
) -> None:
    fake_gcis_client.business_outcome = GCISTimeoutError("timed out")

    response = await client.get("/api/v1/companies/20828393/bizscore")

    assert response.status_code == 200
    result = response.json()["data"]["bizscore"]
    assert result["score"] is None
    assert result["coverage"] == 0.55
    assert result["provisional"] is False
    assert result["missing_dimensions"] == [
        "registration_status",
        "peer_relative_position",
    ]
    assert result["benchmark"]["snapshot_version"] is None
    assert response.json()["meta"]["partial"] is True


async def test_bizscore_route_does_not_score_a_commanded_dissolution(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
) -> None:
    fake_gcis_client.basic_outcome[0]["Company_Status_Desc"] = "核准設立，但已命令解散"
    fake_gcis_client.business_outcome[0]["Company_Status"] = "02"
    fake_gcis_client.business_outcome[0]["Company_Status_Desc"] = (
        "核准設立，但已命令解散"
    )

    response = await client.get("/api/v1/companies/20828393/bizscore")

    assert response.status_code == 200
    result = response.json()["data"]["bizscore"]
    assert result["score"] is None
    assert result["band"] is None
    assert result["dimensions"][0]["score"] == 0
    assert result["status_cap"] is None
    assert "official_status=核准設立，但已命令解散" in result["dimensions"][0][
        "evidence"
    ]


async def test_bizscore_route_withholds_score_on_conflicting_source_status(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
) -> None:
    fake_gcis_client.basic_outcome[0]["Company_Status_Desc"] = "解散"
    fake_gcis_client.business_outcome[0]["Company_Status"] = "01"
    fake_gcis_client.business_outcome[0]["Company_Status_Desc"] = "核准設立"

    response = await client.get("/api/v1/companies/20828393/bizscore")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["company"]["status"] == {
        "code": None, "description": "來源登記狀態不一致，待確認",
    }
    result = payload["data"]["bizscore"]
    assert result["score"] is None
    assert result["band"] is None
    assert result["status_cap"] is None
    assert result["dimensions"][0]["score"] is None
    assert result["dimensions"][0]["available"] is False
    assert "registration_status" in result["missing_dimensions"]
    assert payload["meta"]["partial"] is True
    assert any("GCIS_STATUS_CONFLICT" in warning for warning in payload["meta"]["warnings"])


async def test_bizscore_route_maps_missing_catalog_to_public_503(
    app: FastAPI,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    tmp_path: Path,
) -> None:
    service = BenchmarkCatalogService(
        tmp_path / "missing-catalog.json",
        tmp_path / "missing.sqlite3",
        expected_catalog_version="benchmark-catalog-2026-08-01-v1",
    )
    app.dependency_overrides[get_benchmark_catalog_service] = lambda: service

    response = await client.get("/api/v1/companies/20828393/bizscore")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "BENCHMARK_UNAVAILABLE",
            "message": "目前無法讀取 BizScore 同業基準資料，請稍後再試。",
            "retryable": True,
        }
    }


async def test_bizscore_route_maps_catalog_mismatch_to_public_500(
    app: FastAPI,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    class InvalidBenchmarkService:
        def calculate_company_bizscore(self, *args, **kwargs):
            raise BenchmarkCatalogInvalidError("checksum mismatch")

    app.dependency_overrides[get_benchmark_catalog_service] = (
        lambda: InvalidBenchmarkService()
    )

    response = await client.get("/api/v1/companies/20828393/bizscore")

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "BENCHMARK_CONFIGURATION_ERROR",
            "message": "BizScore 同業基準資料設定不一致，暫時無法計分。",
            "retryable": False,
        }
    }


async def test_missing_basic_company_maps_to_not_found(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    fake_gcis_client.basic_outcome = GCISNotFoundError("not found")

    response = await client.get("/api/v1/companies/00000000")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "COMPANY_NOT_FOUND",
            "message": "找不到此統編的公司登記資料。",
            "retryable": False,
        }
    }


async def test_basic_endpoint_timeout_maps_to_gateway_timeout(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    fake_gcis_client.basic_outcome = GCISTimeoutError("timed out")

    response = await client.get("/api/v1/companies/20828393")

    assert response.status_code == 504
    assert response.json()["detail"] == {
        "code": "UPSTREAM_TIMEOUT",
        "message": "政府資料服務回應逾時，請稍後再試。",
        "retryable": True,
    }


@pytest.mark.parametrize(
    ("upstream_status", "expected_status", "expected_detail"),
    [
        (
            429,
            503,
            {
                "code": "UPSTREAM_RATE_LIMITED",
                "message": "政府資料服務目前流量過高，請稍後再試。",
                "retryable": True,
            },
        ),
        (
            503,
            503,
            {
                "code": "UPSTREAM_UNAVAILABLE",
                "message": "政府資料服務暫時無法使用，請稍後再試。",
                "retryable": True,
            },
        ),
        (
            403,
            502,
            {
                "code": "UPSTREAM_REQUEST_REJECTED",
                "message": "政府資料服務拒絕了目前的查詢格式。",
                "retryable": False,
            },
        ),
    ],
)
async def test_basic_endpoint_maps_classified_upstream_http_errors(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    upstream_status: int,
    expected_status: int,
    expected_detail: dict[str, object],
) -> None:
    fake_gcis_client.basic_outcome = GCISUpstreamHTTPError(upstream_status)

    response = await client.get("/api/v1/companies/20828393")

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail


async def test_search_requires_at_least_two_characters(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    response = await client.get("/api/v1/companies/search", params={"q": "宏"})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert fake_gcis_client.search_calls == []


async def test_search_defaults_to_active_status_and_normalizes_results(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    fake_gcis_client.search_outcome = [
        {
            "Business_Accounting_NO": "20828393",
            "Company_Name": "宏碁股份有限公司",
            "Company_Status": "01",
            "Company_Status_Desc": "核准設立",
            "Capital_Stock_Amount": 40_000_000_000,
            "Company_Setup_Date": "0680718",
            "Change_Of_Approval_Data": "1150622",
        }
    ]

    response = await client.get("/api/v1/companies/search", params={"q": "宏碁"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] == [
        {
            "tax_id": "20828393",
            "name": "宏碁股份有限公司",
            "status": {"code": "01", "description": "核准設立"},
            "registered_capital": 40_000_000_000,
            "established_at": "1979-07-18",
            "last_changed_at": "2026-06-22",
        }
    ]
    assert payload["meta"]["partial"] is False
    assert payload["meta"]["warnings"] == []
    assert fake_gcis_client.search_calls == [("宏碁", "01", 10)]


async def test_empty_search_is_a_successful_empty_list(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    response = await client.get("/api/v1/companies/search", params={"q": "查無資料"})

    assert response.status_code == 200
    assert response.json()["data"] == []


async def test_invalid_search_response_maps_to_bad_gateway(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    fake_gcis_client.search_outcome = GCISInvalidResponseError("invalid")

    response = await client.get("/api/v1/companies/search", params={"q": "宏碁"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "UPSTREAM_INVALID_RESPONSE"


async def test_analysis_route_returns_verified_input_validated_output_and_audit_meta(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
    analysis_provider: FakeCompanyAnalysisProvider,
) -> None:
    response = await client.post("/api/v1/companies/20828393/analysis")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["input"]["company"]["tax_id"] == "20828393"
    assert payload["data"]["input"]["bizscore"]["score"] == 88
    assert payload["data"]["input"]["source_meta"]["source"] == "GCIS"
    assert payload["data"]["analysis"]["status"] == "completed"
    assert payload["data"]["analysis"]["provenance"]["company_tax_id"] == (
        "20828393"
    )
    assert payload["meta"] == {
        "provider": "fake",
        "model": "fake-company-analysis-v1",
        "provider_response_id": "resp_test_123",
        "provider_request_id": "req_test_123",
        "service_tier": "default",
        "generated_at": "2026-08-23T04:30:00Z",
        "prompt_version": "1.0",
        "system_prompt_sha256": (
            "6f070fb298e35e8601ed1d42d76986c01dcc036fdf65cc982e868816225a1234"
        ),
        "input_schema_version": "1.0",
        "output_schema_version": "1.0",
        "input_sha256": payload["meta"]["input_sha256"],
        "output_sha256": payload["meta"]["output_sha256"],
        "attempts": 1,
        "duration_ms": payload["meta"]["duration_ms"],
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 450,
            "total_tokens": 1650,
        },
    }
    assert len(payload["meta"]["input_sha256"]) == 64
    assert len(payload["meta"]["output_sha256"]) == 64
    assert payload["meta"]["duration_ms"] >= 0
    assert analysis_provider.availability_checks == 1
    assert len(analysis_provider.calls) == 1
    prompt = analysis_provider.calls[0]
    assert [message.role for message in prompt.messages] == ["system", "user"]
    assert prompt.response_schema["$id"].endswith(
        "llm-company-analysis-output-v1.schema.json"
    )
    assert "你是 BizCheck AI" not in response.text
    assert fake_gcis_client.basic_calls == ["20828393"]
    assert fake_gcis_client.business_calls == ["20828393"]


async def test_analysis_route_returns_not_configured_before_fetching_company(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
) -> None:
    response = await client.post("/api/v1/companies/20828393/analysis")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "AI_ANALYSIS_NOT_CONFIGURED",
            "message": "AI 企業分析尚未啟用，請先設定分析服務。",
            "retryable": False,
        }
    }
    assert fake_gcis_client.basic_calls == []
    assert fake_gcis_client.business_calls == []


async def test_analysis_route_rejects_invalid_tax_id_before_any_work(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    analysis_provider: FakeCompanyAnalysisProvider,
) -> None:
    response = await client.post("/api/v1/companies/123/analysis")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_TAX_ID"
    assert analysis_provider.availability_checks == 0
    assert analysis_provider.calls == []
    assert fake_gcis_client.basic_calls == []
    assert fake_gcis_client.business_calls == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code", "retryable"),
    [
        (LLMTimeoutError("timeout"), 504, "AI_ANALYSIS_TIMEOUT", True),
        (
            LLMConnectionError("connection"),
            503,
            "AI_ANALYSIS_UNAVAILABLE",
            True,
        ),
        (
            LLMAuthenticationError("auth"),
            503,
            "AI_PROVIDER_AUTH_FAILED",
            False,
        ),
        (
            LLMRateLimitError("rate"),
            503,
            "AI_PROVIDER_RATE_LIMITED",
            True,
        ),
        (
            LLMRequestRejectedError("request"),
            502,
            "AI_PROVIDER_REQUEST_REJECTED",
            False,
        ),
        (
            LLMUpstreamError("upstream"),
            502,
            "AI_PROVIDER_ERROR",
            True,
        ),
        (
            LLMIncompleteResponseError("incomplete"),
            502,
            "AI_ANALYSIS_INCOMPLETE",
            False,
        ),
    ],
)
async def test_analysis_route_maps_provider_failures_to_public_errors(
    error: Exception,
    expected_status: int,
    expected_code: str,
    retryable: bool,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
    analysis_provider: FakeCompanyAnalysisProvider,
) -> None:
    analysis_provider.outcome = error

    response = await client.post("/api/v1/companies/20828393/analysis")

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert response.json()["detail"]["retryable"] is retryable
    assert str(error) not in response.text
    assert len(analysis_provider.calls) == 1


@pytest.mark.parametrize("invalid_kind", ["schema", "forbidden_claim"])
async def test_analysis_route_fails_closed_for_invalid_or_unsafe_output(
    invalid_kind: str,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
    analysis_provider: FakeCompanyAnalysisProvider,
) -> None:
    assert isinstance(analysis_provider.outcome, dict)
    invalid_output = copy.deepcopy(analysis_provider.outcome)
    if invalid_kind == "schema":
        invalid_output.pop("headline")
    else:
        invalid_output["overall_observation"] = "這是安全公司，可以直接合作。"
    analysis_provider.outcome = invalid_output

    response = await client.post("/api/v1/companies/20828393/analysis")

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "AI_ANALYSIS_INVALID_OUTPUT",
            "message": "AI 企業分析未通過資料與安全驗證。",
            "retryable": False,
        }
    }
    assert "安全公司" not in response.text
    assert len(analysis_provider.calls) == 2
    assert "<validation_retry_feedback>" in (
        analysis_provider.calls[1].messages[1].content
    )


async def test_analysis_route_does_not_call_provider_when_gcis_fails(
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
    analysis_provider: FakeCompanyAnalysisProvider,
) -> None:
    fake_gcis_client.basic_outcome = GCISTimeoutError("timed out")

    response = await client.post("/api/v1/companies/20828393/analysis")

    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "UPSTREAM_TIMEOUT"
    assert analysis_provider.availability_checks == 1
    assert analysis_provider.calls == []


@pytest.mark.parametrize("status_blocked", [False, True])
@pytest.mark.parametrize("corrupt_summary", [False, True])
async def test_thu_route_enforces_fact_contract_and_preserves_official_status(
    status_blocked: bool,
    corrupt_summary: bool,
    client: AsyncClient,
    fake_gcis_client: FakeGCISClient,
    benchmark_service: BenchmarkCatalogService,
    analysis_provider: FakeCompanyAnalysisProvider,
    monkeypatch,
) -> None:
    from app.services.thu_prompt import build_thu_system_instruction

    if status_blocked:
        fake_gcis_client.basic_outcome = copy.deepcopy(fake_gcis_client.basic_outcome)
        fake_gcis_client.basic_outcome[0]["Company_Status_Desc"] = "核准設立，但已命令解散"
        fake_gcis_client.business_outcome = copy.deepcopy(fake_gcis_client.business_outcome)
        fake_gcis_client.business_outcome[0]["Company_Status"] = "02"
        fake_gcis_client.business_outcome[0]["Company_Status_Desc"] = "核准設立，但已命令解散"
    analysis_provider.provider_name = "thu"

    async def generate(prompt):
        analysis_provider.calls.append(prompt)
        instruction = build_thu_system_instruction(prompt)
        output = json.loads(
            instruction.split("<complete_output_example>\n")[1]
            .split("\n</complete_output_example>")[0]
        )
        if corrupt_summary:
            output["overall_observation"] = "已成立超過拳拳歲"
        return LLMProviderResult(output=output, provider="thu", model="offline-thu")

    monkeypatch.setattr(analysis_provider, "generate", generate)
    response = await client.post("/api/v1/companies/20828393/analysis")
    if corrupt_summary:
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "AI_ANALYSIS_INVALID_OUTPUT"
        assert "拳拳" not in response.text
        assert len(analysis_provider.calls) == 2
    else:
        assert response.status_code == 200
        data = response.json()["data"]
        if status_blocked:
            assert data["input"]["company"]["status"]["description"] == "核准設立，但已命令解散"
            assert data["input"]["bizscore"]["score"] is None
            assert data["analysis"]["status"] == "insufficient_data"
            assert data["analysis"]["findings"] == []
            assert "停業" not in response.text
        else:
            assert data["analysis"]["status"] == "completed"


async def test_cors_preflight_allows_analysis_post(client: AsyncClient) -> None:
    response = await client.options(
        "/api/v1/companies/20828393/analysis",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert "POST" in response.headers["access-control-allow-methods"]
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )


async def test_openapi_registers_company_routes(client: AsyncClient) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]

    assert "/api/v1/companies/search" in paths
    assert "/api/v1/companies/{tax_id}" in paths
    assert "/api/v1/companies/{tax_id}/bizscore" in paths
    assert "/api/v1/companies/{tax_id}/analysis" in paths
    operations = (
        paths["/api/v1/companies/{tax_id}"]["get"],
        paths["/api/v1/companies/{tax_id}/bizscore"]["get"],
        paths["/api/v1/companies/{tax_id}/analysis"]["post"],
    )
    for operation in operations:
        tax_id_parameter = next(
            parameter
            for parameter in operation["parameters"]
            if parameter["in"] == "path" and parameter["name"] == "tax_id"
        )
        assert tax_id_parameter["schema"]["pattern"] == r"^[0-9]{8}$"
    detail_responses = paths["/api/v1/companies/{tax_id}"]["get"]["responses"]
    invalid_tax_id_schema = detail_responses["422"]["content"][
        "application/json"
    ]["schema"]
    assert invalid_tax_id_schema["$ref"].endswith("/ErrorResponse")
    bizscore_responses = paths["/api/v1/companies/{tax_id}/bizscore"]["get"][
        "responses"
    ]
    assert bizscore_responses["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("/CompanyBizScoreResponse")
    analysis_operation = paths["/api/v1/companies/{tax_id}/analysis"]["post"]
    assert analysis_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/CompanyAnalysisResponse")
    for status_code in ("422", "500", "502", "503", "504"):
        assert analysis_operation["responses"][status_code]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/ErrorResponse")

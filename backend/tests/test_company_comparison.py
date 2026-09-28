from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.api.dependencies import (
    get_benchmark_catalog_service,
    get_company_analysis_provider,
    get_gcis_client,
)
from app.schemas.bizscore import BizScoreResult, CompanyBizScoreResponse
from app.schemas.company import CompanyData
from app.schemas.company_comparison import (
    COMPANY_COMPARISON_DISCLAIMER,
    CompanyComparisonItem,
    CompanyComparisonRequest,
)
from app.schemas.llm_comparison_analysis import AI_COMPARISON_DISCLAIMER
from app.services.company_comparison import (
    CompanyComparisonConsistencyError,
    build_company_comparison,
)
from app.services.benchmark_catalog import BenchmarkCatalogUnavailableError
from app.services.gcis import GCISNotFoundError, GCISTimeoutError
from app.services.llm_provider import (
    DisabledCompanyAnalysisProvider,
    LLMProviderResult,
)


pytestmark = pytest.mark.anyio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIRECTORY = PROJECT_ROOT / "samples"
LLM_SAMPLE_DIRECTORY = SAMPLE_DIRECTORY / "llm"
GCIS_SAMPLE_DIRECTORY = SAMPLE_DIRECTORY / "gcis"
GENERATED_AT = datetime(2026, 8, 24, 9, 30, tzinfo=timezone.utc)

TAX_ID_A = "20828393"
TAX_ID_B = "22099131"
TAX_ID_C = "03557311"
COMPANY_NAMES = {
    TAX_ID_A: "宏碁股份有限公司",
    TAX_ID_B: "比較測試乙股份有限公司",
    TAX_ID_C: "比較測試丙股份有限公司",
}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _llm_input_payload() -> dict[str, object]:
    payload = _read_json(
        LLM_SAMPLE_DIRECTORY / "company-analysis-input-v1.example.json"
    )
    assert isinstance(payload, dict)
    return payload


def _company_bizscore_response(
    tax_id: str,
    *,
    name: str | None = None,
    partial: bool = False,
    stale: bool = False,
    provisional: bool = False,
    unscored: bool = False,
    industry_code: str = "F",
    snapshot_version: str | None = None,
    catalog_version: str = "benchmark-catalog-2026-08-01-v1",
    as_of: str = "2026-08-01",
) -> CompanyBizScoreResponse:
    source = copy.deepcopy(_llm_input_payload())
    company = source["company"]
    bizscore = source["bizscore"]
    source_meta = source["source_meta"]
    assert isinstance(company, dict)
    assert isinstance(bizscore, dict)
    assert isinstance(source_meta, dict)

    company["tax_id"] = tax_id
    company["name"] = name or COMPANY_NAMES.get(tax_id, f"測試公司 {tax_id}")
    source_meta["fetched_at"] = "2026-08-24T17:30:00+08:00"
    source_meta["partial"] = partial
    source_meta["warnings"] = (
        ["GCIS A3 business request timed out."] if partial else []
    )
    source_meta["data_freshness"] = "stale_cache" if stale else "live"
    source_meta["fallback_reason"] = "GCIS_CONNECTION" if stale else None

    benchmark = bizscore["benchmark"]
    peer_benchmark = bizscore["peer_benchmark"]
    industry = bizscore["industry"]
    assert isinstance(benchmark, dict)
    assert isinstance(peer_benchmark, dict)
    assert isinstance(industry, dict)
    benchmark["catalog_version"] = catalog_version
    benchmark["industry_code"] = industry_code
    benchmark["snapshot_version"] = snapshot_version or (
        f"benchmark-2026-08-01-{industry_code}-v1"
    )
    peer_benchmark["industry_code"] = industry_code
    peer_benchmark["benchmark_version"] = benchmark["snapshot_version"]
    primary_group = industry["primary_group"]
    assert isinstance(primary_group, dict)
    primary_group["category_code"] = industry_code
    bizscore["as_of"] = as_of
    bizscore["provisional"] = provisional

    if unscored:
        company["status"] = {"code": "03", "description": "解散"}
        bizscore["score"] = None
        bizscore["band"] = None
        bizscore["coverage"] = 1.0
        bizscore["provisional"] = False
        bizscore["status_cap"] = None
        bizscore["missing_dimensions"] = []
        dimensions = bizscore["dimensions"]
        assert isinstance(dimensions, list)
        assert isinstance(dimensions[0], dict)
        dimensions[0]["score"] = 0
        dimensions[0]["available"] = True

    return CompanyBizScoreResponse.model_validate(
        {
            "data": {"company": company, "bizscore": bizscore},
            "meta": source_meta,
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"tax_ids": [TAX_ID_A]}, id="one-company"),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_B, TAX_ID_C, "04595257"]},
            id="four-companies",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_A]},
            id="duplicate-company",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, "1234567A"]},
            id="non-digit-tax-id",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, "１２３４５６７８"]},
            id="non-ascii-tax-id",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, 22099131]},
            id="non-string-tax-id",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_B], "unexpected": True},
            id="unknown-field",
        ),
    ],
)
def test_comparison_request_rejects_invalid_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CompanyComparisonRequest.model_validate(payload)


def test_comparison_request_accepts_two_or_three_unique_ascii_tax_ids() -> None:
    two = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    three = CompanyComparisonRequest(
        tax_ids=[TAX_ID_C, TAX_ID_A, TAX_ID_B]
    )

    assert two.tax_ids == [TAX_ID_A, TAX_ID_B]
    assert three.tax_ids == [TAX_ID_C, TAX_ID_A, TAX_ID_B]


def test_service_aggregates_metrics_context_and_request_order() -> None:
    request = CompanyComparisonRequest(
        tax_ids=[TAX_ID_C, TAX_ID_A, TAX_ID_B]
    )
    responses = [
        _company_bizscore_response(tax_id)
        for tax_id in request.tax_ids
    ]

    result = build_company_comparison(
        request,
        responses,
        generated_at=GENERATED_AT,
    )

    assert [item.input_index for item in result.data.items] == [0, 1, 2]
    assert [item.company.tax_id for item in result.data.items] == request.tax_ids
    assert [item.metrics.bizscore for item in result.data.items] == [88, 88, 88]
    assert all(item.metrics.bizscore_status == "complete" for item in result.data.items)
    assert all(item.metrics.missing == [] for item in result.data.items)
    assert result.data.context.ordering == "request_order"
    assert result.data.context.tie_handling == "not_ranked"
    assert result.data.context.same_primary_industry is True
    assert result.data.context.peer_comparison_scope == "same_industry_snapshot"
    assert result.data.context.benchmark_catalog_version == (
        "benchmark-catalog-2026-08-01-v1"
    )
    assert result.data.context.benchmark_as_of == date(2026, 8, 1)
    assert result.data.disclaimer == COMPANY_COMPARISON_DISCLAIMER
    assert result.meta.generated_at == GENERATED_AT
    assert result.meta.requested_tax_ids == request.tax_ids
    assert result.meta.requested_count == 3
    assert result.meta.returned_count == 3
    assert result.meta.has_partial_source_data is False
    assert result.meta.has_provisional_scores is False
    assert result.meta.has_unscored_companies is False
    assert result.meta.warnings == []


def test_service_calculates_company_age_at_benchmark_as_of_date() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    responses = [
        _company_bizscore_response(TAX_ID_A),
        _company_bizscore_response(TAX_ID_B),
    ]

    result = build_company_comparison(
        request,
        responses,
        generated_at=GENERATED_AT,
    )

    assert responses[0].data.company.company_age_years == 47.1
    assert result.data.context.benchmark_as_of == date(2026, 8, 1)
    assert result.data.items[0].metrics.company_age_years == 47.0


def test_comparison_item_rejects_metrics_that_disagree_with_nested_data() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    result = build_company_comparison(
        request,
        [
            _company_bizscore_response(TAX_ID_A),
            _company_bizscore_response(TAX_ID_B),
        ],
        generated_at=GENERATED_AT,
    )
    payload = result.data.items[0].model_dump()
    payload["metrics"]["registered_capital"] = 1

    with pytest.raises(
        ValidationError,
        match="comparison metrics do not match nested source data",
    ):
        CompanyComparisonItem.model_validate(payload)


def test_service_rejects_inconsistent_peer_availability() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    malformed = _company_bizscore_response(TAX_ID_B).model_copy(deep=True)
    malformed.data.bizscore.peer_benchmark.peer_index = None

    with pytest.raises(
        CompanyComparisonConsistencyError,
        match="inconsistent with the comparison contract",
    ):
        build_company_comparison(
            request,
            [_company_bizscore_response(TAX_ID_A), malformed],
            generated_at=GENERATED_AT,
        )


@pytest.mark.parametrize("mismatch", ["mapping", "snapshot-provenance"])
def test_service_rejects_mixed_mapping_or_snapshot_provenance(
    mismatch: str,
) -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    second = _company_bizscore_response(TAX_ID_B).model_copy(deep=True)
    if mismatch == "mapping":
        second.data.bizscore.benchmark.catalog_industry_mapping_version = (
            "different-mapping-v1"
        )
        expected_message = "one mapping contract"
    else:
        second.data.bizscore.benchmark.snapshot_checksum_sha256 = "different"
        expected_message = "snapshot metadata"

    with pytest.raises(
        CompanyComparisonConsistencyError,
        match=expected_message,
    ):
        build_company_comparison(
            request,
            [_company_bizscore_response(TAX_ID_A), second],
            generated_at=GENERATED_AT,
        )


def test_service_requires_timezone_aware_generation_timestamp() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])

    with pytest.raises(
        CompanyComparisonConsistencyError,
        match="aggregated comparison response",
    ):
        build_company_comparison(
            request,
            [
                _company_bizscore_response(TAX_ID_A),
                _company_bizscore_response(TAX_ID_B),
            ],
            generated_at=datetime(2026, 8, 24, 9, 30),
        )


def test_service_preserves_partial_provisional_and_unscored_states() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B, TAX_ID_C])
    responses = [
        _company_bizscore_response(TAX_ID_A),
        _company_bizscore_response(
            TAX_ID_B,
            partial=True,
            provisional=True,
        ),
        _company_bizscore_response(TAX_ID_C, unscored=True),
    ]

    result = build_company_comparison(
        request,
        responses,
        generated_at=GENERATED_AT,
    )

    assert [item.metrics.bizscore_status for item in result.data.items] == [
        "complete",
        "provisional",
        "unavailable",
    ]
    assert result.data.items[2].metrics.bizscore is None
    assert "bizscore" in result.data.items[2].metrics.missing
    assert result.meta.has_partial_source_data is True
    assert result.meta.has_provisional_scores is True
    assert result.meta.has_unscored_companies is True
    assert [(warning.code, warning.tax_id) for warning in result.meta.warnings] == [
        ("PARTIAL_SOURCE_DATA", TAX_ID_B),
        ("SOURCE_WARNING", TAX_ID_B),
        ("PROVISIONAL_SCORE", TAX_ID_B),
        ("NO_NUMERIC_SCORE", TAX_ID_C),
    ]


def test_service_exposes_stale_source_snapshot_warning() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    result = build_company_comparison(
        request,
        [
            _company_bizscore_response(TAX_ID_A, stale=True),
            _company_bizscore_response(TAX_ID_B),
        ],
        generated_at=GENERATED_AT,
    )

    assert result.data.items[0].source_meta.data_freshness == "stale_cache"
    assert result.data.items[0].source_meta.fetched_at.isoformat() == (
        "2026-08-24T17:30:00+08:00"
    )
    assert [(warning.code, warning.tax_id) for warning in result.meta.warnings] == [
        ("STALE_SOURCE_DATA", TAX_ID_A)
    ]


def test_service_marks_different_industry_snapshots_without_ranking() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    responses = [
        _company_bizscore_response(TAX_ID_A, industry_code="F"),
        _company_bizscore_response(TAX_ID_B, industry_code="C"),
    ]

    result = build_company_comparison(
        request,
        responses,
        generated_at=GENERATED_AT,
    )

    assert result.data.context.same_primary_industry is False
    assert result.data.context.peer_comparison_scope == (
        "different_industry_snapshots"
    )
    assert result.data.context.tie_handling == "not_ranked"
    assert [warning.code for warning in result.meta.warnings] == [
        "DIFFERENT_INDUSTRY_BENCHMARKS"
    ]


def test_service_rejects_response_count_or_company_mismatch() -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])

    with pytest.raises(
        CompanyComparisonConsistencyError,
        match="requested company count",
    ):
        build_company_comparison(
            request,
            [_company_bizscore_response(TAX_ID_A)],
            generated_at=GENERATED_AT,
        )

    with pytest.raises(
        CompanyComparisonConsistencyError,
        match="does not match the requested tax ID",
    ):
        build_company_comparison(
            request,
            [
                _company_bizscore_response(TAX_ID_C),
                _company_bizscore_response(TAX_ID_B),
            ],
            generated_at=GENERATED_AT,
        )


@pytest.mark.parametrize("mismatch", ["catalog", "as_of"])
def test_service_rejects_mixed_comparison_basis(mismatch: str) -> None:
    request = CompanyComparisonRequest(tax_ids=[TAX_ID_A, TAX_ID_B])
    second_kwargs: dict[str, str] = {}
    if mismatch == "catalog":
        second_kwargs["catalog_version"] = "benchmark-catalog-other-v1"
    else:
        second_kwargs["as_of"] = "2026-08-02"

    with pytest.raises(
        CompanyComparisonConsistencyError,
        match="one Benchmark catalog and as-of date",
    ):
        build_company_comparison(
            request,
            [
                _company_bizscore_response(TAX_ID_A),
                _company_bizscore_response(TAX_ID_B, **second_kwargs),
            ],
            generated_at=GENERATED_AT,
        )


def _gcis_records(
    tax_id: str,
    name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    basic = copy.deepcopy(
        _read_json(GCIS_SAMPLE_DIRECTORY / "company-basic-20828393.json")
    )
    business = copy.deepcopy(
        _read_json(GCIS_SAMPLE_DIRECTORY / "company-business-20828393.json")
    )
    assert isinstance(basic, list) and isinstance(basic[0], dict)
    assert isinstance(business, list) and isinstance(business[0], dict)
    basic[0]["Business_Accounting_NO"] = tax_id
    basic[0]["Company_Name"] = name
    business[0]["Business_Accounting_NO"] = tax_id
    business[0]["Company_Name"] = name
    return basic, business


class MultiCompanyFakeGCISClient:
    def __init__(self) -> None:
        self.basic_outcomes: dict[str, object] = {}
        self.business_outcomes: dict[str, object] = {}
        for tax_id, name in COMPANY_NAMES.items():
            basic, business = _gcis_records(tax_id, name)
            self.basic_outcomes[tax_id] = basic
            self.business_outcomes[tax_id] = business
        self.basic_calls: list[str] = []
        self.business_calls: list[str] = []

    async def get_company_basic(self, tax_id: str) -> list[dict[str, object]]:
        self.basic_calls.append(tax_id)
        return self._resolve(self.basic_outcomes.get(tax_id))

    async def get_company_business_items(
        self,
        tax_id: str,
    ) -> list[dict[str, object]]:
        self.business_calls.append(tax_id)
        return self._resolve(self.business_outcomes.get(tax_id))

    @staticmethod
    def _resolve(outcome: object) -> list[dict[str, object]]:
        if outcome is None:
            raise GCISNotFoundError("not found")
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, list)
        return copy.deepcopy(outcome)


class FakeComparisonBenchmarkService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, tuple[str, ...]]] = []
        self.catalog_versions: dict[str, str] = {}

    def calculate_company_bizscore(
        self,
        company: CompanyData,
        *,
        input_partial: bool = False,
        input_warnings: list[str] | None = None,
    ) -> BizScoreResult:
        warnings = tuple(input_warnings or ())
        self.calls.append((company.tax_id, input_partial, warnings))
        payload = copy.deepcopy(_llm_input_payload()["bizscore"])
        assert isinstance(payload, dict)
        benchmark = payload["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark["catalog_version"] = self.catalog_versions.get(
            company.tax_id,
            "benchmark-catalog-2026-08-01-v1",
        )

        if input_partial:
            payload["score"] = None
            payload["band"] = None
            payload["coverage"] = 0.55
            payload["provisional"] = False
            payload["missing_dimensions"] = [
                "registration_status",
                "peer_relative_position",
            ]
            dimensions = payload["dimensions"]
            assert isinstance(dimensions, list)
            for index in (0, 4):
                assert isinstance(dimensions[index], dict)
                dimensions[index]["score"] = None
                dimensions[index]["available"] = False
            benchmark["industry_code"] = None
            benchmark["snapshot_version"] = None
            benchmark["snapshot_source_version"] = None
            benchmark["snapshot_checksum_sha256"] = None
            benchmark["sample_count"] = 0
            industry = payload["industry"]
            assert isinstance(industry, dict)
            industry["available"] = False
            industry["primary_group"] = None
            industry["groups"] = []
            peer = payload["peer_benchmark"]
            assert isinstance(peer, dict)
            peer["industry_code"] = None
            peer["benchmark_version"] = None
            peer["sample_size"] = 0
            peer["age_percentile"] = None
            peer["capital_percentile"] = None
            peer["peer_index"] = None
            peer_dimension = peer["dimension"]
            assert isinstance(peer_dimension, dict)
            peer_dimension["score"] = None
            peer_dimension["available"] = False

        return BizScoreResult.model_validate(payload)


@pytest.fixture
def comparison_gcis_client(app: FastAPI) -> MultiCompanyFakeGCISClient:
    fake = MultiCompanyFakeGCISClient()
    app.dependency_overrides[get_gcis_client] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_gcis_client, None)


@pytest.fixture
def comparison_benchmark_service(
    app: FastAPI,
) -> FakeComparisonBenchmarkService:
    fake = FakeComparisonBenchmarkService()
    app.dependency_overrides[get_benchmark_catalog_service] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_benchmark_catalog_service, None)


@pytest.fixture
async def comparison_client(
    app: FastAPI,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> AsyncClient:
    del comparison_gcis_client, comparison_benchmark_service
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as test_client:
            yield test_client


@pytest.mark.parametrize(
    "tax_ids",
    [
        pytest.param([TAX_ID_A, TAX_ID_B], id="two-companies"),
        pytest.param([TAX_ID_C, TAX_ID_A, TAX_ID_B], id="three-companies"),
    ],
)
async def test_compare_route_returns_two_or_three_companies_in_request_order(
    tax_ids: list[str],
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    response = await comparison_client.post(
        "/api/v1/companies/compare",
        json={"tax_ids": tax_ids},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["input_index"] for item in payload["data"]["items"]] == list(
        range(len(tax_ids))
    )
    assert [
        item["company"]["tax_id"] for item in payload["data"]["items"]
    ] == tax_ids
    assert payload["meta"]["requested_tax_ids"] == tax_ids
    assert payload["meta"]["requested_count"] == len(tax_ids)
    assert payload["meta"]["returned_count"] == len(tax_ids)
    assert payload["data"]["context"]["ordering"] == "request_order"
    assert payload["data"]["context"]["tie_handling"] == "not_ranked"
    assert payload["data"]["disclaimer"] == COMPANY_COMPARISON_DISCLAIMER
    assert Counter(comparison_gcis_client.basic_calls) == Counter(tax_ids)
    assert Counter(comparison_gcis_client.business_calls) == Counter(tax_ids)
    assert Counter(call[0] for call in comparison_benchmark_service.calls) == Counter(
        tax_ids
    )


async def test_compare_route_preserves_a3_partial_result_and_nulls(
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    comparison_gcis_client.business_outcomes[TAX_ID_B] = GCISTimeoutError(
        "timed out"
    )

    response = await comparison_client.post(
        "/api/v1/companies/compare",
        json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
    )

    assert response.status_code == 200
    payload = response.json()
    partial_item = payload["data"]["items"][1]
    assert partial_item["company"]["tax_id"] == TAX_ID_B
    assert partial_item["company"]["business_items"] == []
    assert partial_item["company"]["status"] == {
        "code": None,
        "description": "核准設立",
    }
    assert partial_item["source_meta"]["partial"] is True
    assert partial_item["metrics"]["registration_status"] == "核准設立"
    assert partial_item["metrics"]["bizscore"] is None
    assert partial_item["metrics"]["peer_index"] is None
    assert set(partial_item["metrics"]["missing"]) >= {
        "bizscore",
        "peer_index",
    }
    assert payload["meta"]["has_partial_source_data"] is True
    assert payload["meta"]["has_unscored_companies"] is True
    warning_codes = [warning["code"] for warning in payload["meta"]["warnings"]]
    assert "PARTIAL_SOURCE_DATA" in warning_codes
    assert "SOURCE_WARNING" in warning_codes
    assert "NO_NUMERIC_SCORE" in warning_codes
    assert "PEER_COMPARISON_UNAVAILABLE" in warning_codes
    partial_calls = [
        call for call in comparison_benchmark_service.calls if call[0] == TAX_ID_B
    ]
    assert len(partial_calls) == 1
    assert partial_calls[0][1] is True
    assert any("timed out" in warning for warning in partial_calls[0][2])


@pytest.mark.parametrize(
    ("upstream_error", "status_code", "error_detail"),
    [
        pytest.param(
            GCISTimeoutError("private upstream timeout detail"),
            504,
            {
                "code": "UPSTREAM_TIMEOUT",
                "message": "政府資料服務回應逾時，請稍後再試。",
                "retryable": True,
            },
            id="timeout",
        ),
        pytest.param(
            GCISNotFoundError("private not-found detail"),
            404,
            {
                "code": "COMPANY_NOT_FOUND",
                "message": "找不到此統編的公司登記資料。",
                "retryable": False,
            },
            id="not-found",
        ),
    ],
)
async def test_compare_route_is_atomic_when_one_company_fails(
    upstream_error: BaseException,
    status_code: int,
    error_detail: dict[str, object],
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
) -> None:
    comparison_gcis_client.basic_outcomes[TAX_ID_B] = upstream_error

    response = await comparison_client.post(
        "/api/v1/companies/compare",
        json={"tax_ids": [TAX_ID_A, TAX_ID_B, TAX_ID_C]},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": error_detail}
    assert "data" not in response.json()
    assert "private" not in response.text


async def test_compare_route_is_atomic_when_benchmark_is_unavailable(
    app: FastAPI,
    comparison_client: AsyncClient,
) -> None:
    class UnavailableBenchmarkService:
        def calculate_company_bizscore(self, *args, **kwargs):
            raise BenchmarkCatalogUnavailableError("private missing snapshot")

    app.dependency_overrides[get_benchmark_catalog_service] = (
        lambda: UnavailableBenchmarkService()
    )

    response = await comparison_client.post(
        "/api/v1/companies/compare",
        json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "BENCHMARK_UNAVAILABLE",
            "message": "目前無法讀取 BizScore 同業基準資料，請稍後再試。",
            "retryable": True,
        }
    }
    assert "data" not in response.json()
    assert "private missing snapshot" not in response.text


async def test_compare_route_maps_aggregation_mismatch_to_public_500(
    comparison_client: AsyncClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    comparison_benchmark_service.catalog_versions[TAX_ID_B] = (
        "benchmark-catalog-other-v1"
    )

    response = await comparison_client.post(
        "/api/v1/companies/compare",
        json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "COMPARISON_CONFIGURATION_ERROR",
            "message": "公司比較資料設定不一致，暫時無法產生結果。",
            "retryable": False,
        }
    }
    assert "benchmark-catalog-other-v1" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"tax_ids": [TAX_ID_A]}, id="too-few"),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_B, TAX_ID_C, "04595257"]},
            id="too-many",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_A]},
            id="duplicate",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, "1234567A"]},
            id="invalid-tax-id",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, "１２３４５６７８"]},
            id="non-ascii-tax-id",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, 22099131]},
            id="non-string-tax-id",
        ),
        pytest.param({}, id="missing-tax-ids"),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_B], "unexpected": "field"},
            id="unknown-field",
        ),
    ],
)
async def test_compare_route_uses_custom_422_without_upstream_calls(
    body: dict[str, object],
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    response = await comparison_client.post(
        "/api/v1/companies/compare",
        json=body,
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "INVALID_COMPARISON_REQUEST",
            "message": "比較請求需提供 2 至 3 個不重複的 8 碼統一編號。",
            "retryable": False,
        }
    }
    assert comparison_gcis_client.basic_calls == []
    assert comparison_gcis_client.business_calls == []
    assert comparison_benchmark_service.calls == []


async def test_compare_route_maps_malformed_json_to_custom_422(
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    response = await comparison_client.post(
        "/api/v1/companies/compare",
        content=b"{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_COMPARISON_REQUEST"
    assert comparison_gcis_client.basic_calls == []
    assert comparison_gcis_client.business_calls == []
    assert comparison_benchmark_service.calls == []


async def test_compare_route_allows_configured_frontend_post_preflight(
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    response = await comparison_client.options(
        "/api/v1/companies/compare",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )
    assert "POST" in response.headers["access-control-allow-methods"]
    assert comparison_gcis_client.basic_calls == []
    assert comparison_gcis_client.business_calls == []
    assert comparison_benchmark_service.calls == []


async def test_compare_route_openapi_contract(
    comparison_client: AsyncClient,
) -> None:
    paths = (await comparison_client.get("/openapi.json")).json()["paths"]
    operation = paths["/api/v1/companies/compare"]["post"]

    request_schema = operation["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert request_schema["$ref"].endswith("/CompanyComparisonRequest")
    components = (await comparison_client.get("/openapi.json")).json()[
        "components"
    ]["schemas"]
    request_model = components["CompanyComparisonRequest"]
    tax_ids_schema = request_model["properties"]["tax_ids"]
    assert request_model["additionalProperties"] is False
    assert tax_ids_schema["minItems"] == 2
    assert tax_ids_schema["maxItems"] == 3
    assert tax_ids_schema["uniqueItems"] is True
    assert tax_ids_schema["items"]["pattern"] == "^[0-9]{8}$"
    assert operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/CompanyComparisonResponse")
    for status_code in ("404", "422", "500", "502", "503", "504"):
        assert operation["responses"][status_code]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/ErrorResponse")


async def test_comparison_analysis_route_returns_explicit_disabled_fallback(
    app: FastAPI,
    comparison_client: AsyncClient,
) -> None:
    app.dependency_overrides[get_company_analysis_provider] = lambda: (
        DisabledCompanyAnalysisProvider("configured-model")
    )
    try:
        response = await comparison_client.post(
            "/api/v1/companies/compare/analysis",
            json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
        )
    finally:
        app.dependency_overrides.pop(get_company_analysis_provider, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["status"] == "fallback"
    assert payload["data"]["analysis"] is None
    assert payload["data"]["fallback"] == {
        "code": "not_configured",
        "title": "AI 比較重點尚未啟用",
        "message": (
            "已驗證的公司比較資料仍可使用；請直接查看各欄位，並另行核對交易條件與履約文件。"
        ),
        "retryable": False,
    }
    assert [
        item["company"]["tax_id"]
        for item in payload["data"]["comparison"]["data"]["items"]
    ] == [TAX_ID_A, TAX_ID_B]
    assert payload["meta"]["provider"] == "disabled"
    assert payload["meta"]["model"] == "configured-model"
    assert payload["meta"]["output_sha256"] is None
    assert payload["meta"]["cache"]["status"] == "miss"
    assert payload["meta"]["fallback"] == {
        "active": True,
        "code": "not_configured",
        "retryable": False,
    }


class _SuccessfulComparisonProvider:
    provider_name = "fake-openai"
    model_name = "fake-comparison-model"

    def __init__(self) -> None:
        self.calls = 0

    def ensure_available(self) -> None:
        return None

    async def generate(self, prompt):
        del prompt
        self.calls += 1
        tax_ids = [TAX_ID_A, TAX_ID_B]
        company_paths = [
            f"/comparison/data/items/{index}/company/status/description"
            for index in range(2)
        ]
        peer_path = "/comparison/data/context/peer_comparison_scope"
        return LLMProviderResult(
            output={
                "schema_version": "1.0",
                "status": "completed",
                "headline": "公開登記資料並列觀察",
                "overall_observation": (
                    "請並列查看公開欄位，並針對合作條件另行查核。"
                ),
                "company_observations": [
                    {
                        "tax_id": tax_id,
                        "topic": "registration_status",
                        "title": "登記狀態欄位",
                        "observation": "公開登記狀態已有欄位可供查看。",
                        "evidence_paths": [company_paths[index]],
                        "caveat": (
                            "登記狀態不能代表付款、履約或實際營運狀況。"
                        ),
                    }
                    for index, tax_id in enumerate(tax_ids)
                ],
                "comparison_observations": [
                    {
                        "topic": "peer_scope",
                        "title": "同業比較範圍",
                        "observation": (
                            "同業相對位置應依輸入所列基準範圍分別閱讀。"
                        ),
                        "evidence_paths": [peer_path],
                        "caveat": "同業相對位置不是信用或公司優劣結論。",
                    }
                ],
                "verification_items": [
                    {
                        "priority": "優先",
                        "question": "是否已向各公司核對交易條件與履約文件？",
                        "reason": "公開登記資料不包含個別交易安排。",
                        "related_evidence_paths": [],
                    }
                ],
                "limitations": [
                    {
                        "code": "public_data_only",
                        "message": "內容僅依政府公開登記資料整理。",
                        "related_evidence_paths": [],
                    },
                    {
                        "code": "ai_generated",
                        "message": "文字由人工智慧依已驗證輸入整理。",
                        "related_evidence_paths": [],
                    },
                    {
                        "code": "not_ranked",
                        "message": "公司順序只沿用使用者選取順序，不代表名次。",
                        "related_evidence_paths": [],
                    },
                ],
                "provenance": {
                    "input_schema_version": "1.0",
                    "prompt_version": "1.0",
                    "requested_tax_ids": tax_ids,
                    "comparison_version": "1.0",
                    "bizscore_version": "1.0",
                    "benchmark_catalog_version": (
                        "benchmark-catalog-2026-08-01-v1"
                    ),
                    "data_as_of": "2026-08-01",
                    "disclaimer_version": "1.0",
                    "evidence_paths_used": [*company_paths, peer_path],
                },
                "disclaimer": AI_COMPARISON_DISCLAIMER,
            },
            provider=self.provider_name,
            model=self.model_name,
            response_id="resp-comparison-route",
            created_at=datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc),
        )

    async def aclose(self) -> None:
        return None


async def test_comparison_analysis_route_generated_then_cache_hit(
    app: FastAPI,
    comparison_client: AsyncClient,
) -> None:
    provider = _SuccessfulComparisonProvider()
    app.dependency_overrides[get_company_analysis_provider] = lambda: provider
    try:
        first = await comparison_client.post(
            "/api/v1/companies/compare/analysis",
            json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
        )
        second = await comparison_client.post(
            "/api/v1/companies/compare/analysis",
            json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
        )
    finally:
        app.dependency_overrides.pop(get_company_analysis_provider, None)

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["data"]["status"] == "completed"
    assert first_payload["data"]["analysis"]["disclaimer"] == (
        AI_COMPARISON_DISCLAIMER
    )
    assert first_payload["data"]["fallback"] is None
    assert first_payload["meta"]["cache"]["status"] == "miss"
    assert second_payload["meta"]["cache"]["status"] == "hit"
    assert first_payload["meta"]["cache"]["key_sha256"] == (
        second_payload["meta"]["cache"]["key_sha256"]
    )
    assert provider.calls == 1


async def test_comparison_analysis_force_refresh_is_explicit_bypass_on_fallback(
    app: FastAPI,
    comparison_client: AsyncClient,
) -> None:
    app.dependency_overrides[get_company_analysis_provider] = lambda: (
        DisabledCompanyAnalysisProvider("configured-model")
    )
    try:
        response = await comparison_client.post(
            "/api/v1/companies/compare/analysis",
            json={
                "tax_ids": [TAX_ID_A, TAX_ID_B],
                "force_refresh": True,
            },
        )
    finally:
        app.dependency_overrides.pop(get_company_analysis_provider, None)

    assert response.status_code == 200
    assert response.json()["meta"]["cache"]["status"] == "bypass"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"tax_ids": [TAX_ID_A]}, id="too-few"),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_A]},
            id="duplicate",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_B], "force_refresh": "true"},
            id="non-boolean-refresh",
        ),
        pytest.param(
            {"tax_ids": [TAX_ID_A, TAX_ID_B], "unexpected": True},
            id="unknown-field",
        ),
    ],
)
async def test_comparison_analysis_custom_422_has_no_upstream_calls(
    body: dict[str, object],
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
    comparison_benchmark_service: FakeComparisonBenchmarkService,
) -> None:
    response = await comparison_client.post(
        "/api/v1/companies/compare/analysis",
        json=body,
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "INVALID_COMPARISON_ANALYSIS_REQUEST",
            "message": (
                "請輸入 2 或 3 個不重複的 8 碼統一編號，force_refresh 必須是布林值。"
            ),
            "retryable": False,
        }
    }
    assert comparison_gcis_client.basic_calls == []
    assert comparison_gcis_client.business_calls == []
    assert comparison_benchmark_service.calls == []


async def test_comparison_analysis_malformed_json_is_not_tax_id_route_collision(
    comparison_client: AsyncClient,
) -> None:
    response = await comparison_client.post(
        "/api/v1/companies/compare/analysis",
        content=b"{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == (
        "INVALID_COMPARISON_ANALYSIS_REQUEST"
    )


class _CountingProvider:
    provider_name = "counting"
    model_name = "counting-model"

    def __init__(self) -> None:
        self.ensure_calls = 0
        self.generate_calls = 0

    def ensure_available(self) -> None:
        self.ensure_calls += 1

    async def generate(self, prompt):
        del prompt
        self.generate_calls += 1
        raise AssertionError("Provider must not be called before comparison succeeds.")

    async def aclose(self) -> None:
        return None


async def test_comparison_analysis_gcis_failure_makes_zero_model_calls(
    app: FastAPI,
    comparison_client: AsyncClient,
    comparison_gcis_client: MultiCompanyFakeGCISClient,
) -> None:
    provider = _CountingProvider()
    app.dependency_overrides[get_company_analysis_provider] = lambda: provider
    comparison_gcis_client.basic_outcomes[TAX_ID_B] = GCISTimeoutError(
        "private timeout"
    )
    try:
        response = await comparison_client.post(
            "/api/v1/companies/compare/analysis",
            json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
        )
    finally:
        app.dependency_overrides.pop(get_company_analysis_provider, None)

    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "UPSTREAM_TIMEOUT"
    assert provider.ensure_calls == 0
    assert provider.generate_calls == 0


async def test_comparison_analysis_benchmark_failure_makes_zero_model_calls(
    app: FastAPI,
    comparison_client: AsyncClient,
) -> None:
    class UnavailableBenchmarkService:
        def calculate_company_bizscore(self, *args, **kwargs):
            raise BenchmarkCatalogUnavailableError("private benchmark error")

    provider = _CountingProvider()
    app.dependency_overrides[get_company_analysis_provider] = lambda: provider
    app.dependency_overrides[get_benchmark_catalog_service] = (
        lambda: UnavailableBenchmarkService()
    )
    try:
        response = await comparison_client.post(
            "/api/v1/companies/compare/analysis",
            json={"tax_ids": [TAX_ID_A, TAX_ID_B]},
        )
    finally:
        app.dependency_overrides.pop(get_company_analysis_provider, None)
        app.dependency_overrides.pop(get_benchmark_catalog_service, None)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "BENCHMARK_UNAVAILABLE"
    assert provider.ensure_calls == 0
    assert provider.generate_calls == 0


async def test_comparison_analysis_route_cors_preflight(
    comparison_client: AsyncClient,
) -> None:
    response = await comparison_client.options(
        "/api/v1/companies/compare/analysis",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )
    assert "POST" in response.headers["access-control-allow-methods"]


async def test_comparison_analysis_route_openapi_contract(
    comparison_client: AsyncClient,
) -> None:
    document = (await comparison_client.get("/openapi.json")).json()
    operation = document["paths"][
        "/api/v1/companies/compare/analysis"
    ]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    response_ref = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    assert request_ref.endswith("/CompanyComparisonAnalysisRequest")
    assert response_ref.endswith("/CompanyComparisonAnalysisResponse")

    components = document["components"]["schemas"]
    request_model = components["CompanyComparisonAnalysisRequest"]
    assert request_model["additionalProperties"] is False
    assert request_model["properties"]["tax_ids"]["minItems"] == 2
    assert request_model["properties"]["tax_ids"]["maxItems"] == 3
    assert request_model["properties"]["tax_ids"]["uniqueItems"] is True
    assert request_model["properties"]["force_refresh"]["default"] is False
    for status_code in ("404", "422", "500", "502", "503", "504"):
        assert operation["responses"][status_code]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/ErrorResponse")

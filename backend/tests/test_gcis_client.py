import json
from pathlib import Path

import httpx
import pytest

from app.services.gcis import (
    BASIC_DATASET_ID,
    BUSINESS_DATASET_ID,
    KEYWORD_DATASET_ID,
    MAX_CONFIGURED_RETRIES,
    GCISClient,
    GCISClientConfig,
    GCISConnectionError,
    GCISErrorCategory,
    GCISInvalidResponseError,
    GCISNotFoundError,
    GCISTimeoutError,
    GCISUpstreamHTTPError,
    GCISValidationError,
)


pytestmark = pytest.mark.anyio

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = WORKSPACE_ROOT / "samples" / "gcis"
BASE_URL = "https://data.gcis.nat.gov.tw/od/data/api"


def load_sample(filename: str) -> object:
    with (SAMPLE_DIR / filename).open(encoding="utf-8") as sample_file:
        return json.load(sample_file)


def config(**overrides: object) -> GCISClientConfig:
    values: dict[str, object] = {
        "base_url": BASE_URL,
        "timeout_seconds": 1.0,
        "max_retries": 0,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return GCISClientConfig(**values)  # type: ignore[arg-type]


async def test_get_company_basic_uses_expected_dataset_and_filter() -> None:
    sample = load_sample("company-basic-20828393.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(BASIC_DATASET_ID)
        assert request.url.params["$format"] == "json"
        assert request.url.params["$filter"] == (
            "Business_Accounting_NO eq 20828393"
        )
        assert request.url.params["$skip"] == "0"
        assert request.url.params["$top"] == "50"
        return httpx.Response(200, json=sample)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        records = await client.get_company_basic("20828393")

    assert len(records) == 1
    assert records[0]["Company_Name"] == "宏碁股份有限公司"


async def test_get_company_business_items_reads_recorded_sample() -> None:
    sample = load_sample("company-business-20828393.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(BUSINESS_DATASET_ID)
        return httpx.Response(200, json=sample)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        records = await client.get_company_business_items("20828393")

    assert len(records[0]["Cmp_Business"]) == 22
    assert records[0]["Cmp_Business"][0]["Business_Seq_NO"] == "0001"


async def test_search_companies_builds_keyword_filter() -> None:
    response_payload = [
        {
            "Business_Accounting_NO": "20828393",
            "Company_Name": "宏碁股份有限公司",
            "Company_Status": "01",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(KEYWORD_DATASET_ID)
        assert request.url.params["$filter"] == (
            "Company_Name like 宏碁 and Company_Status eq 01"
        )
        assert request.url.params["$top"] == "10"
        return httpx.Response(200, json=response_payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        records = await client.search_companies("  宏碁  ")

    assert records == response_payload


async def test_direct_lookup_treats_empty_body_as_not_found() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"",
            headers={"Content-Type": "application/json; charset=UTF-8"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        with pytest.raises(GCISNotFoundError):
            await client.get_company_basic("00000000")


async def test_keyword_search_treats_empty_body_as_empty_results() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"",
            headers={"Content-Type": "application/json; charset=UTF-8"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        records = await client.search_companies("查無資料")

    assert records == []


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"not-json", "application/json"),
        (b'{"unexpected": "object"}', "application/json"),
        (b"[]", "text/plain"),
    ],
)
async def test_invalid_response_shapes_are_rejected(
    content: bytes,
    content_type: str,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"Content-Type": content_type},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        with pytest.raises(GCISInvalidResponseError):
            await client.search_companies("宏碁")


async def test_retryable_upstream_status_is_retried_once() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(
            config(max_retries=1),
            http_client=http_client,
        )
        records = await client.search_companies("宏碁")

    assert records == []
    assert attempts == 2


async def test_non_retryable_upstream_status_preserves_status_code() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="not authorized")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        with pytest.raises(GCISUpstreamHTTPError) as captured:
            await client.get_company_basic("20828393")

    assert captured.value.status_code == 403


async def test_timeout_is_translated_to_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        with pytest.raises(GCISTimeoutError) as captured:
            await client.get_company_basic("20828393")

    assert captured.value.category == GCISErrorCategory.TIMEOUT
    assert captured.value.retryable is True
    assert captured.value.attempts == 1


async def test_timeout_retries_only_to_configured_limit() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(
            config(max_retries=2),
            http_client=http_client,
        )
        with pytest.raises(GCISTimeoutError) as captured:
            await client.get_company_basic("20828393")

    assert attempts == 3
    assert captured.value.attempts == 3


async def test_connection_error_is_retried_and_classified() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(
            config(max_retries=1),
            http_client=http_client,
        )
        with pytest.raises(GCISConnectionError) as captured:
            await client.search_companies("宏碁")

    assert attempts == 2
    assert captured.value.category == GCISErrorCategory.CONNECTION
    assert captured.value.retryable is True
    assert captured.value.attempts == 2


@pytest.mark.parametrize(
    ("status_code", "category", "retryable"),
    (
        (429, GCISErrorCategory.RATE_LIMITED, True),
        (503, GCISErrorCategory.SERVICE_UNAVAILABLE, True),
        (403, GCISErrorCategory.REQUEST_REJECTED, False),
        (302, GCISErrorCategory.UPSTREAM_HTTP, False),
    ),
)
async def test_upstream_http_errors_are_classified(
    status_code: int,
    category: GCISErrorCategory,
    retryable: bool,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="upstream response")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)
        with pytest.raises(GCISUpstreamHTTPError) as captured:
            await client.search_companies("宏碁")

    assert captured.value.status_code == status_code
    assert captured.value.category == category
    assert captured.value.retryable is retryable
    assert captured.value.attempts == 1


async def test_availability_probe_uses_short_timeout_and_does_not_retry() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.url.path.endswith(BASIC_DATASET_ID)
        assert request.url.params["$filter"] == (
            "Business_Accounting_NO eq 00000000"
        )
        assert request.url.params["$top"] == "1"
        timeout = request.extensions["timeout"]
        assert timeout["connect"] == 0.25
        assert timeout["read"] == 0.25
        raise httpx.ReadTimeout("probe timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(
            config(
                max_retries=MAX_CONFIGURED_RETRIES,
                probe_timeout_seconds=0.25,
            ),
            http_client=http_client,
        )
        with pytest.raises(GCISTimeoutError) as captured:
            await client.check_availability()

    assert attempts == 1
    assert captured.value.attempts == 1


@pytest.mark.parametrize(
    "overrides",
    (
        {"max_retries": MAX_CONFIGURED_RETRIES + 1},
        {"retry_backoff_seconds": 10.01},
        {"probe_timeout_seconds": 0},
    ),
)
async def test_retry_configuration_is_bounded(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        config(**overrides)


async def test_exponential_retry_delay_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("app.services.gcis.asyncio.sleep", record_sleep)
    async with httpx.AsyncClient() as http_client:
        client = GCISClient(
            config(retry_backoff_seconds=10),
            http_client=http_client,
        )
        await client._sleep_before_retry(2)

    assert delays == [10.0]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("tax_id", "tax_id must contain exactly 8 digits"),
        ("status", "company_status must contain exactly 2 digits"),
        ("keyword", "keyword contains characters"),
        ("limit", "limit must be between 1 and 1000"),
    ],
)
async def test_query_values_are_validated(operation: str, message: str) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Invalid input must not send an HTTP request.")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = GCISClient(config(), http_client=http_client)

        with pytest.raises(GCISValidationError, match=message):
            if operation == "tax_id":
                await client.get_company_basic("123")
            elif operation == "status":
                await client.search_companies("宏碁", company_status="1")
            elif operation == "keyword":
                await client.search_companies("宏碁&status=99")
            else:
                await client.search_companies("宏碁", limit=0)

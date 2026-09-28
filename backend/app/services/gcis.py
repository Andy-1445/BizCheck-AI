from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx


BASIC_DATASET_ID = "5F64D864-61CB-4D0D-8AD9-492047CC1EA6"
BUSINESS_DATASET_ID = "236EE382-4942-41A9-BD03-CA0709025E7C"
KEYWORD_DATASET_ID = "6BBA2268-1367-4B42-9CCA-BC17499EBE8C"

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_CONFIGURED_RETRIES = 3
MAX_RETRY_BACKOFF_SECONDS = 10.0
TAX_ID_PATTERN = re.compile(r"^[0-9]{8}$")
COMPANY_STATUS_PATTERN = re.compile(r"^[0-9]{2}$")
SAFE_KEYWORD_PATTERN = re.compile(r"^[\w\s.\-．－·・]+$", re.UNICODE)

JSONRecord = dict[str, Any]


class GCISErrorCategory(str, Enum):
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMITED = "rate_limited"
    SERVICE_UNAVAILABLE = "service_unavailable"
    REQUEST_REJECTED = "request_rejected"
    INVALID_RESPONSE = "invalid_response"
    UPSTREAM_HTTP = "upstream_http"
    UNKNOWN = "unknown"


class GCISError(RuntimeError):
    """Base class for classified errors raised by the GCIS adapter."""

    category = GCISErrorCategory.UNKNOWN
    retryable = False

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        self.attempts = max(1, attempts)
        super().__init__(message)


class GCISValidationError(GCISError):
    """The caller supplied an invalid query value."""

    category = GCISErrorCategory.VALIDATION


class GCISNotFoundError(GCISError):
    """GCIS returned no record for a direct company lookup."""

    category = GCISErrorCategory.NOT_FOUND


class GCISTimeoutError(GCISError):
    """GCIS did not respond before the configured timeout."""

    category = GCISErrorCategory.TIMEOUT
    retryable = True


class GCISConnectionError(GCISError):
    """The request could not reach GCIS."""

    category = GCISErrorCategory.CONNECTION
    retryable = True


class GCISInvalidResponseError(GCISError):
    """GCIS returned a response that did not match the expected JSON shape."""

    category = GCISErrorCategory.INVALID_RESPONSE
    retryable = True


class GCISUpstreamHTTPError(GCISError):
    """GCIS returned a non-success HTTP status."""

    def __init__(self, status_code: int, *, attempts: int = 1) -> None:
        self.status_code = status_code
        if status_code == 429:
            self.category = GCISErrorCategory.RATE_LIMITED
            self.retryable = True
        elif status_code in RETRYABLE_STATUS_CODES:
            self.category = GCISErrorCategory.SERVICE_UNAVAILABLE
            self.retryable = True
        elif 400 <= status_code < 500:
            self.category = GCISErrorCategory.REQUEST_REJECTED
            self.retryable = False
        else:
            self.category = GCISErrorCategory.UPSTREAM_HTTP
            self.retryable = False
        super().__init__(f"GCIS returned HTTP {status_code}.", attempts=attempts)


@dataclass(frozen=True, slots=True)
class GCISClientConfig:
    base_url: str
    timeout_seconds: float = 10.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.2
    probe_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        normalized_base_url = self.base_url.strip().rstrip("/")
        if not normalized_base_url.startswith(("http://", "https://")):
            raise ValueError("GCIS base_url must use http or https.")
        if self.timeout_seconds <= 0:
            raise ValueError("GCIS timeout_seconds must be greater than zero.")
        if self.max_retries < 0:
            raise ValueError("GCIS max_retries cannot be negative.")
        if self.max_retries > MAX_CONFIGURED_RETRIES:
            raise ValueError(
                f"GCIS max_retries cannot exceed {MAX_CONFIGURED_RETRIES}."
            )
        if self.retry_backoff_seconds < 0:
            raise ValueError("GCIS retry_backoff_seconds cannot be negative.")
        if self.retry_backoff_seconds > MAX_RETRY_BACKOFF_SECONDS:
            raise ValueError(
                "GCIS retry_backoff_seconds cannot exceed "
                f"{MAX_RETRY_BACKOFF_SECONDS}."
            )
        if self.probe_timeout_seconds <= 0:
            raise ValueError("GCIS probe_timeout_seconds must be greater than zero.")

        object.__setattr__(self, "base_url", normalized_base_url)


class GCISClient:
    """Asynchronous client for the GCIS company-registration APIs.

    A client can own its internal ``httpx.AsyncClient`` or receive one through
    dependency injection. Injecting a client enables deterministic tests with
    ``httpx.MockTransport`` and will later allow the FastAPI lifespan to share
    one connection pool.
    """

    def __init__(
        self,
        config: GCISClientConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": "BizCheckAI/0.1",
            },
        )

    async def __aenter__(self) -> GCISClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def get_company_basic(self, tax_id: str) -> list[JSONRecord]:
        validated_tax_id = self._validate_tax_id(tax_id)
        return await self._request_records(
            dataset_id=BASIC_DATASET_ID,
            filter_expression=f"Business_Accounting_NO eq {validated_tax_id}",
            limit=50,
            empty_is_not_found=True,
        )

    async def get_company_business_items(
        self,
        tax_id: str,
    ) -> list[JSONRecord]:
        validated_tax_id = self._validate_tax_id(tax_id)
        return await self._request_records(
            dataset_id=BUSINESS_DATASET_ID,
            filter_expression=f"Business_Accounting_NO eq {validated_tax_id}",
            limit=50,
            empty_is_not_found=True,
        )

    async def search_companies(
        self,
        keyword: str,
        company_status: str = "01",
        limit: int = 10,
    ) -> list[JSONRecord]:
        normalized_keyword = self._validate_keyword(keyword)
        validated_status = self._validate_company_status(company_status)
        validated_limit = self._validate_limit(limit)

        return await self._request_records(
            dataset_id=KEYWORD_DATASET_ID,
            filter_expression=(
                f"Company_Name like {normalized_keyword} "
                f"and Company_Status eq {validated_status}"
            ),
            limit=validated_limit,
            empty_is_not_found=False,
        )

    async def check_availability(self) -> None:
        """Run a bounded, non-retrying probe against a GCIS JSON dataset."""

        await self._request_records(
            dataset_id=BASIC_DATASET_ID,
            filter_expression="Business_Accounting_NO eq 00000000",
            limit=1,
            empty_is_not_found=False,
            timeout_seconds=self.config.probe_timeout_seconds,
            max_retries=0,
        )

    async def _request_records(
        self,
        *,
        dataset_id: str,
        filter_expression: str,
        limit: int,
        empty_is_not_found: bool,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> list[JSONRecord]:
        url = f"{self.config.base_url}/{dataset_id}"
        params = {
            "$format": "json",
            "$filter": filter_expression,
            "$skip": "0",
            "$top": str(limit),
        }

        response = await self._get_with_retry(
            url,
            params=params,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        attempts = int(response.extensions.get("bizcheck_attempts", 1))
        body = response.text.strip()

        if not body:
            if empty_is_not_found:
                raise GCISNotFoundError(
                    "GCIS returned an empty response body.",
                    attempts=attempts,
                )
            return []

        content_type = response.headers.get("content-type", "").lower()
        if "json" not in content_type:
            raise GCISInvalidResponseError(
                "GCIS response did not declare a JSON content type.",
                attempts=attempts,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise GCISInvalidResponseError(
                "GCIS response body was not valid JSON.",
                attempts=attempts,
            ) from exc

        if not isinstance(payload, list):
            raise GCISInvalidResponseError(
                "GCIS response root must be a JSON array.",
                attempts=attempts,
            )
        if any(not isinstance(item, dict) for item in payload):
            raise GCISInvalidResponseError(
                "Every GCIS response item must be a JSON object.",
                attempts=attempts,
            )
        if not payload and empty_is_not_found:
            raise GCISNotFoundError(
                "GCIS returned an empty result array.",
                attempts=attempts,
            )

        return payload

    async def _get_with_retry(
        self,
        url: str,
        *,
        params: dict[str, str],
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        request_timeout = (
            self.config.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        final_attempt = (
            self.config.max_retries if max_retries is None else max_retries
        )

        for attempt in range(final_attempt + 1):
            try:
                response = await self._http_client.get(
                    url,
                    params=params,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                if attempt < final_attempt:
                    await self._sleep_before_retry(attempt)
                    continue
                raise GCISTimeoutError(
                    "GCIS request timed out.",
                    attempts=attempt + 1,
                ) from exc
            except httpx.RequestError as exc:
                if attempt < final_attempt:
                    await self._sleep_before_retry(attempt)
                    continue
                raise GCISConnectionError(
                    "Could not connect to GCIS.",
                    attempts=attempt + 1,
                ) from exc

            if 200 <= response.status_code < 300:
                response.extensions["bizcheck_attempts"] = attempt + 1
                return response

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < final_attempt
            ):
                await self._sleep_before_retry(attempt)
                continue

            raise GCISUpstreamHTTPError(
                response.status_code,
                attempts=attempt + 1,
            )

        raise AssertionError("GCIS retry loop exited unexpectedly.")

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(
            self.config.retry_backoff_seconds * (2**attempt),
            MAX_RETRY_BACKOFF_SECONDS,
        )
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _validate_tax_id(tax_id: str) -> str:
        normalized_tax_id = tax_id.strip()
        if not TAX_ID_PATTERN.fullmatch(normalized_tax_id):
            raise GCISValidationError("tax_id must contain exactly 8 digits.")
        return normalized_tax_id

    @staticmethod
    def _validate_company_status(company_status: str) -> str:
        normalized_status = company_status.strip()
        if not COMPANY_STATUS_PATTERN.fullmatch(normalized_status):
            raise GCISValidationError(
                "company_status must contain exactly 2 digits."
            )
        return normalized_status

    @staticmethod
    def _validate_keyword(keyword: str) -> str:
        normalized_keyword = " ".join(keyword.strip().split())
        if not 2 <= len(normalized_keyword) <= 100:
            raise GCISValidationError(
                "keyword length must be between 2 and 100 characters."
            )
        if not SAFE_KEYWORD_PATTERN.fullmatch(normalized_keyword):
            raise GCISValidationError(
                "keyword contains characters that are unsafe for GCIS filters."
            )
        return normalized_keyword

    @staticmethod
    def _validate_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise GCISValidationError("limit must be an integer.")
        if not 1 <= limit <= 1000:
            raise GCISValidationError("limit must be between 1 and 1000.")
        return limit

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from pydantic import ValidationError

from app.api.dependencies import (
    get_benchmark_catalog_service,
    get_company_analysis_provider,
    get_company_comparison_analysis_cache,
    get_gcis_client,
    get_gcis_response_cache,
)
from app.schemas.bizscore import CompanyBizScoreData, CompanyBizScoreResponse
from app.schemas.company_analysis import (
    CompanyAnalysisData,
    CompanyAnalysisMeta,
    CompanyAnalysisResponse,
    LLMTokenUsage,
)
from app.schemas.common import ErrorDetail, ErrorResponse
from app.schemas.company import (
    CompanyResponse,
    CompanySearchItem,
    CompanySearchResponse,
    ResponseMeta,
)
from app.schemas.company_comparison import (
    CompanyComparisonRequest,
    CompanyComparisonResponse,
)
from app.schemas.company_comparison_analysis import (
    CompanyComparisonAnalysisRequest,
    CompanyComparisonAnalysisResponse,
)
from app.schemas.llm_analysis import CompanyAnalysisLLMInput
from app.services.company_analysis import (
    CompanyAnalysisInvalidOutputError,
    CompanyAnalysisService,
)
from app.services.company_comparison import (
    CompanyComparisonConsistencyError,
    build_company_comparison,
)
from app.services.company_comparison_analysis import (
    CompanyComparisonAnalysisExecution,
    CompanyComparisonAnalysisService,
)
from app.services.company_normalizer import (
    CompanyNormalizationError,
    normalize_company_data,
    normalize_company_search_item,
)
from app.services.benchmark_catalog import (
    BenchmarkCatalogInvalidError,
    BenchmarkCatalogService,
    BenchmarkCatalogUnavailableError,
)
from app.services.gcis import (
    GCISError,
    GCISErrorCategory,
    GCISClient,
    GCISConnectionError,
    GCISInvalidResponseError,
    GCISNotFoundError,
    GCISTimeoutError,
    GCISUpstreamHTTPError,
    GCISValidationError,
)
from app.services.gcis_response_cache import GCISResponseCache
from app.services.llm_provider import (
    CompanyAnalysisProvider,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMIncompleteResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.services.llm_cache import AsyncLRUTTLCache


router = APIRouter()
TAIPEI_TIMEZONE = timezone(timedelta(hours=8))
ASCII_TAX_ID_PATTERN = r"^[0-9]{8}$"
ASCII_TAX_ID_RE = re.compile(ASCII_TAX_ID_PATTERN)

COMPANY_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No company registration was found for the tax ID.",
    },
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": "GCIS returned an invalid or failed response.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "GCIS is currently unreachable.",
    },
    status.HTTP_504_GATEWAY_TIMEOUT: {
        "model": ErrorResponse,
        "description": "GCIS did not respond before the timeout.",
    },
}
COMPANY_DETAIL_ERROR_RESPONSES = {
    **COMPANY_ERROR_RESPONSES,
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The tax ID must contain exactly eight digits.",
    },
}
COMPANY_BIZSCORE_ERROR_RESPONSES = {
    **COMPANY_DETAIL_ERROR_RESPONSES,
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "The configured Benchmark catalog is inconsistent.",
    },
}
COMPANY_COMPARISON_ERROR_RESPONSES = {
    **COMPANY_BIZSCORE_ERROR_RESPONSES,
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "The request must contain two or three unique eight-digit tax IDs."
        ),
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": (
            "The Benchmark catalog or aggregated comparison data is inconsistent."
        ),
    },
}
COMPANY_COMPARISON_ANALYSIS_ERROR_RESPONSES = {
    **COMPANY_COMPARISON_ERROR_RESPONSES,
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "The request must contain two or three unique eight-digit tax IDs "
            "and a boolean force_refresh value."
        ),
    },
}
COMPANY_ANALYSIS_ERROR_RESPONSES = {
    **COMPANY_BIZSCORE_ERROR_RESPONSES,
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": "The AI provider returned a failed or unsafe response.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "AI analysis is not configured or is temporarily unavailable.",
    },
    status.HTTP_504_GATEWAY_TIMEOUT: {
        "model": ErrorResponse,
        "description": "The AI provider did not respond before the timeout.",
    },
}


def _api_error(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
) -> HTTPException:
    detail = ErrorDetail(
        code=code,
        message=message,
        retryable=retryable,
    )
    return HTTPException(
        status_code=status_code,
        detail=detail.model_dump(),
    )


def _require_ascii_tax_id(request: Request) -> None:
    """Reject an invalid company path before resolving service dependencies."""

    tax_id = request.path_params.get("tax_id")
    if not isinstance(tax_id, str) or ASCII_TAX_ID_RE.fullmatch(tax_id) is None:
        raise _api_error(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="INVALID_TAX_ID",
            message="請輸入 8 碼統一編號。",
            retryable=False,
        )


def _translate_gcis_error(error: GCISError) -> HTTPException:
    if isinstance(error, GCISNotFoundError):
        return _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            code="COMPANY_NOT_FOUND",
            message="找不到此統編的公司登記資料。",
            retryable=False,
        )
    if isinstance(error, GCISTimeoutError):
        return _api_error(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            code="UPSTREAM_TIMEOUT",
            message="政府資料服務回應逾時，請稍後再試。",
            retryable=True,
        )
    if isinstance(error, GCISConnectionError):
        return _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="UPSTREAM_UNAVAILABLE",
            message="目前無法連線至政府資料服務，請稍後再試。",
            retryable=True,
        )
    if isinstance(error, GCISValidationError):
        return _api_error(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="INVALID_QUERY",
            message=str(error),
            retryable=False,
        )
    if isinstance(error, GCISInvalidResponseError):
        return _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="UPSTREAM_INVALID_RESPONSE",
            message="政府資料服務回應暫時無法解析。",
            retryable=True,
        )
    if isinstance(error, GCISUpstreamHTTPError):
        if error.category == GCISErrorCategory.RATE_LIMITED:
            return _api_error(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="UPSTREAM_RATE_LIMITED",
                message="政府資料服務目前流量過高，請稍後再試。",
                retryable=True,
            )
        if error.category == GCISErrorCategory.SERVICE_UNAVAILABLE:
            return _api_error(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="UPSTREAM_UNAVAILABLE",
                message="政府資料服務暫時無法使用，請稍後再試。",
                retryable=True,
            )
        if error.category == GCISErrorCategory.REQUEST_REJECTED:
            return _api_error(
                status_code=status.HTTP_502_BAD_GATEWAY,
                code="UPSTREAM_REQUEST_REJECTED",
                message="政府資料服務拒絕了目前的查詢格式。",
                retryable=False,
            )
        return _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="UPSTREAM_HTTP_ERROR",
            message=f"政府資料服務回傳 HTTP {error.status_code}。",
            retryable=error.retryable,
        )
    return _api_error(
        status_code=status.HTTP_502_BAD_GATEWAY,
        code="UPSTREAM_ERROR",
        message="政府資料服務發生非預期錯誤。",
        retryable=True,
    )


def _translate_llm_error(error: LLMProviderError) -> HTTPException:
    if isinstance(error, LLMNotConfiguredError):
        return _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AI_ANALYSIS_NOT_CONFIGURED",
            message="AI 企業分析尚未啟用，請先設定分析服務。",
            retryable=False,
        )
    if isinstance(error, LLMConfigurationError):
        return _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="AI_ANALYSIS_CONFIGURATION_ERROR",
            message="AI 企業分析設定不一致，暫時無法產生報告。",
            retryable=False,
        )
    if isinstance(error, LLMTimeoutError):
        return _api_error(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            code="AI_ANALYSIS_TIMEOUT",
            message="AI 企業分析回應逾時，請稍後再試。",
            retryable=True,
        )
    if isinstance(error, LLMConnectionError):
        return _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AI_ANALYSIS_UNAVAILABLE",
            message="目前無法連線至 AI 企業分析服務，請稍後再試。",
            retryable=True,
        )
    if isinstance(error, LLMAuthenticationError):
        return _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AI_PROVIDER_AUTH_FAILED",
            message="AI 企業分析服務的憑證無法使用，請聯絡系統管理者。",
            retryable=False,
        )
    if isinstance(error, LLMRateLimitError):
        return _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AI_PROVIDER_RATE_LIMITED",
            message="AI 企業分析服務目前繁忙，請稍後再試。",
            retryable=True,
        )
    if isinstance(error, LLMRequestRejectedError):
        return _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="AI_PROVIDER_REQUEST_REJECTED",
            message="AI 企業分析服務未接受本次請求。",
            retryable=False,
        )
    if isinstance(error, LLMIncompleteResponseError):
        return _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="AI_ANALYSIS_INCOMPLETE",
            message="AI 企業分析未完成，未產生可用報告。",
            retryable=False,
        )
    if isinstance(error, LLMUpstreamError):
        return _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="AI_PROVIDER_ERROR",
            message="AI 企業分析服務發生錯誤，請稍後再試。",
            retryable=True,
        )
    return _api_error(
        status_code=status.HTTP_502_BAD_GATEWAY,
        code="AI_PROVIDER_INVALID_RESPONSE",
        message="AI 企業分析回應無法解析。",
        retryable=True,
    )


def _invalid_normalized_data(error: CompanyNormalizationError) -> HTTPException:
    return _api_error(
        status_code=status.HTTP_502_BAD_GATEWAY,
        code="UPSTREAM_INVALID_RESPONSE",
        message=f"政府資料缺少必要欄位：{error}",
        retryable=True,
    )


def _now() -> datetime:
    return datetime.now(TAIPEI_TIMEZONE)


def _partial_business_warning(error: GCISError) -> str:
    if isinstance(error, GCISTimeoutError):
        return "GCIS A3 business request timed out."
    if isinstance(error, GCISConnectionError):
        return "GCIS A3 business request could not connect."
    if isinstance(error, GCISNotFoundError):
        return "GCIS A3 business record was not found."
    if isinstance(error, GCISInvalidResponseError):
        return "GCIS A3 business response was invalid."
    if isinstance(error, GCISUpstreamHTTPError):
        return f"GCIS A3 business request returned HTTP {error.status_code}."
    return "GCIS A3 business request failed."


@router.get(
    "/search",
    response_model=CompanySearchResponse,
    responses=COMPANY_ERROR_RESPONSES,
)
async def search_companies(
    q: Annotated[str, Query(min_length=2, max_length=100)],
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    gcis_response_cache: Annotated[
        GCISResponseCache,
        Depends(get_gcis_response_cache),
    ],
    company_status: Annotated[
        str,
        Query(alias="status", pattern=r"^\d{2}$"),
    ] = "01",
) -> CompanySearchResponse:
    async def fetch_live() -> CompanySearchResponse:
        return await _fetch_live_company_search_response(
            q,
            company_status,
            gcis_client,
        )

    try:
        return await gcis_response_cache.resolve_search(
            q,
            company_status,
            10,
            fetch_live,
        )
    except GCISError as error:
        raise _translate_gcis_error(error) from error


async def _fetch_live_company_search_response(
    q: str,
    company_status: str,
    gcis_client: GCISClient,
) -> CompanySearchResponse:
    records = await gcis_client.search_companies(
        q,
        company_status=company_status,
        limit=10,
    )
    items: list[CompanySearchItem] = []
    warnings: list[str] = []
    for index, record in enumerate(records):
        try:
            result = normalize_company_search_item(record)
        except CompanyNormalizationError as error:
            warnings.append(f"Search result {index} was ignored: {error}")
            continue

        items.append(result.data)
        warnings.extend(
            f"Search result {index}: {warning}" for warning in result.warnings
        )

    return CompanySearchResponse(
        data=items,
        meta=ResponseMeta(fetched_at=_now(), warnings=warnings),
    )


@router.post(
    "/compare",
    response_model=CompanyComparisonResponse,
    responses=COMPANY_COMPARISON_ERROR_RESPONSES,
)
async def compare_companies(
    payload: CompanyComparisonRequest,
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    gcis_response_cache: Annotated[
        GCISResponseCache,
        Depends(get_gcis_response_cache),
    ],
    benchmark_service: Annotated[
        BenchmarkCatalogService,
        Depends(get_benchmark_catalog_service),
    ],
) -> CompanyComparisonResponse:
    return await _fetch_company_comparison(
        payload,
        gcis_client,
        gcis_response_cache,
        benchmark_service,
    )


@router.post(
    "/compare/analysis",
    response_model=CompanyComparisonAnalysisResponse,
    responses=COMPANY_COMPARISON_ANALYSIS_ERROR_RESPONSES,
)
async def create_company_comparison_analysis(
    request: Request,
    payload: CompanyComparisonAnalysisRequest,
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    gcis_response_cache: Annotated[
        GCISResponseCache,
        Depends(get_gcis_response_cache),
    ],
    benchmark_service: Annotated[
        BenchmarkCatalogService,
        Depends(get_benchmark_catalog_service),
    ],
    analysis_provider: Annotated[
        CompanyAnalysisProvider,
        Depends(get_company_analysis_provider),
    ],
    analysis_cache: Annotated[
        AsyncLRUTTLCache[CompanyComparisonAnalysisExecution],
        Depends(get_company_comparison_analysis_cache),
    ],
) -> CompanyComparisonAnalysisResponse:
    comparison = await _fetch_company_comparison(
        CompanyComparisonRequest(tax_ids=payload.tax_ids),
        gcis_client,
        gcis_response_cache,
        benchmark_service,
    )
    try:
        service = CompanyComparisonAnalysisService(
            analysis_provider,
            analysis_cache,
            max_attempts=request.app.state.settings.llm_max_attempts,
        )
        return await service.analyze(
            comparison,
            force_refresh=payload.force_refresh,
        )
    except (ValidationError, ValueError) as error:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="AI_COMPARISON_CONFIGURATION_ERROR",
            message="AI 公司比較設定不一致，暫時無法整理比較重點。",
            retryable=False,
        ) from error


async def _fetch_company_comparison(
    payload: CompanyComparisonRequest,
    gcis_client: GCISClient,
    gcis_response_cache: GCISResponseCache,
    benchmark_service: BenchmarkCatalogService,
) -> CompanyComparisonResponse:
    outcomes = await asyncio.gather(
        *(
            _fetch_company_bizscore_response(
                tax_id,
                gcis_client,
                gcis_response_cache,
                benchmark_service,
            )
            for tax_id in payload.tax_ids
        ),
        return_exceptions=True,
    )

    company_responses: list[CompanyBizScoreResponse] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
        company_responses.append(outcome)

    try:
        return build_company_comparison(
            payload,
            company_responses,
            generated_at=_now(),
        )
    except (CompanyComparisonConsistencyError, ValidationError) as error:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="COMPARISON_CONFIGURATION_ERROR",
            message="公司比較資料設定不一致，暫時無法產生結果。",
            retryable=False,
        ) from error


@router.get(
    "/{tax_id}",
    response_model=CompanyResponse,
    responses=COMPANY_DETAIL_ERROR_RESPONSES,
    dependencies=[Depends(_require_ascii_tax_id)],
)
async def get_company(
    tax_id: Annotated[str, Path(pattern=ASCII_TAX_ID_PATTERN)],
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    gcis_response_cache: Annotated[
        GCISResponseCache,
        Depends(get_gcis_response_cache),
    ],
) -> CompanyResponse:
    return await _fetch_company_response(
        tax_id,
        gcis_client,
        gcis_response_cache,
    )


@router.get(
    "/{tax_id}/bizscore",
    response_model=CompanyBizScoreResponse,
    responses=COMPANY_BIZSCORE_ERROR_RESPONSES,
    dependencies=[Depends(_require_ascii_tax_id)],
)
async def get_company_bizscore(
    tax_id: Annotated[str, Path(pattern=ASCII_TAX_ID_PATTERN)],
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    gcis_response_cache: Annotated[
        GCISResponseCache,
        Depends(get_gcis_response_cache),
    ],
    benchmark_service: Annotated[
        BenchmarkCatalogService,
        Depends(get_benchmark_catalog_service),
    ],
) -> CompanyBizScoreResponse:
    return await _fetch_company_bizscore_response(
        tax_id,
        gcis_client,
        gcis_response_cache,
        benchmark_service,
    )


@router.post(
    "/{tax_id}/analysis",
    response_model=CompanyAnalysisResponse,
    responses=COMPANY_ANALYSIS_ERROR_RESPONSES,
    dependencies=[Depends(_require_ascii_tax_id)],
)
async def create_company_analysis(
    request: Request,
    tax_id: Annotated[str, Path(pattern=ASCII_TAX_ID_PATTERN)],
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    gcis_response_cache: Annotated[
        GCISResponseCache,
        Depends(get_gcis_response_cache),
    ],
    benchmark_service: Annotated[
        BenchmarkCatalogService,
        Depends(get_benchmark_catalog_service),
    ],
    analysis_provider: Annotated[
        CompanyAnalysisProvider,
        Depends(get_company_analysis_provider),
    ],
) -> CompanyAnalysisResponse:
    try:
        service = CompanyAnalysisService(
            analysis_provider,
            max_attempts=request.app.state.settings.llm_max_attempts,
        )
        service.ensure_available()
    except ValueError as error:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="AI_ANALYSIS_CONFIGURATION_ERROR",
            message="AI 企業分析設定不一致，暫時無法產生報告。",
            retryable=False,
        ) from error
    except LLMProviderError as error:
        raise _translate_llm_error(error) from error

    company_bizscore = await _fetch_company_bizscore_response(
        tax_id,
        gcis_client,
        gcis_response_cache,
        benchmark_service,
    )
    try:
        analysis_input = CompanyAnalysisLLMInput(
            company=company_bizscore.data.company,
            bizscore=company_bizscore.data.bizscore,
            source_meta=company_bizscore.meta,
        )
        execution = await service.analyze(analysis_input)
    except ValidationError as error:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="AI_ANALYSIS_CONFIGURATION_ERROR",
            message="AI 企業分析輸入契約不一致，暫時無法產生報告。",
            retryable=False,
        ) from error
    except ValueError as error:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="AI_ANALYSIS_CONFIGURATION_ERROR",
            message="AI 企業分析設定不一致，暫時無法產生報告。",
            retryable=False,
        ) from error
    except CompanyAnalysisInvalidOutputError as error:
        raise _api_error(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="AI_ANALYSIS_INVALID_OUTPUT",
            message="AI 企業分析未通過資料與安全驗證。",
            retryable=False,
        ) from error
    except LLMProviderError as error:
        raise _translate_llm_error(error) from error

    provider_result = execution.provider_result
    usage = None
    if provider_result.usage is not None:
        usage = LLMTokenUsage(
            input_tokens=provider_result.usage.input_tokens,
            output_tokens=provider_result.usage.output_tokens,
            total_tokens=provider_result.usage.total_tokens,
        )
    return CompanyAnalysisResponse(
        data=CompanyAnalysisData(
            input=execution.analysis_input,
            analysis=execution.analysis,
        ),
        meta=CompanyAnalysisMeta(
            provider=provider_result.provider,
            model=provider_result.model,
            provider_response_id=provider_result.response_id,
            provider_request_id=provider_result.request_id,
            service_tier=provider_result.service_tier,
            generated_at=execution.generated_at,
            prompt_version=execution.prompt.prompt_version,
            system_prompt_sha256=execution.prompt.system_prompt_sha256,
            input_sha256=execution.input_sha256,
            output_sha256=execution.output_sha256,
            attempts=execution.attempts,
            duration_ms=execution.duration_ms,
            usage=usage,
        ),
    )


async def _fetch_company_bizscore_response(
    tax_id: str,
    gcis_client: GCISClient,
    gcis_response_cache: GCISResponseCache,
    benchmark_service: BenchmarkCatalogService,
) -> CompanyBizScoreResponse:
    company_response = await _fetch_company_response(
        tax_id,
        gcis_client,
        gcis_response_cache,
    )
    try:
        bizscore = await asyncio.to_thread(
            benchmark_service.calculate_company_bizscore,
            company_response.data,
            input_partial=company_response.meta.partial,
            input_warnings=company_response.meta.warnings,
        )
    except BenchmarkCatalogUnavailableError as error:
        raise _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="BENCHMARK_UNAVAILABLE",
            message="目前無法讀取 BizScore 同業基準資料，請稍後再試。",
            retryable=True,
        ) from error
    except BenchmarkCatalogInvalidError as error:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="BENCHMARK_CONFIGURATION_ERROR",
            message="BizScore 同業基準資料設定不一致，暫時無法計分。",
            retryable=False,
        ) from error

    return CompanyBizScoreResponse(
        data=CompanyBizScoreData(
            company=company_response.data,
            bizscore=bizscore,
        ),
        meta=company_response.meta,
    )


async def _fetch_company_response(
    tax_id: str,
    gcis_client: GCISClient,
    gcis_response_cache: GCISResponseCache,
) -> CompanyResponse:
    async def fetch_live() -> CompanyResponse:
        return await _fetch_live_company_response(tax_id, gcis_client)

    try:
        return await gcis_response_cache.resolve_company(tax_id, fetch_live)
    except GCISError as error:
        raise _translate_gcis_error(error) from error
    except CompanyNormalizationError as error:
        raise _invalid_normalized_data(error) from error


async def _fetch_live_company_response(
    tax_id: str,
    gcis_client: GCISClient,
) -> CompanyResponse:
    basic_outcome, business_outcome = await asyncio.gather(
        gcis_client.get_company_basic(tax_id),
        gcis_client.get_company_business_items(tax_id),
        return_exceptions=True,
    )

    if isinstance(basic_outcome, GCISError):
        raise basic_outcome
    if isinstance(basic_outcome, BaseException):
        raise basic_outcome

    business_error: GCISError | None = None
    if isinstance(business_outcome, GCISError):
        business_error = business_outcome
        business_record = None
    elif isinstance(business_outcome, BaseException):
        raise business_outcome
    else:
        business_record = business_outcome[0]

    result = normalize_company_data(
        basic_outcome[0],
        business_record,
    )

    warnings = list(result.warnings)
    if len(basic_outcome) > 1:
        warnings.append("GCIS A1 returned multiple records; the first record was used.")
    if not isinstance(business_outcome, BaseException) and len(business_outcome) > 1:
        warnings.append("GCIS A3 returned multiple records; the first record was used.")
    if business_error is not None:
        warnings.append(_partial_business_warning(business_error))

    return CompanyResponse(
        data=result.data,
        meta=ResponseMeta(
            fetched_at=_now(),
            partial=result.partial,
            warnings=warnings,
        ),
    )

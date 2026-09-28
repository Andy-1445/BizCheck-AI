from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.schemas.common import ErrorDetail


async def validation_exception_handler(
    request: Request,
    error: RequestValidationError,
) -> Response:
    """Use public error contracts for company tax ID validation failures."""

    if _is_company_comparison_analysis_request(request):
        detail = ErrorDetail(
            code="INVALID_COMPARISON_ANALYSIS_REQUEST",
            message="請輸入 2 或 3 個不重複的 8 碼統一編號，force_refresh 必須是布林值。",
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": detail.model_dump()},
        )

    if _is_company_comparison_request(request):
        detail = ErrorDetail(
            code="INVALID_COMPARISON_REQUEST",
            message="比較請求需提供 2 至 3 個不重複的 8 碼統一編號。",
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": detail.model_dump()},
        )

    if any(_is_tax_id_path_error(item) for item in error.errors()):
        detail = ErrorDetail(
            code="INVALID_TAX_ID",
            message="請輸入 8 碼統一編號。",
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": detail.model_dump()},
        )

    return await request_validation_exception_handler(request, error)


def _is_company_comparison_request(request: Request) -> bool:
    return (
        request.method == "POST"
        and request.url.path.rstrip("/").endswith("/companies/compare")
    )


def _is_company_comparison_analysis_request(request: Request) -> bool:
    return (
        request.method == "POST"
        and request.url.path.rstrip("/").endswith(
            "/companies/compare/analysis"
        )
    )


def _is_tax_id_path_error(item: dict[str, Any]) -> bool:
    location = item.get("loc", ())
    return (
        isinstance(location, (tuple, list))
        and len(location) >= 2
        and location[0] == "path"
        and location[-1] == "tax_id"
    )

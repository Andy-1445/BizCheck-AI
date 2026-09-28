from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError

from app.schemas.bizscore import CompanyBizScoreResponse
from app.schemas.company_comparison import (
    COMPANY_COMPARISON_DISCLAIMER,
    CompanyComparisonContext,
    CompanyComparisonData,
    CompanyComparisonItem,
    CompanyComparisonMeta,
    CompanyComparisonMetrics,
    CompanyComparisonRequest,
    CompanyComparisonResponse,
    CompanyComparisonWarning,
    ComparisonMetricKey,
    PeerComparisonScope,
)
from app.services.company_normalizer import calculate_company_age_years


class CompanyComparisonConsistencyError(RuntimeError):
    """Comparison inputs are individually valid but mutually inconsistent."""


def build_company_comparison(
    request: CompanyComparisonRequest,
    company_responses: list[CompanyBizScoreResponse],
    *,
    generated_at: datetime,
) -> CompanyComparisonResponse:
    if len(company_responses) != len(request.tax_ids):
        raise CompanyComparisonConsistencyError(
            "Comparison responses must match the requested company count."
        )

    items: list[CompanyComparisonItem] = []
    for input_index, (requested_tax_id, response) in enumerate(
        zip(request.tax_ids, company_responses, strict=True)
    ):
        if response.data.company.tax_id != requested_tax_id:
            raise CompanyComparisonConsistencyError(
                "GCIS returned a company that does not match the requested tax ID."
            )
        try:
            items.append(
                CompanyComparisonItem(
                    input_index=input_index,
                    company=response.data.company,
                    bizscore=response.data.bizscore,
                    metrics=_build_metrics(response),
                    source_meta=response.meta,
                )
            )
        except ValidationError as error:
            raise CompanyComparisonConsistencyError(
                "A company response is inconsistent with the comparison contract."
            ) from error

    catalog_versions = {
        item.bizscore.benchmark.catalog_version for item in items
    }
    as_of_dates = {item.bizscore.as_of for item in items}
    catalog_as_of_dates = {
        item.bizscore.benchmark.catalog_as_of for item in items
    }
    catalog_mapping_versions = {
        item.bizscore.benchmark.catalog_industry_mapping_version
        for item in items
    }
    classification_versions = {
        item.bizscore.benchmark.classification_version for item in items
    }
    if (
        len(catalog_versions) != 1
        or len(as_of_dates) != 1
        or catalog_as_of_dates != as_of_dates
        or len(catalog_mapping_versions) != 1
        or len(classification_versions) != 1
    ):
        raise CompanyComparisonConsistencyError(
            "All comparison items must use one Benchmark catalog and as-of date, "
            "with one mapping contract."
        )

    snapshot_provenance: dict[
        tuple[str | None, str | None],
        tuple[str | None, str | None, int],
    ] = {}
    for item in items:
        benchmark = item.bizscore.benchmark
        snapshot_key = (benchmark.industry_code, benchmark.snapshot_version)
        provenance = (
            benchmark.snapshot_source_version,
            benchmark.snapshot_checksum_sha256,
            benchmark.sample_count,
        )
        prior = snapshot_provenance.setdefault(snapshot_key, provenance)
        if prior != provenance:
            raise CompanyComparisonConsistencyError(
                "Comparison items use inconsistent Benchmark snapshot metadata."
            )

    peer_scope = _peer_comparison_scope(items)
    industry_codes = [
        (
            item.bizscore.industry.primary_group.category_code
            if item.bizscore.industry.primary_group is not None
            else None
        )
        for item in items
    ]
    same_primary_industry = (
        all(code is not None for code in industry_codes)
        and len(set(industry_codes)) == 1
    )
    warnings = _build_warnings(items, peer_scope)

    try:
        return CompanyComparisonResponse(
            data=CompanyComparisonData(
                items=items,
                context=CompanyComparisonContext(
                    benchmark_catalog_version=next(iter(catalog_versions)),
                    benchmark_as_of=next(iter(catalog_as_of_dates)),
                    peer_comparison_scope=peer_scope,
                    same_primary_industry=same_primary_industry,
                ),
                disclaimer=COMPANY_COMPARISON_DISCLAIMER,
            ),
            meta=CompanyComparisonMeta(
                generated_at=generated_at,
                requested_tax_ids=request.tax_ids,
                requested_count=len(request.tax_ids),
                returned_count=len(items),
                has_partial_source_data=any(
                    item.source_meta.partial for item in items
                ),
                has_provisional_scores=any(
                    item.bizscore.provisional for item in items
                ),
                has_unscored_companies=any(
                    item.bizscore.score is None for item in items
                ),
                warnings=warnings,
            ),
        )
    except ValidationError as error:
        raise CompanyComparisonConsistencyError(
            "The aggregated comparison response is internally inconsistent."
        ) from error


def _build_metrics(
    response: CompanyBizScoreResponse,
) -> CompanyComparisonMetrics:
    company = response.data.company
    bizscore = response.data.bizscore
    registration_status = company.status.description or company.status.code
    company_age_years = calculate_company_age_years(
        company.established_at,
        as_of=bizscore.as_of,
    )
    values: dict[ComparisonMetricKey, object | None] = {
        "registration_status": registration_status,
        "company_age": company_age_years,
        "registered_capital": company.capital.registered,
        "last_changed_at": company.last_changed_at,
        "bizscore": bizscore.score,
        "peer_index": bizscore.peer_benchmark.peer_index,
        "industry": bizscore.benchmark.industry_code,
    }
    missing = [key for key, value in values.items() if value is None]
    bizscore_status = (
        "unavailable"
        if bizscore.score is None
        else "provisional"
        if bizscore.provisional
        else "complete"
    )
    return CompanyComparisonMetrics(
        registration_status=registration_status,
        company_age_years=company_age_years,
        registered_capital=company.capital.registered,
        last_changed_at=company.last_changed_at,
        bizscore=bizscore.score,
        bizscore_coverage=bizscore.coverage,
        bizscore_provisional=bizscore.provisional,
        bizscore_status=bizscore_status,
        peer_index=bizscore.peer_benchmark.peer_index,
        industry_code=bizscore.benchmark.industry_code,
        missing=missing,
    )


def _peer_comparison_scope(
    items: list[CompanyComparisonItem],
) -> PeerComparisonScope:
    scopes = [
        (
            item.bizscore.peer_benchmark.dimension.available,
            item.bizscore.peer_benchmark.peer_index,
            item.bizscore.benchmark.industry_code,
            item.bizscore.benchmark.snapshot_version,
        )
        for item in items
    ]
    if not all(
        available
        and peer_index is not None
        and industry_code is not None
        and snapshot_version is not None
        for available, peer_index, industry_code, snapshot_version in scopes
    ):
        return "unavailable"
    if len(
        {
            (industry_code, snapshot_version)
            for _, _, industry_code, snapshot_version in scopes
        }
    ) == 1:
        return "same_industry_snapshot"
    return "different_industry_snapshots"


def _build_warnings(
    items: list[CompanyComparisonItem],
    peer_scope: PeerComparisonScope,
) -> list[CompanyComparisonWarning]:
    warnings: list[CompanyComparisonWarning] = []
    for item in items:
        tax_id = item.company.tax_id
        if item.source_meta.data_freshness == "stale_cache":
            warnings.append(
                CompanyComparisonWarning(
                    code="STALE_SOURCE_DATA",
                    tax_id=tax_id,
                    message=(
                        "GCIS 即時資料暫不可用；此公司使用最近一次成功資料快照。"
                    ),
                )
            )
        if item.source_meta.partial:
            warnings.append(
                CompanyComparisonWarning(
                    code="PARTIAL_SOURCE_DATA",
                    tax_id=tax_id,
                    message="此公司的政府公開資料為部分資料。",
                )
            )
        for message in item.source_meta.warnings:
            warnings.append(
                CompanyComparisonWarning(
                    code="SOURCE_WARNING",
                    tax_id=tax_id,
                    message=_truncate(message),
                )
            )
        if item.bizscore.provisional:
            warnings.append(
                CompanyComparisonWarning(
                    code="PROVISIONAL_SCORE",
                    tax_id=tax_id,
                    message="此公司的 BizScore 為暫定結果。",
                )
            )
        if item.bizscore.score is None:
            warnings.append(
                CompanyComparisonWarning(
                    code="NO_NUMERIC_SCORE",
                    tax_id=tax_id,
                    message="此公司的可計分資料不足，沒有數字總分。",
                )
            )

    if peer_scope == "different_industry_snapshots":
        warnings.append(
            CompanyComparisonWarning(
                code="DIFFERENT_INDUSTRY_BENCHMARKS",
                message=(
                    "各公司使用不同產業快照；PR 僅表示各自在所屬同業群組中的位置。"
                ),
            )
        )
    elif peer_scope == "unavailable":
        warnings.append(
            CompanyComparisonWarning(
                code="PEER_COMPARISON_UNAVAILABLE",
                message="至少一家公司缺少可用的同業比較資料。",
            )
        )
    return warnings


def _truncate(value: str, limit: int = 240) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"

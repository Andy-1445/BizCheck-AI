from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.llm_analysis import (
    CompanyAnalysisLLMOutput,
    build_llm_output_json_schema,
)
from app.services.llm_prompt import LLMOutputSafetyError
from app.services.llm_validation_diagnostics import (
    clear_latest_llm_validation_diagnostic,
    get_latest_llm_validation_diagnostic,
    log_llm_validation_failure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_schema_diagnostic_logs_only_allowlisted_fields_and_reason_codes(
    caplog,
) -> None:
    sensitive_company = "不可記錄的公司名稱"
    sensitive_tax_id = "20828393"
    try:
        CompanyAnalysisLLMOutput.model_validate(
            {
                "headline": sensitive_company,
                sensitive_company: sensitive_tax_id,
            }
        )
    except ValidationError as error:
        with caplog.at_level(logging.WARNING, logger="bizcheck.llm.validation"):
            diagnostic = log_llm_validation_failure(
                task="company_analysis",
                provider="thu",
                provider_mode="json_schema",
                attempt=1,
                error=error,
                response_schema=build_llm_output_json_schema(),
            )
    else:
        raise AssertionError("Invalid output unexpectedly passed validation.")

    log_text = caplog.text
    assert sensitive_company not in log_text
    assert sensitive_tax_id not in log_text
    assert diagnostic.stage == "schema"
    assert diagnostic.provider_mode == "json_schema"
    assert all("不可記錄" not in field for field in diagnostic.fields)
    assert "required_field_missing" in diagnostic.reasons


def test_latest_diagnostic_snapshot_contains_only_sanitized_contract() -> None:
    clear_latest_llm_validation_diagnostic()
    error = LLMOutputSafetyError(
        "findings[0].observation contains numeric claims not grounded by evidence"
    )

    expected = log_llm_validation_failure(
        task="company_analysis",
        provider="thu",
        provider_mode="json_schema",
        attempt=1,
        error=error,
        response_schema=build_llm_output_json_schema(),
    )

    snapshot = get_latest_llm_validation_diagnostic()
    assert snapshot is not None
    sequence, diagnostic = snapshot
    assert sequence == 1
    assert diagnostic == expected
    assert diagnostic.fields == ("/findings/0/observation",)
    assert diagnostic.reasons == ("numeric_claim_not_grounded",)


def test_safety_diagnostic_never_logs_raw_exception_text(caplog) -> None:
    sensitive_company = "不可記錄的公司名稱"
    sensitive_tax_id = "20828393"
    raw_error = LLMOutputSafetyError(
        "forbidden phrases: "
        + sensitive_company
        + sensitive_tax_id
        + "; provenance.company_tax_id does not match input"
    )

    with caplog.at_level(logging.WARNING, logger="bizcheck.llm.validation"):
        diagnostic = log_llm_validation_failure(
            task="company_analysis",
            provider="thu",
            provider_mode="json_object",
            attempt=2,
            error=raw_error,
            response_schema=build_llm_output_json_schema(),
        )

    log_text = caplog.text
    assert sensitive_company not in log_text
    assert sensitive_tax_id not in log_text
    assert diagnostic.stage == "safety"
    assert diagnostic.provider_mode == "json_object"
    assert "forbidden_phrase" in diagnostic.reasons
    assert "provenance_mismatch" in diagnostic.reasons
    serialized = json.dumps(diagnostic.reasons)
    assert sensitive_company not in serialized


def test_nested_schema_field_from_defs_is_logged_without_unknown_marker(
    caplog,
) -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "samples"
            / "llm"
            / "company-analysis-output-v1.example.json"
        ).read_text(encoding="utf-8")
    )
    del payload["findings"][0]["evidence_paths"]

    try:
        CompanyAnalysisLLMOutput.model_validate(payload)
    except ValidationError as error:
        with caplog.at_level(logging.WARNING, logger="bizcheck.llm.validation"):
            diagnostic = log_llm_validation_failure(
                task="company_analysis",
                provider="thu",
                provider_mode="json_schema",
                attempt=1,
                error=error,
                response_schema=build_llm_output_json_schema(),
            )
    else:
        raise AssertionError("Missing nested field unexpectedly passed validation.")

    assert diagnostic.fields == ("/findings/0/evidence_paths",)
    assert diagnostic.reasons == ("required_field_missing",)
    assert "<unknown>" not in caplog.text


@pytest.mark.parametrize(
    ("mutate", "expected_field", "expected_reason"),
    [
        pytest.param(
            lambda payload: payload["provenance"].update(
                {"evidence_paths_used": ["/source_meta/source"]}
            ),
            "/provenance/evidence_paths_used",
            "provenance_evidence_union_mismatch",
            id="evidence-union",
        ),
        pytest.param(
            lambda payload: payload["findings"].append(
                dict(payload["findings"][0])
            ),
            "/findings",
            "duplicate_finding_topic",
            id="duplicate-finding-topic",
        ),
        pytest.param(
            lambda payload: payload["limitations"].append(
                dict(payload["limitations"][0])
            ),
            "/limitations",
            "duplicate_limitation_code",
            id="duplicate-limitation-code",
        ),
        pytest.param(
            lambda payload: payload.update(
                {
                    "limitations": [
                        {
                            "code": "partial_source_data",
                            "message": "資料有缺漏。",
                            "related_evidence_paths": [],
                        },
                        {
                            "code": "missing_dimension",
                            "message": "構面有缺漏。",
                            "related_evidence_paths": [],
                        },
                    ]
                }
            ),
            "/limitations",
            "required_limitation_missing",
            id="required-limitations",
        ),
    ],
)
def test_root_value_errors_use_fixed_safe_reason_codes(
    mutate,
    expected_field: str,
    expected_reason: str,
) -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "samples"
            / "llm"
            / "company-analysis-output-v1.example.json"
        ).read_text(encoding="utf-8")
    )
    mutate(payload)

    try:
        CompanyAnalysisLLMOutput.model_validate(payload)
    except ValidationError as error:
        diagnostic = log_llm_validation_failure(
            task="company_analysis",
            provider="thu",
            provider_mode="json_schema",
            attempt=1,
            error=error,
            response_schema=build_llm_output_json_schema(),
        )
    else:
        raise AssertionError("Invalid root constraint unexpectedly passed.")

    assert diagnostic.fields == (expected_field,)
    assert diagnostic.reasons == (expected_reason,)

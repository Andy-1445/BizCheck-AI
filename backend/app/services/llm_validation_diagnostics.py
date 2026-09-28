from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from threading import Lock

from pydantic import ValidationError

from app.services.llm_prompt import LLMOutputSafetyError
from app.services.llm_provider import LLMInvalidResponseError


LOGGER = logging.getLogger("bizcheck.llm.validation")
_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_FIELD_REFERENCE_PATTERN = re.compile(
    r"^(?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\]|\.[A-Za-z_][A-Za-z0-9_]*)*)"
)
_PYDANTIC_REASON_CODES = {
    "missing": "required_field_missing",
    "extra_forbidden": "extra_field_forbidden",
    "literal_error": "enum_or_constant_mismatch",
    "string_pattern_mismatch": "string_pattern_mismatch",
    "string_too_short": "string_too_short",
    "string_too_long": "string_too_long",
    "too_short": "collection_too_short",
    "too_long": "collection_too_long",
    "list_type": "list_type_required",
    "string_type": "string_type_required",
    "date_from_datetime_parsing": "date_format_invalid",
    "date_parsing": "date_format_invalid",
    "value_error": "value_constraint_failed",
}
_PYDANTIC_VALUE_ERROR_RULES = (
    (
        "insufficient_data output must not contain findings.",
        "/findings",
        "insufficient_data_forbids_findings",
    ),
    (
        "completed output must contain at least one finding.",
        "/findings",
        "completed_requires_finding",
    ),
    (
        "findings must not repeat the same topic.",
        "/findings",
        "duplicate_finding_topic",
    ),
    (
        "company observations must not repeat tax ID and topic.",
        "/company_observations",
        "duplicate_company_observation_topic",
    ),
    (
        "comparison observations must not repeat a topic.",
        "/comparison_observations",
        "duplicate_comparison_observation_topic",
    ),
    (
        "limitations must not repeat the same code.",
        "/limitations",
        "duplicate_limitation_code",
    ),
    (
        "limitations must include public_data_only and ai_generated.",
        "/limitations",
        "required_limitation_missing",
    ),
    (
        "limitations must include public_data_only, ai_generated, and not_ranked.",
        "/limitations",
        "required_limitation_missing",
    ),
    (
        "provenance.evidence_paths_used must exactly match all referenced paths.",
        "/provenance/evidence_paths_used",
        "provenance_evidence_union_mismatch",
    ),
    (
        "provenance.evidence_paths_used must exactly match referenced paths.",
        "/provenance/evidence_paths_used",
        "provenance_evidence_union_mismatch",
    ),
    (
        "finding.evidence_paths must not contain duplicates.",
        "/findings",
        "duplicate_evidence_path",
    ),
    (
        "verification_item.related_evidence_paths must not contain duplicates.",
        "/verification_items",
        "duplicate_evidence_path",
    ),
    (
        "limitation.related_evidence_paths must not contain duplicates.",
        "/limitations",
        "duplicate_evidence_path",
    ),
    (
        "provenance.evidence_paths_used must not contain duplicates.",
        "/provenance/evidence_paths_used",
        "duplicate_evidence_path",
    ),
)


@dataclass(frozen=True, slots=True)
class LLMValidationDiagnostic:
    task: str
    provider: str
    provider_mode: str
    attempt: int
    stage: str
    fields: tuple[str, ...]
    reasons: tuple[str, ...]


_LATEST_DIAGNOSTIC_LOCK = Lock()
_LATEST_DIAGNOSTIC_SEQUENCE = 0
_LATEST_DIAGNOSTIC: LLMValidationDiagnostic | None = None


def get_latest_llm_validation_diagnostic(
) -> tuple[int, LLMValidationDiagnostic] | None:
    with _LATEST_DIAGNOSTIC_LOCK:
        if _LATEST_DIAGNOSTIC is None:
            return None
        return _LATEST_DIAGNOSTIC_SEQUENCE, _LATEST_DIAGNOSTIC


def clear_latest_llm_validation_diagnostic() -> None:
    global _LATEST_DIAGNOSTIC, _LATEST_DIAGNOSTIC_SEQUENCE
    with _LATEST_DIAGNOSTIC_LOCK:
        _LATEST_DIAGNOSTIC = None
        _LATEST_DIAGNOSTIC_SEQUENCE = 0


def log_llm_validation_failure(
    *,
    task: str,
    provider: str,
    provider_mode: str = "unknown",
    attempt: int,
    error: Exception,
    response_schema: dict[str, object],
) -> LLMValidationDiagnostic:
    """Log only allowlisted field paths and reason codes for rejected output.

    The raw provider output, exception text, Pydantic input values, company
    identifiers, and verified CompanyData are intentionally never logged.
    """

    diagnostic = build_llm_validation_diagnostic(
        task=task,
        provider=provider,
        provider_mode=provider_mode,
        attempt=attempt,
        error=error,
        response_schema=response_schema,
    )
    global _LATEST_DIAGNOSTIC, _LATEST_DIAGNOSTIC_SEQUENCE
    with _LATEST_DIAGNOSTIC_LOCK:
        _LATEST_DIAGNOSTIC_SEQUENCE += 1
        _LATEST_DIAGNOSTIC = diagnostic
    LOGGER.warning(
        "llm_output_validation_failed %s",
        json.dumps(
            asdict(diagnostic),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return diagnostic


def build_llm_validation_diagnostic(
    *,
    task: str,
    provider: str,
    provider_mode: str = "unknown",
    attempt: int,
    error: Exception,
    response_schema: dict[str, object],
) -> LLMValidationDiagnostic:
    allowed_fields = _schema_field_names(response_schema)
    safe_task = _safe_token(task, fallback="unknown_task")
    safe_provider = _safe_token(provider, fallback="unknown_provider")
    safe_provider_mode = _safe_token(
        provider_mode,
        fallback="unknown_mode",
    )

    if isinstance(error, ValidationError):
        fields: list[str] = []
        reasons: list[str] = []
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )[:20]:
            location = item.get("loc")
            error_type = item.get("type")
            safe_field, safe_reason = _classify_pydantic_issue(
                error_type=error_type,
                message=item.get("msg"),
                location=location,
                allowed_fields=allowed_fields,
            )
            fields.append(safe_field)
            reasons.append(
                safe_reason
            )
        return LLMValidationDiagnostic(
            task=safe_task,
            provider=safe_provider,
            provider_mode=safe_provider_mode,
            attempt=attempt,
            stage="schema",
            fields=_deduplicate(fields) or ("$",),
            reasons=_deduplicate(reasons) or ("schema_constraint_failed",),
        )

    if isinstance(error, LLMOutputSafetyError):
        fields, reasons = _classify_safety_error(str(error), allowed_fields)
        return LLMValidationDiagnostic(
            task=safe_task,
            provider=safe_provider,
            provider_mode=safe_provider_mode,
            attempt=attempt,
            stage="safety",
            fields=fields,
            reasons=reasons,
        )

    if isinstance(error, LLMInvalidResponseError):
        return LLMValidationDiagnostic(
            task=safe_task,
            provider=safe_provider,
            provider_mode=safe_provider_mode,
            attempt=attempt,
            stage="provider_response",
            fields=("$",),
            reasons=("invalid_json_or_response_shape",),
        )

    return LLMValidationDiagnostic(
        task=safe_task,
        provider=safe_provider,
        provider_mode=safe_provider_mode,
        attempt=attempt,
        stage="unknown",
        fields=("$",),
        reasons=("validation_failed",),
    )


def _classify_safety_error(
    message: str,
    allowed_fields: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    fields: list[str] = []
    reasons: list[str] = []
    for segment in message.split("; ")[:20]:
        field = _safe_field_reference(segment, allowed_fields)
        if "violates THU fixed output contract" in segment:
            fields.append(field)
            reasons.append("fixed_output_contract_mismatch")
        elif "violates THU narrative policy" in segment:
            fields.append(field)
            reasons.append("narrative_policy_violation")
        elif "contains unsupported narrative assertion" in segment:
            fields.append(field)
            reasons.append("unsupported_narrative_assertion")
        elif "contains unsupported funding capability inference" in segment:
            fields.append(field)
            reasons.append("unsupported_funding_capability_inference")
        elif "violates THU verification contract" in segment:
            fields.append(field)
            reasons.append("verification_contract_mismatch")
        elif "evidence path does not exist" in segment:
            fields.append("/provenance/evidence_paths_used")
            reasons.append("evidence_path_missing")
        elif "must point to a leaf value" in segment:
            fields.append("/provenance/evidence_paths_used")
            reasons.append("evidence_path_not_leaf")
        elif "must not be a null leaf" in segment:
            fields.append("/provenance/evidence_paths_used")
            reasons.append("evidence_path_null")
        elif "volatile timestamps cannot be evidence" in segment:
            fields.append("/provenance/evidence_paths_used")
            reasons.append("volatile_evidence_forbidden")
        elif "forbidden phrases" in segment:
            fields.append("$text")
            reasons.append("forbidden_phrase")
        elif "numeric claims" in segment:
            fields.append(field)
            reasons.append("numeric_claim_not_grounded")
        elif "evidence does not belong" in segment:
            fields.append(field)
            reasons.append("evidence_company_mismatch")
        elif "evidence path does not support topic" in segment or (
            "evidence does not support" in segment
        ):
            fields.append(field)
            reasons.append("evidence_topic_mismatch")
        elif "required limitation is missing" in segment:
            fields.append("/limitations")
            reasons.append("required_limitation_missing")
        elif "limitation is not supported by input" in segment:
            fields.append("/limitations")
            reasons.append("unsupported_limitation")
        elif "must not contain Markdown" in segment:
            fields.append(field)
            reasons.append("markdown_forbidden")
        elif "must contain Chinese text" in segment:
            fields.append(field)
            reasons.append("zh_tw_text_required")
        elif "must not be blank" in segment:
            fields.append(field)
            reasons.append("blank_text")
        elif "must be an actual question" in segment:
            fields.append(field)
            reasons.append("question_format_required")
        elif "caveat is required" in segment:
            fields.append(field)
            reasons.append("caveat_required")
        elif "status must be" in segment:
            fields.append("/status")
            reasons.append("status_mismatch")
        elif "provenance." in segment or "evidence_paths_used" in segment:
            fields.append(field if field != "$" else "/provenance")
            reasons.append("provenance_mismatch")
        elif "company_observations must cover" in segment:
            fields.append("/company_observations")
            reasons.append("company_coverage_or_order_mismatch")
        else:
            fields.append(field)
            reasons.append("safety_constraint_failed")
    return (
        _deduplicate(fields) or ("$",),
        _deduplicate(reasons) or ("safety_constraint_failed",),
    )


def _schema_field_names(schema: object) -> frozenset[str]:
    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        properties = value.get("properties")
        if isinstance(properties, dict):
            for key, child in properties.items():
                if isinstance(key, str) and _SAFE_TOKEN_PATTERN.fullmatch(key):
                    names.add(key)
                visit(child)
        definitions = value.get("$defs")
        if isinstance(definitions, dict):
            for definition in definitions.values():
                visit(definition)
        for key in ("items", "anyOf", "oneOf", "allOf"):
            visit(value.get(key))

    visit(schema)
    return frozenset(names)


def _safe_location(
    location: object,
    allowed_fields: frozenset[str],
) -> str:
    if not isinstance(location, (tuple, list)):
        return "$"
    parts: list[str] = []
    for segment in location:
        if isinstance(segment, bool):
            parts.append("<unknown>")
        elif isinstance(segment, int) and 0 <= segment <= 999:
            parts.append(str(segment))
        elif isinstance(segment, str) and segment in allowed_fields:
            parts.append(segment)
        else:
            parts.append("<unknown>")
    return "/" + "/".join(parts) if parts else "$"


def _classify_pydantic_issue(
    *,
    error_type: object,
    message: object,
    location: object,
    allowed_fields: frozenset[str],
) -> tuple[str, str]:
    if error_type == "value_error" and isinstance(message, str):
        for fixed_message, field, reason in _PYDANTIC_VALUE_ERROR_RULES:
            if message.endswith(fixed_message):
                return field, reason
    safe_type = error_type if isinstance(error_type, str) else ""
    return (
        _safe_location(location, allowed_fields),
        _PYDANTIC_REASON_CODES.get(
            safe_type,
            "schema_constraint_failed",
        ),
    )


def _safe_field_reference(
    message: str,
    allowed_fields: frozenset[str],
) -> str:
    match = _FIELD_REFERENCE_PATTERN.match(message)
    if match is None:
        return "$"
    raw_parts = re.sub(r"\[([0-9]+)\]", r".\1", match.group("field")).split(".")
    safe_parts: list[str] = []
    for part in raw_parts:
        if part.isdigit() and int(part) <= 999:
            safe_parts.append(part)
        elif part in allowed_fields:
            safe_parts.append(part)
        else:
            safe_parts.append("<unknown>")
    return "/" + "/".join(safe_parts)


def _safe_token(value: str, *, fallback: str) -> str:
    stripped = value.strip().lower()
    return stripped if _SAFE_TOKEN_PATTERN.fullmatch(stripped) else fallback


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))

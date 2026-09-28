from __future__ import annotations

import json
import re
import unicodedata

from app.schemas.llm_analysis import (
    AI_ANALYSIS_DISCLAIMER,
    CompanyAnalysisLLMInput,
)
from app.schemas.llm_comparison_analysis import (
    AI_COMPARISON_DISCLAIMER,
    CompanyComparisonLLMInput,
)
from app.services.llm_comparison_prompt import CompanyComparisonPromptPackage
from app.services.llm_output_normalizer import build_server_owned_manifest_schema
from app.services.llm_prompt import (
    CompanyAnalysisPromptPackage,
    LLMOutputSafetyError,
    extract_numeric_claims,
)


THUPromptPackage = CompanyAnalysisPromptPackage | CompanyComparisonPromptPackage
# Chinese words such as「一般」「一致」「另外」are not numeric claims.
# Evidence/number validation remains mandatory after generation in every mode.
_THU_NARRATIVE_NO_DIGITS_PATTERN = r"^[^0-9０-９%％]*$"
THU_SAFETY_REVISION = "2026-09-24.1"
THU_VERIFICATION_REASONS = (
    "公開登記資料不包含個別交易安排。",
    "公開登記資料不能取代本次交易文件的查核。",
    "登記欄位與本次交易文件仍需交叉核對。",
)
_FIXED_COMPANY_FIELDS = (
    "status", "headline", "overall_observation", "findings", "limitations",
)
_FIXED_COMPARISON_FIELDS = (
    "status", "headline", "overall_observation", "company_observations",
    "comparison_observations", "limitations",
)
_THU_NARRATIVE_FIELDS = frozenset(
    {
        "caveat",
        "headline",
        "message",
        "observation",
        "overall_observation",
        "question",
        "reason",
        "title",
    }
)


def build_thu_system_instruction(prompt: THUPromptPackage) -> str:
    """Add a complete, input-consistent JSON example for non-strict gateways."""

    system_message, user_message = prompt.messages
    task = getattr(prompt, "task", "company_analysis")
    example = _build_input_consistent_example(prompt, user_message.content)
    provider_schema = build_thu_response_schema(prompt)
    allowed_evidence_paths = _allowed_evidence_paths(
        _extract_verified_input(user_message.content),
        task=task,
    )
    limitation_codes = [
        item["code"]
        for item in example["limitations"]
        if isinstance(item, dict) and isinstance(item.get("code"), str)
    ]
    if task == "company_comparison_analysis":
        observation_contract = (
            "company_observations 與 comparison_observations 必須逐項沿用本次 "
            "complete_output_example 已提供的所有欄位，不得改寫、刪除或增加項目。"
        )
        example_description = (
            "下方範例已依本次 verified input 產生；company_observations 與 "
            "comparison_observations 是後端挑選的安全資料觀察，必須逐字沿用。"
        )
    else:
        observation_contract = (
            "findings 必須逐項沿用本次 complete_output_example 已提供的 topic、title、"
            "observation、evidence_paths 與 caveat，不得改寫、刪除或增加 finding。"
        )
        example_description = (
            "下方範例已依本次 verified input 產生；findings 是後端挑選的安全資料觀察，"
            "必須逐字沿用。"
        )
    schema_json = json.dumps(
        provider_schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    example_json = json.dumps(
        example,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    # Input-derived version strings remain data even inside system examples.
    for token, escape in (("&", r"\u0026"), ("<", r"\u003c"), (">", r"\u003e")):
        schema_json = schema_json.replace(token, escape)
        example_json = example_json.replace(token, escape)
    field_checklist = _field_checklist(prompt)
    return (
        f"{system_message.content}\n\n"
        "<thu_json_output_rules>\n"
        "你必須只輸出一個 JSON object。第一個字元必須是 {，最後一個字元必須是 }。\n"
        "不得輸出 Markdown、code fence、思考過程、前言或後記。\n"
        "所有 object 都禁止額外欄位；所有範例中出現的欄位都必須保留。\n"
        f"{field_checklist}\n"
        "陣列即使沒有可用內容也要輸出 []，允許 null 的欄位不得改成空字串。\n"
        "enum、版本、固定 disclaimer 必須逐字使用 Schema 或 verified input 的值。\n"
        "provenance 的版本、日期與統編必須逐字複製 verified input。\n"
        "provenance.evidence_paths_used 必須固定輸出 []；後端會依其他欄位的實際引用重新建立此欄位，模型不得自行加入路徑。\n"
        "所有 evidence path 必須是 verified input 中存在且非 null 的葉節點。\n"
        "所有 evidence path 只能逐字選用 allowed_evidence_paths_json 清單中的值。\n"
        "每個 finding 的 evidence_paths 只保留最直接支持該 topic 的路徑，最多六筆；verification_items 與 limitations 沒有必要時使用空陣列。\n"
        f"{observation_contract}\n"
        "本次 limitations.code 必須且只能使用 required_limitation_codes_json 清單中的值，不得自行增加其他 limitation。\n"
        "本段 THU 專用規則比前文更嚴格：即使有 evidence，所有自然語言欄位仍不得寫阿拉伯數字或全形數字；不得用中文數字、代字、諧音或佔位字描述分數、PR、百分比、金額、日期、年數、筆數或排名。一般、一致、另外等正常用語可以使用；數值已由固定規則介面顯示。\n"
        "status、headline、overall_observation、limitations 也必須逐字沿用 complete_output_example。資料不足時 findings 必須為 []。只有 verification_items 可以依輸入提出具體查核問題；不得在問題或 reason 中新增公司狀態、數量或能力斷言。\n"
        "所有說明文字使用繁體中文。\n"
        "verification_items 產生一至四題不重複的查核問題，以 complete_output_example 的文件查核問題為起點。question 以『是否已』開頭並以問號結尾，聚焦可執行的文件核對動作：簽約主體與統編、簽約授權、合約付款及驗收條款、最新登記文件、擬合作業務範圍。\n"
        "question 不要描述或評價公司，不要談資金動員、調度、籌資、融資或財務實力；也不要把資本、年資或異動推論成任何能力。不得預設資本有差異、公司有異常或文件已缺漏。\n"
        "reason 必須逐字選用 allowed_verification_reasons_json 中的一句，不得自行改寫或補充。這些理由只說明為何需要查核，不是對公司的判斷。不要為了引用而加入 evidence path：一般合約問題使用 []；只有直接核對特定登記欄位才引用該葉節點。\n"
        f"<allowed_verification_reasons_json>{json.dumps(THU_VERIFICATION_REASONS, ensure_ascii=False)}</allowed_verification_reasons_json>\n"
        f"<allowed_evidence_paths_json>{json.dumps(allowed_evidence_paths, ensure_ascii=False, separators=(',', ':'))}</allowed_evidence_paths_json>\n"
        f"<required_limitation_codes_json>{json.dumps(limitation_codes, ensure_ascii=False, separators=(',', ':'))}</required_limitation_codes_json>\n"
        f"{example_description}"
        "可以改善查核問題，但不得改寫其他固定欄位、provenance、限制條件或聲明。\n"
        f"<complete_output_example>\n{example_json}\n</complete_output_example>\n"
        "下方 JSON Schema 是最終欄位契約：\n"
        f"<response_json_schema>{schema_json}</response_json_schema>\n"
        "</thu_json_output_rules>"
    )


def build_thu_response_schema(
    prompt: THUPromptPackage,
) -> dict[str, object]:
    """Build an input-specific THU schema with safe evidence and limitations."""

    _, user_message = prompt.messages
    payload = _extract_verified_input(user_message.content)
    task = getattr(prompt, "task", "company_analysis")
    allowed_paths = _allowed_evidence_paths(payload, task=task)
    example = _build_input_consistent_example(prompt, user_message.content)
    limitation_codes = [
        item["code"]
        for item in example["limitations"]
        if isinstance(item, dict) and isinstance(item.get("code"), str)
    ]
    schema = build_server_owned_manifest_schema(prompt.response_schema)
    fixed_fields = (
        _FIXED_COMPARISON_FIELDS
        if task == "company_comparison_analysis"
        else _FIXED_COMPANY_FIELDS
    )
    for name in fixed_fields:
        field_schema = schema["properties"][name]
        field_schema["enum"] = [example[name]]
        if isinstance(example[name], list):
            field_schema["minItems"] = len(example[name])
            field_schema["maxItems"] = len(example[name])

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        pattern = value.get("pattern")
        if isinstance(pattern, str) and pattern.startswith("^/"):
            value["enum"] = allowed_paths
        for item in value.values():
            visit(item)

    visit(schema)
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        verification_name = (
            "LLMComparisonVerificationItem"
            if task == "company_comparison_analysis" else "LLMVerificationItem"
        )
        definitions[verification_name]["properties"]["reason"]["enum"] = list(THU_VERIFICATION_REASONS)
        schema["properties"]["verification_items"]["maxItems"] = 4
        definition_name = (
            "LLMComparisonLimitation"
            if task == "company_comparison_analysis"
            else "LLMAnalysisLimitation"
        )
        limitation_definition = definitions.get(definition_name)
        if isinstance(limitation_definition, dict):
            properties = limitation_definition.get("properties")
            if isinstance(properties, dict):
                code_schema = properties.get("code")
                if isinstance(code_schema, dict):
                    code_schema["enum"] = limitation_codes
    return apply_thu_narrative_schema_guards(schema)


def validate_thu_output_contract(
    prompt: THUPromptPackage,
    output: dict[str, object],
) -> None:
    """Enforce the advertised THU contract locally, including non-strict modes.

    Never repair or replace a model assertion silently. A mismatch follows the
    normal sanitized diagnostic / bounded retry / fail-closed path.
    """
    example = _build_input_consistent_example(prompt, prompt.messages[1].content)
    fixed_fields = (
        _FIXED_COMPARISON_FIELDS
        if getattr(prompt, "task", "company_analysis") == "company_comparison_analysis"
        else _FIXED_COMPANY_FIELDS
    )
    errors = [
        f"{name} violates THU fixed output contract"
        for name in fixed_fields
        if output.get(name) != example[name]
    ]
    # Canonical validators check structure, evidence, forbidden phrases and
    # numeric claims first. These additional guards apply to the remaining
    # model-authored questions even when the gateway ignores JSON Schema.
    if len(output.get("verification_items", [])) > 4:
        errors.append("verification_items violates THU verification contract")
    for index, item in enumerate(output.get("verification_items", [])):
        if item["reason"] not in THU_VERIFICATION_REASONS:
            errors.append(f"verification_items[{index}].reason violates THU verification contract")
        for name in ("question", "reason"):
            text = unicodedata.normalize("NFKC", item[name])
            text = "".join(c for c in text if unicodedata.category(c) != "Cf")
            if re.search(r"[0-9%]", text) or extract_numeric_claims(text):
                errors.append(f"verification_items[{index}].{name} violates THU narrative policy")
            compact_text = "".join(c for c in text if c.isalnum())
            if re.search(
                r"拳拳|壺仔|(?i:todo|tbd|xxx)|待填|佔位|占位|"
                r"(?:已成立|成立超過|年資為|資本額為|資本為|現況為|目前為停業)",
                compact_text,
            ):
                errors.append(f"verification_items[{index}].{name} contains unsupported narrative assertion")
    if errors:
        raise LLMOutputSafetyError("; ".join(errors))


def apply_thu_narrative_schema_guards(
    schema: dict[str, object],
) -> dict[str, object]:
    """Apply THU-only no-digit patterns to natural-language fields."""

    guarded = json.loads(json.dumps(schema))

    def constrain_strings(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                constrain_strings(item)
            return
        if not isinstance(value, dict):
            return
        if value.get("type") == "string":
            value["pattern"] = _THU_NARRATIVE_NO_DIGITS_PATTERN
        for item in value.values():
            constrain_strings(item)

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        properties = value.get("properties")
        if isinstance(properties, dict):
            for name, child_schema in properties.items():
                if name in _THU_NARRATIVE_FIELDS:
                    constrain_strings(child_schema)
                else:
                    visit(child_schema)
        for key in ("$defs", "definitions"):
            definitions = value.get(key)
            if isinstance(definitions, dict):
                for child_schema in definitions.values():
                    visit(child_schema)
        for key in ("items", "anyOf", "oneOf", "allOf"):
            visit(value.get(key))

    visit(guarded)
    return guarded


def restore_thu_collection_bounds(
    strict_schema: dict[str, object],
    canonical_schema: dict[str, object],
) -> dict[str, object]:
    """Restore canonical array bounds supported by THU's JSON Schema mode."""

    restored = json.loads(json.dumps(strict_schema))

    def merge(target: object, source: object) -> None:
        if isinstance(target, list) and isinstance(source, list):
            for target_item, source_item in zip(target, source, strict=False):
                merge(target_item, source_item)
            return
        if not isinstance(target, dict) or not isinstance(source, dict):
            return
        for constraint in ("minItems", "maxItems"):
            if constraint in source:
                target[constraint] = source[constraint]
        for key, target_item in target.items():
            if key in source:
                merge(target_item, source[key])

    merge(restored, canonical_schema)
    return restored


def _allowed_evidence_paths(
    payload: dict[str, object],
    *,
    task: str,
) -> list[str]:
    allowed_roots = (
        ("/comparison/data", "/comparison/meta")
        if task == "company_comparison_analysis"
        else ("/company", "/bizscore", "/source_meta")
    )
    paths: list[str] = []

    def pointer_token(value: str) -> str:
        return value.replace("~", "~0").replace("/", "~1")

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}/{pointer_token(str(key))}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
            return
        if value is None or path.endswith(("/fetched_at", "/generated_at")):
            return
        if path.startswith(allowed_roots):
            paths.append(path)

    visit(payload, "")
    return paths


def _field_checklist(prompt: THUPromptPackage) -> str:
    task = getattr(prompt, "task", "company_analysis")
    if task == "company_comparison_analysis":
        return (
            "根物件必須且只能包含：schema_version、status、headline、"
            "overall_observation、company_observations、comparison_observations、"
            "verification_items、limitations、provenance、disclaimer。\n"
            "company_observations 每一項必須且只能包含：tax_id、topic、title、"
            "observation、evidence_paths、caveat。\n"
            "comparison_observations 每一項必須且只能包含：topic、title、"
            "observation、evidence_paths、caveat。\n"
            "verification_items 每一項必須且只能包含：priority、question、reason、"
            "related_evidence_paths。\n"
            "limitations 每一項必須且只能包含：code、message、"
            "related_evidence_paths。\n"
            "provenance 必須且只能包含：input_schema_version、prompt_version、"
            "requested_tax_ids、comparison_version、bizscore_version、"
            "benchmark_catalog_version、data_as_of、disclaimer_version、"
            "evidence_paths_used。"
        )
    return (
        "根物件必須且只能包含：schema_version、status、headline、"
        "overall_observation、findings、verification_items、limitations、"
        "provenance、disclaimer。\n"
        "findings 每一項必須且只能包含：topic、title、observation、"
        "evidence_paths、caveat；caveat 不得省略，completed 時不得為 null。\n"
        "verification_items 每一項必須且只能包含：priority、question、reason、"
        "related_evidence_paths。\n"
        "limitations 每一項必須且只能包含：code、message、"
        "related_evidence_paths。\n"
        "provenance 必須且只能包含：input_schema_version、prompt_version、"
        "company_tax_id、bizscore_version、benchmark_catalog_version、data_as_of、"
        "disclaimer_version、evidence_paths_used。"
    )


def _build_input_consistent_example(
    prompt: THUPromptPackage,
    user_content: str,
) -> dict[str, object]:
    payload = _extract_verified_input(user_content)
    task = getattr(prompt, "task", "company_analysis")
    if task == "company_comparison_analysis":
        verified = CompanyComparisonLLMInput.model_validate(payload)
        return _comparison_example(verified)
    verified = CompanyAnalysisLLMInput.model_validate(payload)
    return _company_example(verified)


def _extract_verified_input(user_content: str) -> dict[str, object]:
    opening = "<verified_input_json>"
    closing = "</verified_input_json>"
    start = user_content.find(opening)
    end = user_content.find(closing)
    if start < 0 or end <= start:
        raise ValueError("THU prompt is missing verified_input_json markers.")
    raw_json = user_content[start + len(opening) : end].strip()
    payload = json.loads(raw_json)
    if not isinstance(payload, dict):
        raise ValueError("THU verified_input_json root must be an object.")
    return payload


def _company_example(
    verified: CompanyAnalysisLLMInput,
) -> dict[str, object]:
    no_numeric_score = (
        verified.bizscore.score is None or verified.bizscore.coverage < 0.8
    )
    status = "insufficient_data" if no_numeric_score else "completed"
    source_path = "/source_meta/source"
    findings: list[dict[str, object]] = []
    if status == "completed":
        dimensions = {item.key: item for item in verified.bizscore.dimensions}
        finding_candidates = (
            (
                dimensions["registration_status"].available,
                {
                    "topic": "registration_status",
                    "title": "登記狀態",
                    "observation": "公開登記狀態已有欄位可供核對。",
                    "evidence_paths": ["/company/status/description"],
                    "caveat": "登記狀態不能代表付款、履約或實際營運狀況。",
                },
            ),
            (
                dimensions["company_age"].available
                and verified.company.established_at is not None,
                {
                    "topic": "company_age",
                    "title": "成立年資",
                    "observation": "成立日期已有公開欄位可供核對，年資分數由固定規則計算。",
                    "evidence_paths": ["/company/established_at"],
                    "caveat": "成立時間不能代表獲利、償債或目前實際營運狀況。",
                },
            ),
            (
                dimensions["registered_capital_scale"].available
                and verified.company.capital.registered is not None,
                {
                    "topic": "registered_capital_scale",
                    "title": "登記資本規模",
                    "observation": "登記資本已有公開欄位可供核對，規模分數由固定規則計算。",
                    "evidence_paths": ["/company/capital/registered"],
                    "caveat": "登記資本不能代表可動用現金、營收、獲利或償債能力。",
                },
            ),
            (
                dimensions["registration_change_recency"].available
                and verified.company.last_changed_at is not None,
                {
                    "topic": "registration_change_recency",
                    "title": "登記異動距今",
                    "observation": "最近登記異動日期已有公開欄位可供核對。",
                    "evidence_paths": ["/company/last_changed_at"],
                    "caveat": "登記異動本身不能代表異常或經營不穩。",
                },
            ),
            (
                dimensions["peer_relative_position"].available
                and verified.bizscore.peer_benchmark.dimension.available,
                {
                    "topic": "peer_relative_position",
                    "title": "同業相對位置",
                    "observation": "同業基準已有相對位置欄位可供核對。",
                    "evidence_paths": [
                        "/bizscore/peer_benchmark/dimension/available"
                    ],
                    "caveat": "同業相對位置只比較公開指標，不能代表信用、履約或投資結論。",
                },
            ),
        )
        findings.extend(
            finding for available, finding in finding_candidates if available
        )
        if not findings:
            findings.append(
                {
                    "topic": "data_completeness",
                    "title": "公開資料範圍",
                    "observation": "本次分析使用已提供的公開資料來源欄位。",
                    "evidence_paths": [source_path],
                    "caveat": "公開資料仍不能取代交易文件與實際條件查核。",
                }
            )

    limitations = [
        {
            "code": "public_data_only",
            "message": "內容只整理本次輸入中的公開資料欄位。",
            "related_evidence_paths": [source_path],
        },
        {
            "code": "ai_generated",
            "message": "文字由人工智慧依已驗證輸入產生，仍需另行查核。",
            "related_evidence_paths": [],
        },
    ]
    conditional_limitations = (
        (
            "partial_source_data",
            verified.source_meta.partial or bool(verified.source_meta.warnings),
            "本次來源資料含有缺漏或警示。",
        ),
        (
            "missing_dimension",
            bool(verified.bizscore.missing_dimensions)
            or any(not item.available for item in verified.bizscore.dimensions),
            "部分體質構面沒有可用資料。",
        ),
        (
            "provisional_score",
            verified.bizscore.provisional,
            "本次分數帶有暫定狀態。",
        ),
        (
            "no_numeric_score",
            no_numeric_score,
            "本次未提供數字總分，請先核對資料完整性與登記狀態。",
        ),
        (
            "benchmark_unavailable",
            verified.bizscore.benchmark.snapshot_version is None
            or not verified.bizscore.peer_benchmark.dimension.available,
            "本次沒有可用的同業基準資料。",
        ),
    )
    for code, required, message in conditional_limitations:
        if required:
            limitations.append(
                {
                    "code": code,
                    "message": message,
                    "related_evidence_paths": [],
                }
            )

    return {
        "schema_version": "1.0",
        "status": status,
        "headline": "公開資料重點與合作前查核事項",
        "overall_observation": "本次內容僅整理已提供欄位，實際交易條件仍應另外確認。",
        "findings": findings,
        "verification_items": _verification_example(),
        "limitations": limitations,
        "provenance": {
            "input_schema_version": verified.schema_version,
            "prompt_version": "1.0",
            "company_tax_id": verified.company.tax_id,
            "bizscore_version": verified.bizscore.version,
            "benchmark_catalog_version": (
                verified.bizscore.benchmark.catalog_version
            ),
            "data_as_of": verified.bizscore.as_of.isoformat(),
            "disclaimer_version": verified.disclaimer_version,
            "evidence_paths_used": [],
        },
        "disclaimer": AI_ANALYSIS_DISCLAIMER,
    }


def _verification_example(*, comparison: bool = False) -> list[dict[str, object]]:
    subject = "各公司的" if comparison else "本次交易的"
    questions = (
        f"是否已核對{subject}簽約主體與統一編號？",
        f"是否已確認{subject}簽約授權文件？",
        f"是否已核對{subject}合約付款條件與驗收條款？",
    )
    return [
        {
            "priority": "優先" if index == 0 else "一般",
            "question": question,
            "reason": THU_VERIFICATION_REASONS[index],
            "related_evidence_paths": [],
        }
        for index, question in enumerate(questions)
    ]


def _comparison_example(
    verified: CompanyComparisonLLMInput,
) -> dict[str, object]:
    comparison = verified.comparison
    # Aggregate scope is unavailable if ANY item is unavailable, not necessarily
    # all of them. Use the same four availability checks as the comparison API.
    has_usable_benchmark = any(
        item.bizscore.peer_benchmark.dimension.available
        and item.bizscore.peer_benchmark.peer_index is not None
        and item.bizscore.benchmark.industry_code is not None
        and item.bizscore.benchmark.snapshot_version is not None
        for item in comparison.data.items
    )
    unavailable_benchmark_message = (
        "部分公司缺少可用的同業基準資料；其他公司的同業資料仍可各自查看，無法對全部公司進行同業位置比較。"
        if has_usable_benchmark
        else "所有公司都缺少可用的同業基準資料，無法進行同業位置比較。"
    )
    company_observations: list[dict[str, object]] = []
    for index, item in enumerate(comparison.data.items):
        status_path = (
            f"/comparison/data/items/{index}/company/status/description"
        )
        company_observations.append(
            {
                "tax_id": item.company.tax_id,
                "topic": "registration_status",
                "title": "登記狀態欄位",
                "observation": "公開登記狀態已有欄位可供逐項核對。",
                "evidence_paths": [status_path],
                "caveat": "登記狀態不能代表付款、履約或實際營運狀況。",
            }
        )

    peer_scope_path = "/comparison/data/context/peer_comparison_scope"
    limitations = [
        {
            "code": "public_data_only",
            "message": "內容只整理本次比較中的公開資料欄位。",
            "related_evidence_paths": [],
        },
        {
            "code": "ai_generated",
            "message": "文字由人工智慧依已驗證輸入產生，仍需另行查核。",
            "related_evidence_paths": [],
        },
        {
            "code": "not_ranked",
            "message": "公司順序只沿用使用者選取次序，不形成名次。",
            "related_evidence_paths": [],
        },
    ]
    warning_codes = {warning.code for warning in comparison.meta.warnings}
    conditional_limitations = (
        (
            "partial_source_data",
            comparison.meta.has_partial_source_data
            or "PARTIAL_SOURCE_DATA" in warning_codes
            or "SOURCE_WARNING" in warning_codes,
            "部分來源資料含有缺漏或警示。",
        ),
        (
            "provisional_score",
            comparison.meta.has_provisional_scores,
            "部分分數帶有暫定狀態。",
        ),
        (
            "no_numeric_score",
            comparison.meta.has_unscored_companies,
            "部分公司未提供數字總分，請先核對資料完整性與登記狀態。",
        ),
        (
            "cross_industry_comparison",
            comparison.data.context.peer_comparison_scope
            == "different_industry_snapshots",
            "各公司使用不同產業同業基準，不能形成共同名次。",
        ),
        (
            "benchmark_unavailable",
            comparison.data.context.peer_comparison_scope == "unavailable",
            unavailable_benchmark_message,
        ),
    )
    for code, required, message in conditional_limitations:
        if required:
            limitations.append(
                {
                    "code": code,
                    "message": message,
                    "related_evidence_paths": [],
                }
            )

    return {
        "schema_version": "1.0",
        "status": (
            "insufficient_data"
            if comparison.meta.has_unscored_companies
            else "completed"
        ),
        "headline": "公開登記資料並列觀察",
        "overall_observation": "請並列查看各公司公開欄位，並針對交易條件另行查核。",
        "company_observations": company_observations,
        "comparison_observations": [
            {
                "topic": "peer_scope",
                "title": "同業比較範圍",
                "observation": "各公司的同業基準範圍應分開閱讀。",
                "evidence_paths": [peer_scope_path],
                "caveat": "不同同業基準不能形成共同名次。",
            }
        ],
        "verification_items": _verification_example(comparison=True),
        "limitations": limitations,
        "provenance": {
            "input_schema_version": verified.schema_version,
            "prompt_version": "1.0",
            "requested_tax_ids": comparison.meta.requested_tax_ids,
            "comparison_version": comparison.data.version,
            "bizscore_version": "1.0",
            "benchmark_catalog_version": (
                comparison.data.context.benchmark_catalog_version
            ),
            "data_as_of": comparison.data.context.benchmark_as_of.isoformat(),
            "disclaimer_version": verified.disclaimer_version,
            "evidence_paths_used": [],
        },
        "disclaimer": AI_COMPARISON_DISCLAIMER,
    }

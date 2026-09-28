from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import pytest

from app.services.llm_comparison_prompt import validate_company_comparison_output
from app.services.llm_prompt import (
    LLMOutputSafetyError,
    validate_company_analysis_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIRECTORY = PROJECT_ROOT / "samples" / "llm"


def _read_sample(name: str) -> dict[str, object]:
    payload = json.loads((SAMPLE_DIRECTORY / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _single_input() -> dict[str, object]:
    return _read_sample("company-analysis-input-v1.example.json")


def _single_output() -> dict[str, object]:
    return _read_sample("company-analysis-output-v1.example.json")


def _comparison_input() -> dict[str, object]:
    return _read_sample("company-comparison-input-v1.example.json")


def _comparison_output() -> dict[str, object]:
    return _read_sample("company-comparison-output-v1.example.json")


@pytest.mark.parametrize(
    "unsupported_claim",
    (
        "公開資料證實這家公司已經破產。",
        "公司已取得國際品質認證。",
        "資料顯示這家公司可信，可以簽約。",
    ),
)
def test_single_company_rejects_reproduced_unsupported_claims(
    unsupported_claim: str,
) -> None:
    output = _single_output()
    output["overall_observation"] = unsupported_claim

    with pytest.raises(LLMOutputSafetyError, match="forbidden phrases"):
        validate_company_analysis_output(_single_input(), output)

    for smuggled_claim in (
        "其他資料未提供，但公司已取得國際品質認證。",
        "此資料不代表完整結論而該公司可信。",
        "目前資訊不足以判定全部事項且可以簽約。",
        "資料不能證明付款能力而這家公司更佳。",
    ):
        smuggled_output = _single_output()
        smuggled_output["overall_observation"] = smuggled_claim
        with pytest.raises(LLMOutputSafetyError, match="forbidden phrases"):
            validate_company_analysis_output(_single_input(), smuggled_output)


@pytest.mark.parametrize(
    "unsupported_claim",
    (
        "公開資料證實 A 公司已經破產。",
        "A 公司整體較佳，可優先洽談。",
    ),
)
def test_comparison_rejects_reproduced_unsupported_claims(
    unsupported_claim: str,
) -> None:
    output = _comparison_output()
    output["overall_observation"] = unsupported_claim

    with pytest.raises(LLMOutputSafetyError, match="forbidden phrases"):
        validate_company_comparison_output(_comparison_input(), output)


@pytest.mark.parametrize(
    "numeric_claim",
    (
        "登記資本額為新臺幣九百九十九億元。",
        "登記資本額為 9e99 元。",
    ),
)
def test_single_company_rejects_chinese_and_scientific_numeric_hallucinations(
    numeric_claim: str,
) -> None:
    output = _single_output()
    output["findings"][2]["observation"] = numeric_claim

    with pytest.raises(LLMOutputSafetyError, match="not grounded by evidence"):
        validate_company_analysis_output(_single_input(), output)


def test_single_company_allows_grounded_chinese_unit_conversion() -> None:
    output = _single_output()
    output["findings"][2]["observation"] = (
        "本次資料中的登記資本額為新臺幣四百億元。"
    )

    validated = validate_company_analysis_output(_single_input(), output)

    assert validated.findings[2].observation.endswith("四百億元。")


SingleOutputMutation = Callable[[dict[str, object]], None]


def _set_headline(output: dict[str, object]) -> None:
    output["headline"] = "公開資料顯示 777 家關係企業"


def _set_overall(output: dict[str, object]) -> None:
    output["overall_observation"] = "公開資料顯示 777 家關係企業。"


def _set_finding_title(output: dict[str, object]) -> None:
    output["findings"][0]["title"] = "登記狀態 777"


def _set_finding_observation(output: dict[str, object]) -> None:
    output["findings"][0]["observation"] = "公開資料顯示 777 家關係企業。"


def _set_finding_caveat(output: dict[str, object]) -> None:
    output["findings"][0]["caveat"] = "這不代表已查核 777 份文件。"


def _set_verification_question(output: dict[str, object]) -> None:
    output["verification_items"][0]["question"] = "是否已查核 777 份文件？"


def _set_verification_reason(output: dict[str, object]) -> None:
    output["verification_items"][0]["reason"] = "公開資料未包含 777 份文件。"


def _set_limitation_message(output: dict[str, object]) -> None:
    output["limitations"][1]["message"] = "人工智慧遺漏率為 777%。"


@pytest.mark.parametrize(
    "mutate",
    (
        _set_headline,
        _set_overall,
        _set_finding_title,
        _set_finding_observation,
        _set_finding_caveat,
        _set_verification_question,
        _set_verification_reason,
        _set_limitation_message,
    ),
    ids=(
        "headline",
        "overall-observation",
        "finding-title",
        "finding-observation",
        "finding-caveat",
        "verification-question",
        "verification-reason",
        "limitation-message",
    ),
)
def test_single_company_numeric_grounding_covers_every_visible_text_field(
    mutate: SingleOutputMutation,
) -> None:
    output = _single_output()
    mutate(output)

    with pytest.raises(LLMOutputSafetyError, match="not grounded by evidence"):
        validate_company_analysis_output(_single_input(), output)


@pytest.mark.parametrize(
    "numeric_claim",
    (
        "A 公司有九十九分，仍須另行查核。",
        "A 公司有 9e99 分，仍須另行查核。",
    ),
)
def test_comparison_rejects_chinese_and_scientific_numeric_hallucinations(
    numeric_claim: str,
) -> None:
    output = _comparison_output()
    output["company_observations"][0]["observation"] = numeric_claim

    with pytest.raises(LLMOutputSafetyError, match="not grounded by evidence"):
        validate_company_comparison_output(_comparison_input(), output)


def test_comparison_rejects_null_leaf_evidence() -> None:
    analysis_input = copy.deepcopy(_comparison_input())
    analysis_input["comparison"]["data"]["items"][0]["company"]["status"][
        "description"
    ] = None
    analysis_input["comparison"]["data"]["items"][0]["metrics"][
        "registration_status"
    ] = "01"

    with pytest.raises(LLMOutputSafetyError, match="null leaf"):
        validate_company_comparison_output(analysis_input, _comparison_output())


def test_saved_comparison_examples_pass_production_validator() -> None:
    validated = validate_company_comparison_output(
        _comparison_input(),
        _comparison_output(),
    )

    assert "provisional_score" in {
        limitation.code for limitation in validated.limitations
    }


def test_single_company_allows_certification_verification_question() -> None:
    output = _single_output()
    output["verification_items"][0]["question"] = (
        "是否已提供第三方品質認證文件？"
    )

    validated = validate_company_analysis_output(_single_input(), output)

    assert "認證" in validated.verification_items[0].question


def test_single_company_allows_cautious_decision_limitation() -> None:
    output = _single_output()
    output["findings"][0]["caveat"] = (
        "登記狀態不代表公司可信或可以簽約。"
    )

    validated = validate_company_analysis_output(_single_input(), output)

    assert "不代表" in (validated.findings[0].caveat or "")


def test_single_company_allows_unsupported_stem_quoted_from_evidence() -> None:
    analysis_input = _single_input()
    analysis_input["company"]["status"]["description"] = "破產"
    output = _single_output()
    output["findings"][0]["observation"] = "公開登記狀態為破產。"

    validated = validate_company_analysis_output(analysis_input, output)

    assert validated.findings[0].observation.endswith("破產。")


def test_comparison_allows_certification_verification_question() -> None:
    output = _comparison_output()
    output["verification_items"][0]["question"] = (
        "是否已向各公司索取第三方品質認證文件？"
    )

    validated = validate_company_comparison_output(_comparison_input(), output)

    assert "認證" in validated.verification_items[0].question


def test_comparison_allows_cautious_relative_position_limitation() -> None:
    output = _comparison_output()
    output["overall_observation"] = (
        "本次公開欄位不足以判定哪家公司較佳，仍須另行查核。"
    )

    validated = validate_company_comparison_output(_comparison_input(), output)

    assert "不足以判定" in validated.overall_observation


@pytest.mark.parametrize(
    "summary",
    (
        "本次比較 2 家公司。",
        "本次比較兩家公司。",
    ),
)
def test_comparison_allows_verified_company_count_in_summary(
    summary: str,
) -> None:
    output = _comparison_output()
    output["headline"] = summary

    validated = validate_company_comparison_output(_comparison_input(), output)

    assert validated.headline == summary


@pytest.mark.parametrize(
    "summary",
    (
        "本次比較 3 家公司。",
        "本次比較 2 家公司，但登記資本為 2 億元。",
        "本次比較 2 家公司，資料基準日為 2099 年。",
    ),
)
def test_comparison_summary_rejects_unverified_or_non_count_numbers(
    summary: str,
) -> None:
    output = _comparison_output()
    output["headline"] = summary

    with pytest.raises(LLMOutputSafetyError, match="not grounded by evidence"):
        validate_company_comparison_output(_comparison_input(), output)

    for signed_count in (
        "本次比較 -2 家公司。",
        "本次比較 −2 家公司。",
        "本次比較負2家公司。",
        "本次比較 - 2 家公司。",
        "本次比較 − 2 家公司。",
        "本次比較負 2 家公司。",
        "本次比較 + 2 家公司。",
        "本次比較負數 2 家公司。",
        "本次比較負數的 2 家公司。",
        "本次比較負值 2 家公司。",
        "本次比較 -​2 家公司。",
        "本次比較 ➖ 2 家公司。",
        "本次比較不到 2 家公司。",
        "本次比較 2 家公司嗎？",
        "本次比較 2 家公司並非正確數量。",
        "本次比較 2 家公司不是事實。",
    ):
        signed_output = _comparison_output()
        signed_output["headline"] = signed_count
        with pytest.raises(LLMOutputSafetyError, match="not grounded by evidence"):
            validate_company_comparison_output(_comparison_input(), signed_output)

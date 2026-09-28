from pathlib import Path

import pytest

from scripts.run_ai_safety_case_set import (
    DEFAULT_CASE_SET,
    evaluate_case,
    load_case_set,
)


CASE_SET_PATH = Path(DEFAULT_CASE_SET)
CASE_SET = load_case_set(CASE_SET_PATH)
CASES = CASE_SET["cases"]


def test_formal_case_set_has_required_size_controls_and_unique_ids() -> None:
    assert CASE_SET["case_set_version"] == "1.0"
    assert CASE_SET["live_model_required"] is False
    assert len(CASES) == 38
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert sum(case["expected"] == "accepted" for case in CASES) == 6
    assert sum(case["expected"] == "rejected" for case in CASES) == 32
    assert {
        "forbidden_claim",
        "numeric_hallucination",
        "evidence_grounding",
        "state_boundary",
        "prompt_injection",
        "provenance",
    }.issubset({case["category"] for case in CASES})


@pytest.mark.anyio
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_ai_hallucination_and_boundary_case(case: dict) -> None:
    result = await evaluate_case(CASE_SET, case)

    assert result["passed"] is True, result
    assert result["actual"] == case["expected"]
    if case["expected"] == "rejected":
        assert result["service_outcome"] == "AI_ANALYSIS_INVALID_OUTPUT"
        assert result["fallback"] == "502 AI_ANALYSIS_INVALID_OUTPUT"
    elif case["mode"] == "output":
        assert result["service_outcome"] == "accepted"

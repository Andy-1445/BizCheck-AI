"""GCIS status-source consistency and scoring gates (official company table 7)."""
from datetime import date
import json
from pathlib import Path

import pytest

from app.services.bizscore import score_registration_status, STATUS_CONFLICT_DESCRIPTION
from app.services.company_normalizer import normalize_company_data

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "gcis"


def records():
    return [json.loads((SAMPLES / filename).read_text(encoding="utf-8"))[0] for filename in (
        "company-basic-20828393.json", "company-business-20828393.json",
    )]


@pytest.mark.parametrize("code,description", [
    ("01", "核准設立，但已命令解散"), ("02", "停業"), ("03", "解散"),
    ("06", "列入廢止中"), ("08", "破產"), ("01", "核准設立（已解散）"),
])
def test_contradictory_or_wrong_code_system_cannot_be_scored(code, description):
    result = score_registration_status(code, description)
    assert result.dimension.score is None
    assert result.dimension.available is False
    assert result.total_score_eligible is False
    assert result.status_cap is None
    assert result.outcome == "insufficient_data"
    assert "GCIS_STATUS_CONFLICT" in result.dimension.warnings[0]


@pytest.mark.parametrize("code,description", [
    ("01", " 核准設立 "), ("02", "核准設立, 但已命令解散"),
    ("22", "破產程序終結（終止）"), ("23", "破產程序終結（終止）清算中"),
])
def test_display_punctuation_does_not_change_registration_meaning(code, description):
    result = score_registration_status(code, description)
    assert result.dimension.available is True
    assert result.total_score_eligible is (code == "01")


@pytest.mark.parametrize("a1code,a1desc,a3code,a3desc", [
    (None, "核准設立，但已命令解散", "01", "核准設立"),
    (None, "核准設立", "02", "核准設立，但已命令解散"),
    ("02", "核准設立", "01", "核准設立"),
    (None, "核准設立", "01", "已解散"),
    (None, "停業", "02", "停業"),
])
def test_conflicting_endpoints_withhold_status_code(a1code, a1desc, a3code, a3desc):
    basic, business = records()
    basic.update(Company_Status=a1code, Company_Status_Desc=a1desc)
    business.update(Company_Status=a3code, Company_Status_Desc=a3desc)
    result = normalize_company_data(basic, business, as_of=date(2026, 9, 22))
    assert result.partial is True
    assert result.data.status.code is None
    assert result.data.status.description == STATUS_CONFLICT_DESCRIPTION
    assert any("GCIS_STATUS_CONFLICT" in warning for warning in result.warnings)
    assert result.data.capital.registered == 40_000_000_000
    assert not score_registration_status(result.data.status.code, result.data.status.description).total_score_eligible


def test_matching_endpoints_retain_official_commanded_dissolution():
    basic, business = records()
    basic["Company_Status_Desc"] = "核准設立，但已命令解散"
    business.update(Company_Status="02", Company_Status_Desc="核准設立,但已命令解散")
    result = normalize_company_data(basic, business)
    assert result.partial is False
    assert result.data.status.code == "02"
    score = score_registration_status(result.data.status.code, result.data.status.description)
    assert score.dimension.score == 0
    assert score.total_score_eligible is False


def test_missing_a3_never_infers_01_from_description():
    basic, _ = records()
    result = normalize_company_data(basic, None)
    assert result.data.status.code is None
    assert result.partial is True


def test_mismatched_company_record_cannot_supply_status_or_business_items():
    basic, business = records()
    business["Business_Accounting_NO"] = "22099131"
    result = normalize_company_data(basic, business)
    assert result.partial is True
    assert result.data.status.code is None
    assert result.data.business_items == []

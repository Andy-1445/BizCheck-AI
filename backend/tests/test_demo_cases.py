import copy
import json

import pytest

from scripts.verify_demo_cases import SNAPSHOTS, verify_all


def snapshot():
    return json.loads(SNAPSHOTS.read_text(encoding="utf-8"))


def test_three_demo_snapshots_and_controlled_missing_data_replay():
    assert len(verify_all(snapshot())) == 4


@pytest.mark.parametrize("change", ["score", "identity", "missing_case", "version"])
def test_changed_demo_evidence_is_rejected(change):
    bundle = copy.deepcopy(snapshot())
    if change == "score":
        bundle["responses"]["DEMO-01"]["data"]["bizscore"]["score"] = 99
    elif change == "identity":
        bundle["responses"]["DEMO-01"]["data"]["company"]["tax_id"] = "83510057"
    elif change == "missing_case":
        del bundle["responses"]["DEMO-03"]
    else:
        bundle["manifest_version"] = "wrong-version"
    with pytest.raises(AssertionError):
        verify_all(bundle)

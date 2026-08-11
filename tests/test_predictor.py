import copy
import json
from pathlib import Path

import pytest

from predictor import CampaignPredictor, InputValidationError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = json.loads((ROOT / "sample_payloads" / "sales.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictor():
    return CampaignPredictor(ROOT / "models")


def test_valid_prediction_uses_v5(predictor):
    result = predictor.predict(SAMPLE)
    assert 0.0 <= result.probability <= 1.0
    assert result.objective == SAMPLE["Campaign_Objective"]
    assert result.model_version == "V5"
    assert result.freeze_manifest_sha256 == "5f3874c8a663d329d7e19876e924f44ee09dd0683ac72dc5a017811e05125792"


def test_forbidden_post_launch_field_is_rejected(predictor):
    bad = copy.deepcopy(SAMPLE)
    bad["ROAS"] = 9.9
    with pytest.raises(InputValidationError) as exc:
        predictor.predict(bad)
    assert "ROAS" in str(exc.value)


def test_out_of_range_score_is_rejected(predictor):
    bad = copy.deepcopy(SAMPLE)
    bad["Creative_Quality_Score"] = 150
    with pytest.raises(InputValidationError):
        predictor.predict(bad)


def test_unsupported_platform_is_rejected(predictor):
    bad = copy.deepcopy(SAMPLE)
    bad["Platform"] = "Unknown Network"
    with pytest.raises(InputValidationError):
        predictor.predict(bad)


def test_missing_required_field_is_rejected(predictor):
    bad = copy.deepcopy(SAMPLE)
    bad.pop("Budget_SAR", None)
    with pytest.raises(InputValidationError):
        predictor.predict(bad)


def test_model_features_never_include_forbidden_fields(predictor):
    for bundle in predictor.models.values():
        assert not (set(bundle["features"]) & predictor.forbidden)


def test_all_six_objectives_from_v5_samples(predictor):
    expected = set(predictor.models)
    seen = set()
    for path in (ROOT / "sample_payloads").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = predictor.predict(payload)
        seen.add(result.objective)
        assert 0 <= result.probability <= 1
    assert seen == expected


def test_schema_covers_every_model_feature(predictor):
    derived = {"Start_Month", "Start_Quarter", "Start_DayOfWeek", "Budget_Per_1000_Audience_SAR"}
    for bundle in predictor.models.values():
        missing = set(bundle["features"]) - predictor.allowed_input_fields - derived
        assert not missing, f"Model features missing from app schema: {missing}"

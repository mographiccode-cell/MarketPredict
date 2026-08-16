from pathlib import Path

from assessment import apply_auto_assessment
from batch_routes import evaluate_campaign_rows, parse_campaign_csv
from predictor import CampaignPredictor

ROOT = Path(__file__).resolve().parents[1]


def test_six_campaign_csv_is_valid_and_predictable():
    data = (ROOT / "sample_payloads" / "campaign_validation_6.csv").read_bytes()
    rows = parse_campaign_csv(data)
    assert len(rows) == 6
    assert sum(r["source"].get("Actual_Result") == "Success" for r in rows) == 3
    assert sum(r["source"].get("Actual_Result") == "Failure" for r in rows) == 3

    predictor = CampaignPredictor()
    results, summary = evaluate_campaign_rows(rows, lambda: predictor, apply_auto_assessment)
    assert summary["total"] == 6
    assert summary["valid"] == 6
    assert summary["labeled"] == 6
    assert all(r["probability"] is not None and 0 <= r["probability"] <= 1 for r in results)

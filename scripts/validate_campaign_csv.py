from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assessment import apply_auto_assessment
from batch_routes import evaluate_campaign_rows, parse_campaign_csv
from predictor import CampaignPredictor

INPUT = ROOT / "sample_payloads" / "campaign_validation_6.csv"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

rows = parse_campaign_csv(INPUT.read_bytes())
predictor = CampaignPredictor()
results, summary = evaluate_campaign_rows(rows, lambda: predictor, apply_auto_assessment)

result_csv = OUT_DIR / "campaign_validation_results.csv"
with result_csv.open("w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "Campaign_Name", "Objective", "Actual_Result", "Predicted_Result",
        "Success_Probability", "Threshold", "Confidence", "Correct", "Error"
    ])
    for r in results:
        writer.writerow([
            r["campaign_name"], r["objective"], r["actual_text"], r["predicted_text"],
            "" if r["probability"] is None else round(r["probability"], 6),
            "" if r["threshold"] is None else round(r["threshold"], 6),
            r["confidence_label"], "" if r["correct"] is None else int(r["correct"]), r["error"],
        ])

payload = {"summary": summary, "results": results}
(OUT_DIR / "campaign_validation_results.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("=== MARKETPREDICT SIX-CAMPAIGN VALIDATION ===")
for i, r in enumerate(results, 1):
    probability = "ERROR" if r["probability"] is None else f"{r['probability']*100:.2f}%"
    verdict = "CORRECT" if r["correct"] is True else "WRONG" if r["correct"] is False else "N/A"
    print(f"{i}. {r['campaign_name']} | {r['objective']} | actual={r['actual_text']} | predicted={r['predicted_text']} | p={probability} | {verdict}")
print("SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))

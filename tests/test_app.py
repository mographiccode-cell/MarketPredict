import json
import os
import re
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = ROOT / "test_app.db"
os.environ["DB_PATH"] = str(TEST_DB)
if TEST_DB.exists():
    TEST_DB.unlink()

from app import app  # noqa: E402

client = TestClient(app)
SAMPLE = json.loads((ROOT / "sample_payloads" / "sales.json").read_text(encoding="utf-8"))


def csrf(html: str) -> str:
    match = re.search(r'name="_csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def demo_login() -> None:
    login = client.get("/login")
    assert login.status_code == 200
    response = client.post(
        "/demo-login",
        data={"_csrf": csrf(login.text)},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_public_pages_and_model_health():
    assert client.get("/").status_code == 200
    assert client.get("/model").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["model_version"] == "V5"
    assert body["objectives"] == 6


def test_demo_login_evidence_form_prediction_and_history():
    demo_login()

    form = client.get("/predict")
    assert form.status_code == 200
    assert "evidence__Offer_Strength_Score__0" in form.text
    assert "أدلة الجاهزية" in form.text

    prediction = client.post("/api/predict", json=SAMPLE)
    assert prediction.status_code == 200, prediction.text
    result = prediction.json()
    assert result["objective"] == "Sales"
    assert result["model_version"] == "V5"
    assert 0 <= result["probability"] <= 1
    assert result["prediction_id"] >= 1

    history = client.get("/history")
    assert history.status_code == 200
    assert "السجل" in history.text


def test_evidence_overrides_numeric_score_server_side():
    demo_login()
    payload = dict(SAMPLE)
    payload["Creative_Quality_Score"] = 100
    payload.update({
        "evidence__Creative_Quality_Score__0": "1",
        "evidence__Creative_Quality_Score__1": "0",
        "evidence__Creative_Quality_Score__2": "0",
    })
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["probability"] <= 1


def test_batch_csv_upload_renders_six_real_predictions():
    demo_login()
    page = client.get("/batch")
    assert page.status_code == 200
    assert "تحليل ملف الحملات" in page.text
    token = csrf(page.text)
    csv_bytes = (ROOT / "sample_payloads" / "campaign_validation_6.csv").read_bytes()
    response = client.post(
        "/batch",
        data={"_csrf": token},
        files={"file": ("campaign_validation_6.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert "100.0%" in response.text
    assert "97.2%" in response.text
    assert "96.7%" in response.text
    assert "93.5%" in response.text
    assert "13.8%" in response.text
    assert "8.1%" in response.text
    assert "4.2%" in response.text
    assert response.text.count("صحيح ✓") == 6

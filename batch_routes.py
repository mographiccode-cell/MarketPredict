from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import HTMLResponse, StreamingResponse

META_COLUMNS = {"Campaign_Name", "Actual_Result", "Actual_Notes"}
MAX_BATCH_ROWS = 200
MAX_BATCH_BYTES = 2 * 1024 * 1024


def parse_actual_result(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "success", "successful", "نجاح", "ناجحة", "ناجح"}:
        return True
    if text in {"0", "failure", "failed", "fail", "فشل", "فاشلة", "فاشل"}:
        return False
    raise ValueError(f"Actual_Result غير معروف: {value}. استخدم Success أو Failure.")


def parse_campaign_csv(data: bytes) -> list[dict[str, Any]]:
    if not data:
        raise ValueError("الملف فارغ.")
    if len(data) > MAX_BATCH_BYTES:
        raise ValueError("حجم الملف أكبر من 2MB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("يجب حفظ CSV بترميز UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("لم يتم العثور على رؤوس الأعمدة في CSV.")
    required_headers = {"Campaign_Objective", "Start_Date", "Budget_SAR", "Platform"}
    missing = sorted(required_headers - set(reader.fieldnames))
    if missing:
        raise ValueError("أعمدة أساسية مفقودة: " + ", ".join(missing))

    rows: list[dict[str, Any]] = []
    for row_number, source in enumerate(reader, start=2):
        if not any(str(v or "").strip() for v in source.values()):
            continue
        if len(rows) >= MAX_BATCH_ROWS:
            raise ValueError(f"الحد الأقصى {MAX_BATCH_ROWS} حملة في الملف الواحد.")
        clean = {str(k).strip(): str(v).strip() for k, v in source.items() if k is not None and v is not None}
        rows.append({"row_number": row_number, "source": clean})
    if not rows:
        raise ValueError("لا توجد حملات قابلة للقراءة داخل الملف.")
    return rows


def evaluate_campaign_rows(
    rows: list[dict[str, Any]],
    get_predictor: Callable[[], Any],
    apply_auto_assessment: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any], list[str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictor = get_predictor()
    results: list[dict[str, Any]] = []

    for item in rows:
        source = item["source"]
        name = source.get("Campaign_Name") or f"Campaign {item['row_number'] - 1}"
        actual_raw = source.get("Actual_Result", "")
        notes = source.get("Actual_Notes", "")
        model_input = {k: v for k, v in source.items() if k not in META_COLUMNS and v not in (None, "")}
        try:
            actual = parse_actual_result(actual_raw)
            payload, _, assessment_warnings = apply_auto_assessment(model_input)
            prediction = predictor.predict(payload).as_dict()
            all_warnings = assessment_warnings + prediction.get("warnings", [])
            predicted = bool(prediction["predicted_success"])
            correct = None if actual is None else predicted == actual
            results.append({
                "row_number": item["row_number"],
                "campaign_name": name,
                "notes": notes,
                "objective": prediction["objective"],
                "actual": actual,
                "actual_text": "نجاح" if actual is True else "فشل" if actual is False else "غير محدد",
                "predicted_success": predicted,
                "predicted_text": "نجاح مرجح" if predicted else "فشل مرجح",
                "probability": float(prediction["probability"]),
                "threshold": float(prediction["threshold"]),
                "confidence_label": prediction.get("confidence_label", ""),
                "decision_strength": prediction.get("decision_strength", ""),
                "correct": correct,
                "warnings": all_warnings,
                "recommendations": prediction.get("recommendations", []),
                "error": "",
            })
        except Exception as exc:
            details = getattr(exc, "errors", None)
            message = " | ".join(details) if details else str(exc)
            results.append({
                "row_number": item["row_number"],
                "campaign_name": name,
                "notes": notes,
                "objective": source.get("Campaign_Objective", ""),
                "actual": None,
                "actual_text": actual_raw or "غير محدد",
                "predicted_success": None,
                "predicted_text": "تعذر التنبؤ",
                "probability": None,
                "threshold": None,
                "confidence_label": "-",
                "decision_strength": "-",
                "correct": None,
                "warnings": [],
                "recommendations": [],
                "error": message,
            })

    valid = [r for r in results if not r["error"]]
    labeled = [r for r in valid if r["actual"] is not None]
    correct_count = sum(1 for r in labeled if r["correct"])
    summary = {
        "total": len(results),
        "valid": len(valid),
        "errors": len(results) - len(valid),
        "predicted_successes": sum(1 for r in valid if r["predicted_success"]),
        "predicted_failures": sum(1 for r in valid if not r["predicted_success"]),
        "labeled": len(labeled),
        "correct": correct_count,
        "accuracy": (correct_count / len(labeled)) if labeled else None,
    }
    return results, summary


def register_batch_routes(
    app: Any,
    *,
    templates: Any,
    require_login: Callable[[Request], tuple[Any, Any]],
    context: Callable[..., dict[str, Any]],
    check_csrf: Callable[[Request, str | None], None],
    rate_limit: Callable[..., None],
    get_predictor: Callable[[], Any],
    apply_auto_assessment: Callable[..., Any],
    base_dir: Path,
) -> None:
    @app.api_route("/batch", methods=["GET", "POST"], response_class=HTMLResponse)
    async def batch_page(request: Request):
        user, redirect = require_login(request)
        if redirect:
            return redirect
        if request.method == "GET":
            return templates.TemplateResponse(request, "batch.html", context(request, results=[], summary=None, upload_error=""))

        rate_limit(f"batch:{user['id']}", limit=8, window=60)
        form = await request.form()
        check_csrf(request, form.get("_csrf"))
        upload = form.get("file")
        filename = str(getattr(upload, "filename", "") or "")
        if not filename.lower().endswith(".csv"):
            return templates.TemplateResponse(
                request, "batch.html",
                context(request, results=[], summary=None, upload_error="ارفع ملف CSV فقط."),
                status_code=400,
            )
        try:
            data = await upload.read()
            rows = parse_campaign_csv(data)
            results, summary = evaluate_campaign_rows(rows, get_predictor, apply_auto_assessment)
        except ValueError as exc:
            return templates.TemplateResponse(
                request, "batch.html",
                context(request, results=[], summary=None, upload_error=str(exc)),
                status_code=400,
            )
        return templates.TemplateResponse(
            request, "batch.html",
            context(request, results=results, summary=summary, upload_error="", uploaded_filename=filename),
        )

    @app.get("/batch/sample.csv")
    async def batch_sample(request: Request):
        _, redirect = require_login(request)
        if redirect:
            return redirect
        path = base_dir / "sample_payloads" / "campaign_validation_6.csv"
        data = path.read_bytes()
        return StreamingResponse(
            iter([data]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=marketpredict_validation_6_campaigns.csv"},
        )

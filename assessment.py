from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RUBRICS = json.loads((MODEL_DIR / "assessment_rubrics.json").read_text(encoding="utf-8"))
BENCHMARKS = json.loads((MODEL_DIR / "v5_benchmarks.json").read_text(encoding="utf-8"))
SCORE_FIELDS = set(RUBRICS)
EVIDENCE_PREFIX = "evidence__"

DEFAULT_LEVELS = [
    {"value": "weak", "label": "ضعيف", "score": 30, "desc": "غير جاهز ويحتاج معالجة واضحة قبل الإطلاق."},
    {"value": "partial", "label": "جزئي", "score": 55, "desc": "متوفر جزئيًا لكن توجد فجوات مؤثرة."},
    {"value": "ready", "label": "جاهز", "score": 75, "desc": "جاهزية جيدة مع تحسينات محدودة."},
    {"value": "excellent", "label": "ممتاز", "score": 90, "desc": "جاهزية قوية ومكتملة قبل الإطلاق."},
]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def budget_adequacy_score(budget_sar: float, audience_size: float, objective: str) -> tuple[float, dict[str, Any]]:
    if budget_sar <= 0 or audience_size <= 0:
        return 5.0, {"method": "invalid-input-fallback"}
    per_1000 = budget_sar / audience_size * 1000.0
    need = float(BENCHMARKS["budget_need_per_1000_sar"].get(objective, 18.0))
    score = max(5.0, min(99.0, (per_1000 / need) * 50.0))
    return round(score, 1), {
        "method": "V5 deterministic budget adequacy formula",
        "budget_per_1000_sar": round(per_1000, 3),
        "objective_reference_need_per_1000_sar": need,
    }


def rubric_score(field: str, level: str) -> tuple[float, dict[str, Any]]:
    cfg = RUBRICS.get(field)
    if not cfg:
        raise ValueError(f"Unknown rubric field: {field}")
    for item in cfg.get("levels", DEFAULT_LEVELS):
        if item["value"] == level:
            return float(item["score"]), {
                "method": "legacy standardized pre-launch rubric",
                "level": level,
                "label": item["label"],
                "description": item["desc"],
            }
    raise ValueError(f"Unsupported rubric level for {field}: {level}")


def _evidence_value(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "yes", "true", "on"}:
        return 1
    if normalized in {"0", "no", "false", "off"}:
        return 0
    raise ValueError("قيمة دليل الجاهزية غير صالحة. اختر نعم أو لا لكل بند.")


def evidence_score(field: str, payload: dict[str, Any]) -> tuple[float, dict[str, Any]] | None:
    """Derive a 30..90 score from observable yes/no evidence.

    The 30..90 scale intentionally matches the historical rubric range used by
    the frozen V5 model, while replacing subjective labels with reproducible
    pre-launch facts. All checks are required once one check for a field is sent.
    """
    cfg = RUBRICS.get(field) or {}
    checks = list(cfg.get("checks") or [])
    if not checks:
        return None

    names = [f"{EVIDENCE_PREFIX}{field}__{i}" for i in range(len(checks))]
    present = [name for name in names if name in payload]
    if not present:
        return None
    if len(present) != len(names):
        raise ValueError(f"أكمل جميع أدلة: {cfg.get('question', field)}.")

    answers = [_evidence_value(payload[name]) for name in names]
    yes_count = sum(answers)
    # 0/3 -> 30, 1/3 -> 50, 2/3 -> 70, 3/3 -> 90.
    score = round(30.0 + 60.0 * (yes_count / len(answers)), 1)
    evidence = [
        {"check": checks[i], "verified": bool(answers[i])}
        for i in range(len(checks))
    ]
    return score, {
        "method": "evidence-based pre-launch checklist",
        "verified_checks": yes_count,
        "total_checks": len(checks),
        "evidence": evidence,
    }


def apply_auto_assessment(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    out = dict(payload)
    provenance: dict[str, Any] = {}
    warnings: list[str] = []

    objective = str(out.get("Campaign_Objective", ""))
    budget = _float(out.get("Budget_SAR"))
    audience = _float(out.get("Estimated_Audience_Size"))
    if budget is not None and audience is not None and objective:
        score, meta = budget_adequacy_score(budget, audience, objective)
        out["Budget_Adequacy_Score"] = score
        provenance["Budget_Adequacy_Score"] = meta

    for field in SCORE_FIELDS:
        # Evidence always wins over a browser-supplied numeric score. This is an
        # anti-tampering rule: the server recomputes the value from the facts.
        evidence_result = evidence_score(field, out)
        if evidence_result is not None:
            score, meta = evidence_result
            out[field] = score
            provenance[field] = meta
        else:
            # Backward compatibility for older API clients and saved forms.
            rubric_key = f"rubric_{field}"
            if rubric_key in out and out.get(rubric_key) not in (None, ""):
                score, meta = rubric_score(field, str(out[rubric_key]))
                out[field] = score
                provenance[field] = meta
        out.pop(f"rubric_{field}", None)

    # Never pass UI evidence fields into the ML feature vector.
    for key in list(out):
        if key.startswith(EVIDENCE_PREFIX):
            out.pop(key, None)

    if out.get("Brand_Awareness_Score") in (None, ""):
        maturity_map = {"New": 32.0, "Emerging": 55.0, "Established": 80.0}
        maturity = str(out.get("Brand_Maturity", ""))
        if maturity in maturity_map:
            out["Brand_Awareness_Score"] = maturity_map[maturity]
            provenance["Brand_Awareness_Score"] = {
                "method": "brand maturity fallback",
                "brand_maturity": maturity,
            }

    mode = str(out.pop("score_mode", "auto"))
    if mode == "manual":
        warnings.append("تم استخدام عميل قديم يدعم الإدخال اليدوي. واجهة الويب الحالية تحسب درجات الجاهزية من أدلة قابلة للتحقق.")

    for key in list(out):
        if key.startswith("ui_"):
            out.pop(key, None)
    return out, provenance, warnings


def score_label(score: float) -> str:
    if score >= 85:
        return "جاهزية مرتفعة"
    if score >= 70:
        return "جاهزية جيدة"
    if score >= 50:
        return "جاهزية متوسطة"
    return "فجوات واضحة"


def assessment_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "Creative_Quality_Score", "Offer_Strength_Score", "Landing_Page_Quality_Score",
        "Tracking_Readiness_Score", "Mobile_Readiness_Score", "Arabic_Localization_Score",
        "Scheduling_Alignment_Score", "Brand_Awareness_Score", "Auction_Competition_Score",
        "Content_Audience_Fit_Score", "Trend_Relevance_Score", "Trust_Score",
        "Budget_Adequacy_Score",
    ]
    result = []
    for field in fields:
        value = _float(payload.get(field))
        if value is not None:
            result.append({"field": field, "score": round(value, 1), "label": score_label(value)})
    return result

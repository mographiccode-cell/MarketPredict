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

def _float(value: Any) -> float | None:
    if value in (None, ""): return None
    try: v=float(value)
    except (TypeError,ValueError): return None
    return v if math.isfinite(v) else None

def budget_adequacy_score(budget_sar: float,audience_size: float,objective: str)->tuple[float,dict[str,Any]]:
    if budget_sar<=0 or audience_size<=0: return 5.0,{"method":"invalid-input-fallback"}
    per_1000=budget_sar/audience_size*1000.0
    need=float(BENCHMARKS["budget_need_per_1000_sar"].get(objective,18.0))
    score=max(5.0,min(99.0,(per_1000/need)*50.0))
    return round(score,1),{"method":"V5 deterministic budget adequacy formula","budget_per_1000_sar":round(per_1000,3),"objective_reference_need_per_1000_sar":need}

def rubric_score(field:str,level:str)->tuple[float,dict[str,Any]]:
    cfg=RUBRICS.get(field)
    if not cfg: raise ValueError(f"Unknown rubric field: {field}")
    for item in cfg["levels"]:
        if item["value"]==level:
            return float(item["score"]),{"method":"standardized pre-launch rubric","level":level,"label":item["label"],"description":item["desc"]}
    raise ValueError(f"Unsupported rubric level for {field}: {level}")

def apply_auto_assessment(payload:dict[str,Any])->tuple[dict[str,Any],dict[str,Any],list[str]]:
    out=dict(payload);provenance={};warnings=[]
    objective=str(out.get("Campaign_Objective",""));budget=_float(out.get("Budget_SAR"));audience=_float(out.get("Estimated_Audience_Size"))
    if budget is not None and audience is not None and objective:
        score,meta=budget_adequacy_score(budget,audience,objective);out["Budget_Adequacy_Score"]=score;provenance["Budget_Adequacy_Score"]=meta
    for field in SCORE_FIELDS:
        rubric_key=f"rubric_{field}"
        if rubric_key in out and out.get(rubric_key) not in (None,""):
            score,meta=rubric_score(field,str(out[rubric_key]));out[field]=score;provenance[field]=meta
        out.pop(rubric_key,None)
    if out.get("Brand_Awareness_Score") in (None,""):
        maturity_map={"New":32.0,"Emerging":55.0,"Established":80.0};maturity=str(out.get("Brand_Maturity",""))
        if maturity in maturity_map:
            out["Brand_Awareness_Score"]=maturity_map[maturity];provenance["Brand_Awareness_Score"]={"method":"brand maturity fallback","brand_maturity":maturity}
    mode=str(out.pop("score_mode","auto"))
    if mode=="manual": warnings.append("تم استخدام درجات يدوية لبعض عوامل التخطيط؛ يفضل وضع Auto Assessment للحصول على قياس موحد قابل للتكرار.")
    for key in list(out):
        if key.startswith("ui_"): out.pop(key,None)
    return out,provenance,warnings

def score_label(score:float)->str:
    if score>=85:return "ممتاز"
    if score>=70:return "جيد"
    if score>=50:return "متوسط"
    return "يحتاج تحسين"

def assessment_summary(payload:dict[str,Any])->list[dict[str,Any]]:
    fields=["Creative_Quality_Score","Offer_Strength_Score","Landing_Page_Quality_Score","Tracking_Readiness_Score","Mobile_Readiness_Score","Arabic_Localization_Score","Scheduling_Alignment_Score","Brand_Awareness_Score","Auction_Competition_Score","Content_Audience_Fit_Score","Trend_Relevance_Score","Trust_Score","Budget_Adequacy_Score"]
    out=[]
    for field in fields:
        value=_float(payload.get(field))
        if value is not None: out.append({"field":field,"score":round(value,1),"label":score_label(value)})
    return out

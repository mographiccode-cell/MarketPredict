from __future__ import annotations

import json
import math
import warnings as py_warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

class InputValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))

@dataclass
class PredictionResult:
    objective: str
    probability: float
    predicted_success: bool
    threshold: float
    decision_margin: float
    decision_strength: str
    confidence_label: str
    model_roc_auc: float | None
    model_balanced_accuracy: float | None
    p_lgb: float
    p_cat: float
    ensemble_agreement: float
    warnings: list[str]
    recommendations: list[str]
    features_used: int
    model_version: str
    freeze_manifest_sha256: str
    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

class CampaignPredictor:
    SCORE_FIELDS = {"Creative_Quality_Score","Offer_Strength_Score","Landing_Page_Quality_Score","Tracking_Readiness_Score","Mobile_Readiness_Score","Arabic_Localization_Score","Scheduling_Alignment_Score","Brand_Awareness_Score","Auction_Competition_Score","Content_Audience_Fit_Score","Trend_Relevance_Score","Trust_Score","Budget_Adequacy_Score","Price_Competitiveness_Score","Checkout_Ease_Score","Search_Intent_Score","Headline_Relevance_Score","Creative_Memorability_Score","Lead_Form_Ease_Score","Lead_Magnet_Strength_Score","Sales_Cycle_Simplicity_Score","Hook_Strength_Score","Native_Format_Fit_Score","Community_Affinity_Score","App_Store_Page_Quality_Score","Device_Compatibility_Score"}
    HISTORY_NUMERIC = {"Brand_Prior_Campaigns","Brand_Prior_Success_Rate","Brand_Prior_CTR","Brand_Prior_CVR","Brand_Platform_Prior_Campaigns","Brand_Platform_Prior_Success_Rate","Brand_Platform_Prior_CTR","Brand_Platform_Prior_CVR","Brand_Objective_Prior_Campaigns","Brand_Objective_Prior_Success_Rate","Brand_Objective_Prior_CTR","Brand_Objective_Prior_CVR","Days_Since_Last_Campaign","Brand_Recent5_Success_Rate"}
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.suite = joblib.load(self.model_dir / "model_suite.joblib")
        self.safety = json.loads((self.model_dir / "inference_safety_schema.json").read_text(encoding="utf-8"))
        self.schema = json.loads((self.model_dir / "app_schema.json").read_text(encoding="utf-8"))
        self.lockbox = json.loads((self.model_dir / "lockbox_results.json").read_text(encoding="utf-8"))
        self.models = self.suite["models"]
        self.objectives = set(self.models)
        self.forbidden = set(self.safety["forbidden_in_prediction"])
        self.required = set(self.schema["required_fields"])
        self.history_fields = set(self.schema["optional_history_fields"])
        self.objective_required = {k:set(v) for k,v in self.schema["objective_specific_required"].items()}
        self.allowed_input_fields = self.required | self.history_fields | {"Campaign_Objective"} | set().union(*self.objective_required.values())
        if self.suite.get("dataset_version") != "V5": raise RuntimeError("Expected frozen V5 suite")
        if set(self.models) != set(self.schema["objectives"]): raise RuntimeError("App schema does not match frozen suite")
    @staticmethod
    def _pred_cat(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        Z=X.copy()
        for col in bundle["num"]: Z[col]=pd.to_numeric(Z[col],errors="coerce").fillna(bundle["med"][col]).astype(float)
        for col in bundle["cat"]: Z[col]=Z[col].fillna("__MISSING__").astype(str)
        return bundle["model"].predict_proba(Z)[:,1]
    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None,""): return None
        try: result=float(value)
        except (TypeError,ValueError): return None
        return result if math.isfinite(result) else None
    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if isinstance(value,datetime): return value.date()
        if isinstance(value,date): return value
        if not value: return None
        try: return datetime.strptime(str(value)[:10],"%Y-%m-%d").date()
        except ValueError: return None
    def _training_window(self):
        r=self.schema.get("training_date_range") or self.safety.get("training_date_range")
        a,b=(r["min"],r["max"]) if isinstance(r,dict) else r
        return datetime.strptime(a,"%Y-%m-%d").date(),datetime.strptime(b,"%Y-%m-%d").date()
    def validate_and_build_features(self,payload:dict[str,Any]):
        errors=[];warnings=[]
        forbidden_sent=sorted(k for k in payload if k in self.forbidden)
        if forbidden_sent: errors.append("لا يمكن استخدام نتائج ما بعد إطلاق الحملة كمدخلات: "+", ".join(forbidden_sent))
        unknown=sorted(k for k in payload if k not in self.allowed_input_fields and k not in self.forbidden)
        if unknown: errors.append("حقول غير مدعومة: "+", ".join(unknown))
        objective=str(payload.get("Campaign_Objective","")).strip()
        if objective not in self.objectives: errors.append(f"هدف الحملة غير مدعوم: {objective or '<missing>'}"); objective=""
        required=set(self.required)
        if objective: required |= self.objective_required.get(objective,set())
        for field in sorted(required):
            if payload.get(field) in (None,""): errors.append(f"الحقل مطلوب: {field}")
        start_date=self._parse_date(payload.get("Start_Date"))
        if start_date is None: errors.append("Start_Date يجب أن يكون بصيغة YYYY-MM-DD.")
        numeric_ranges=self.schema["numeric_ranges"]
        for field,bounds in numeric_ranges.items():
            if field not in payload or payload.get(field) in (None,""): continue
            value=self._to_float(payload.get(field))
            if value is None: errors.append(f"{field} يجب أن يكون رقمًا صالحًا."); continue
            lo,hi=bounds
            if value<lo or value>hi: errors.append(f"{field} يجب أن يكون بين {lo} و {hi}.")
        for field,options in self.schema["options"].items():
            if field in payload and payload.get(field) not in (None,"") and str(payload[field]) not in options: errors.append(f"قيمة غير مدعومة في {field}: {payload[field]}")
        platform=str(payload.get("Platform",""));placement=str(payload.get("Placement",""));content_type=str(payload.get("Content_Type",""));bidding=str(payload.get("Bidding_Strategy",""));region=str(payload.get("Region",""));city=str(payload.get("City",""))
        if platform and placement and placement not in self.schema["platform_placements"].get(platform,[]): errors.append(f"الموضع {placement!r} غير متوافق مع منصة {platform!r} في V5.")
        if platform and content_type and content_type not in self.schema["platform_content_types"].get(platform,[]): errors.append(f"نوع المحتوى {content_type!r} غير متوافق مع منصة {platform!r} في V5.")
        if objective and bidding and bidding not in self.schema["objective_bidding_strategies"].get(objective,[]): errors.append(f"استراتيجية المزايدة {bidding!r} غير متوافقة مع هدف {objective!r}.")
        if region and city and city not in self.schema["region_city"].get(region,[]): errors.append(f"المدينة {city!r} لا تتبع المنطقة {region!r} في مخطط V5.")
        if errors: raise InputValidationError(errors)
        b=self.models[objective];model_features=b["features"];row={};numeric_model_fields=set(numeric_ranges)|self.SCORE_FIELDS|self.HISTORY_NUMERIC|{"Expected_AOV_SAR","Discount_Percentage","App_Store_Rating","App_Size_MB","Planned_Frequency"}
        for feature in model_features:
            if feature in {"Start_Month","Start_Quarter","Start_DayOfWeek","Budget_Per_1000_Audience_SAR"}: continue
            raw=payload.get(feature,None);row[feature]=np.nan if raw in (None,"") else (float(raw) if feature in numeric_model_fields else str(raw))
        row["Start_Month"]=str(start_date.month);row["Start_Quarter"]=str((start_date.month-1)//3+1);row["Start_DayOfWeek"]=str(start_date.weekday());row["Budget_Per_1000_Audience_SAR"]=float(payload["Budget_SAR"])/float(payload["Estimated_Audience_Size"])*1000.0
        train_start,train_end=self._training_window()
        if start_date<train_start or start_date>train_end: warnings.append(f"تاريخ الحملة {start_date.isoformat()} خارج نافذة تدريب V5 ({train_start.isoformat()} إلى {train_end.isoformat()})؛ قد تزيد مخاطر الانحراف الزمني.")
        if sum(payload.get(f) not in (None,"") for f in self.history_fields)==0: warnings.append("لم تُدخل بيانات تاريخية للعلامة. سيستخدم المودل آلية القيم المفقودة المجمدة، وقد تكون النتيجة أقل تخصيصًا للعلامات الجديدة.")
        for score_field,label in [("Creative_Quality_Score","جودة الإبداع"),("Landing_Page_Quality_Score","صفحة الهبوط"),("Content_Audience_Fit_Score","ملاءمة المحتوى للجمهور"),("Tracking_Readiness_Score","جاهزية التتبع"),("Arabic_Localization_Score","التوطين العربي"),("Trust_Score","عناصر الثقة")]:
            val=float(payload.get(score_field,100))
            if val<45: warnings.append(f"{label} منخفضة ({val:.0f}/100) وتحتاج مراجعة قبل الإطلاق.")
        if float(payload.get("Auction_Competition_Score",0))>82: warnings.append("منافسة المزاد مرتفعة جدًا؛ قد تزيد تكلفة الوصول والتحويل وعدم اليقين.")
        if float(payload.get("Budget_Adequacy_Score",100))<40: warnings.append("الميزانية منخفضة قياسًا بحجم الجمهور والهدف وفق معايرة V5.")
        return objective,pd.DataFrame([row],columns=model_features),warnings
    def _recommendations(self,payload,predicted_success):
        recs=[];rules=[("Creative_Quality_Score",55,"جهّز أكثر من نسخة إبداعية مناسبة للمنصة مع CTA واضح وHook أقوى."),("Landing_Page_Quality_Score",55,"حسّن سرعة صفحة الهبوط وتطابق الرسالة ووضوح CTA قبل رفع الميزانية."),("Tracking_Readiness_Score",65,"أكمل Pixel/Analytics وUTM وأحداث التحويل واختبرها قبل الإطلاق."),("Arabic_Localization_Score",65,"راجع اللغة والسياق السعودي وRTL والـCTA المحلي بدل الاكتفاء بالترجمة."),("Content_Audience_Fit_Score",60,"خصص الرسالة والصيغة وفق نية الجمهور والشريحة والمنصة."),("Trust_Score",60,"أضف عناصر ثقة واضحة مثل المراجعات والسياسات والضمان وطرق التواصل."),("Budget_Adequacy_Score",45,"قلّص الجمهور أو ارفع الميزانية؛ كثافة الميزانية الحالية ضعيفة بالنسبة للهدف.")]
        for field,cut,text in rules:
            val=self._to_float(payload.get(field))
            if val is not None and val<cut: recs.append(text)
        comp=self._to_float(payload.get("Auction_Competition_Score"))
        if comp is not None and comp>=80: recs.append("المزاد مزدحم؛ استخدم Creatives متعددة واختبر شرائح/Placements بديلة بدل التوسع دفعة واحدة.")
        if not predicted_success and not recs: recs.append("النتيجة أقل من Threshold المجمّد؛ اختبر سيناريوهات تحسين العرض والإبداع والاستهداف قبل الإطلاق.")
        return recs[:5]
    def predict(self,payload):
        objective,X,warnings=self.validate_and_build_features(payload);b=self.models[objective]
        with py_warnings.catch_warnings():
            py_warnings.filterwarnings("ignore",message="X does not have valid feature names.*",category=UserWarning);p_lgb=float(b["lgb"].predict_proba(X)[:,1][0])
        p_cat=float(self._pred_cat(b["cat"],X)[0]);raw=b["weight_lgb"]*p_lgb+(1.0-b["weight_lgb"])*p_cat;probability=float(b["calibrator"].predict_proba(np.array([[raw]]))[:,1][0]);threshold=float(b["threshold"]);predicted_success=probability>=threshold;margin=abs(probability-threshold);strength="strong" if margin>=.20 else "moderate" if margin>=.10 else "borderline";metrics=self.lockbox.get("per_objective",{}).get(objective,{});auc=metrics.get("roc_auc");agreement=max(0.0,1.0-abs(p_lgb-p_cat));confidence="مرتفع" if strength=="strong" and (auc or 0)>=.90 and agreement>=.85 else "حذر" if strength=="borderline" or (auc or 0)<.82 else "متوسط"
        if objective=="Awareness": warnings.append("مودل Awareness هو الأقل ROC-AUC بين أهداف V5؛ تعامل مع النتيجة كإشارة تخطيطية وليس حكمًا نهائيًا.")
        return PredictionResult(objective,probability,predicted_success,threshold,margin,strength,confidence,auc,metrics.get("balanced_accuracy"),p_lgb,p_cat,agreement,warnings,self._recommendations(payload,predicted_success),len(b["features"]),self.suite.get("dataset_version","V5"),self.suite.get("freeze_manifest_sha256",""))

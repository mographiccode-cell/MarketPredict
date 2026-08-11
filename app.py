from __future__ import annotations

import csv
import io
import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

from assessment import RUBRICS, apply_auto_assessment, assessment_summary
from auth import hash_password, verify_password
from db import connect, init_db, save_prediction
from predictor import CampaignPredictor, InputValidationError

BASE_DIR = Path(__file__).resolve().parent
APP_TITLE = "MarketPredict V5"

app = FastAPI(title=APP_TITLE, docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "marketpredict-demo-session-v1-change-in-production"), session_cookie="saudi_campaign_session", same_site="lax", https_only=os.environ.get("HTTPS_ONLY", "0") == "1", max_age=60 * 60 * 8)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

init_db()
DEMO_EMAIL = os.environ.get("DEMO_EMAIL", "demo@marketpredict.app").strip().lower()
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "Demo12345!")
DEMO_NAME = "MarketPredict Demo"

def _ensure_demo_user() -> int:
    with connect() as con:
        row = con.execute("SELECT id FROM users WHERE email=?", (DEMO_EMAIL,)).fetchone()
        if row:
            return int(row["id"])
        cur = con.execute("INSERT INTO users(name,email,password_hash,is_admin) VALUES(?,?,?,0)", (DEMO_NAME, DEMO_EMAIL, hash_password(DEMO_PASSWORD)))
        return int(cur.lastrowid)

DEMO_USER_ID = _ensure_demo_user()
predictor = CampaignPredictor()
schema = predictor.schema
_rate: dict[str, deque[float]] = defaultdict(deque)
OBJECTIVE_AR = {"Sales":"المبيعات","Traffic":"الزيارات","Awareness":"الوعي","Lead Generation":"العملاء المحتملون","Engagement":"التفاعل","App Installs":"تثبيت التطبيق"}

def _user(request: Request):
    uid=request.session.get("user_id")
    if not uid:return None
    with connect() as con:return con.execute("SELECT id,name,email,is_admin FROM users WHERE id=?",(uid,)).fetchone()

def _csrf(request: Request)->str:
    token=request.session.get("csrf_token")
    if not token:
        token=secrets.token_urlsafe(32);request.session["csrf_token"]=token
    return token

def _flash(request:Request,message:str,category:str="info"):
    arr=request.session.setdefault("flashes",[]);arr.append([category,message]);request.session["flashes"]=arr[-8:]

def _pop_flashes(request:Request):return request.session.pop("flashes",[])

def _context(request:Request,**kwargs):
    data={"request":request,"g_user":_user(request),"csrf_token":_csrf(request),"flashes":_pop_flashes(request),"schema":schema,"rubrics":RUBRICS,"objective_ar":OBJECTIVE_AR,"app_title":APP_TITLE,"model_version":predictor.suite.get("dataset_version","V5"),"lock_summary":predictor.lockbox.get("summary",{}),"demo_email":DEMO_EMAIL,"demo_password":DEMO_PASSWORD};data.update(kwargs);return data

def _check_csrf(request:Request,token:str|None):
    expected=request.session.get("csrf_token","")
    if not token or not expected or not secrets.compare_digest(str(token),str(expected)):raise PermissionError("CSRF validation failed")

def _require_login(request:Request):
    user=_user(request)
    if not user:return None,RedirectResponse(url=f"/login?next={request.url.path}",status_code=303)
    return user,None

def _rate_limit(key:str,limit:int=20,window:int=60):
    now=time.time();q=_rate[key]
    while q and q[0]<now-window:q.popleft()
    if len(q)>=limit:raise RuntimeError("rate_limited")
    q.append(now)

def _safe_next(value:str|None)->str:
    if value and value.startswith("/") and not value.startswith("//"):return value
    return "/dashboard"

@app.middleware("http")
async def security_headers(request:Request,call_next):
    response=await call_next(request);response.headers["X-Content-Type-Options"]="nosniff";response.headers["X-Frame-Options"]="DENY";response.headers["Referrer-Policy"]="same-origin";response.headers["Permissions-Policy"]="camera=(), microphone=(), geolocation=()";response.headers["Content-Security-Policy"]="default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'";return response

@app.exception_handler(PermissionError)
async def permission_handler(request:Request,exc:PermissionError):return templates.TemplateResponse(request,"error.html",_context(request,code=400,title="طلب غير صالح",message="تعذر التحقق من الطلب. أعد تحميل الصفحة وحاول مرة أخرى."),status_code=400)

@app.get("/",response_class=HTMLResponse)
async def index(request:Request):
    if _user(request):return RedirectResponse("/dashboard",status_code=303)
    return templates.TemplateResponse(request,"landing.html",_context(request))

@app.get("/model",response_class=HTMLResponse)
async def model_page(request:Request):return templates.TemplateResponse(request,"model.html",_context(request,per_objective=predictor.lockbox.get("per_objective",{})))

@app.api_route("/register",methods=["GET","POST"],response_class=HTMLResponse)
async def register(request:Request):
    if request.method=="POST":
        form=dict(await request.form());_check_csrf(request,form.get("_csrf"));name=str(form.get("name","")).strip();email=str(form.get("email","")).strip().lower();password=str(form.get("password",""));errors=[]
        if len(name)<2:errors.append("الاسم يجب أن يكون حرفين على الأقل.")
        if "@" not in email or len(email)>160:errors.append("البريد الإلكتروني غير صالح.")
        if len(password)<8:errors.append("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
        if not errors:
            try:
                with connect() as con:cur=con.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",(name,email,hash_password(password)));request.session.clear();request.session["user_id"]=cur.lastrowid;_csrf(request)
                return RedirectResponse("/dashboard",status_code=303)
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():errors.append("هذا البريد مسجل مسبقًا.")
                else:raise
        for e in errors:_flash(request,e,"error")
        return RedirectResponse("/register",status_code=303)
    return templates.TemplateResponse(request,"register.html",_context(request))

@app.api_route("/login",methods=["GET","POST"],response_class=HTMLResponse)
async def login(request:Request):
    if request.method=="POST":
        form=dict(await request.form());_check_csrf(request,form.get("_csrf"));email=str(form.get("email","")).strip().lower();password=str(form.get("password",""))
        with connect() as con:user=con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
        if user and verify_password(user["password_hash"],password):request.session.clear();request.session["user_id"]=user["id"];_csrf(request);return RedirectResponse(_safe_next(request.query_params.get("next")),status_code=303)
        _flash(request,"بيانات الدخول غير صحيحة.","error");return RedirectResponse(f"/login?next={request.query_params.get('next','')}",status_code=303)
    return templates.TemplateResponse(request,"login.html",_context(request))

@app.post("/demo-login")
async def demo_login(request:Request):
    form=dict(await request.form());_check_csrf(request,form.get("_csrf"));request.session.clear();request.session["user_id"]=_ensure_demo_user();request.session["is_demo"]=True;_csrf(request);return RedirectResponse(_safe_next(request.query_params.get("next")),status_code=303)

@app.post("/logout")
async def logout(request:Request):form=dict(await request.form());_check_csrf(request,form.get("_csrf"));request.session.clear();return RedirectResponse("/",status_code=303)

@app.get("/dashboard",response_class=HTMLResponse)
async def dashboard(request:Request):
    user,redirect=_require_login(request)
    if redirect:return redirect
    with connect() as con:
        recent=con.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 6",(user["id"],)).fetchall();stats=con.execute("SELECT COUNT(*) n, AVG(probability) avg_prob, SUM(predicted_success) successes, AVG(model_roc_auc) avg_auc FROM predictions WHERE user_id=?",(user["id"],)).fetchone();objectives=con.execute("SELECT objective, COUNT(*) n, AVG(probability) avgp FROM predictions WHERE user_id=? GROUP BY objective ORDER BY n DESC",(user["id"],)).fetchall()
    return templates.TemplateResponse(request,"dashboard.html",_context(request,recent=recent,stats=stats,objective_stats=objectives))

@app.api_route("/predict",methods=["GET","POST"],response_class=HTMLResponse)
async def predict_page(request:Request):
    user,redirect=_require_login(request)
    if redirect:return redirect
    if request.method=="POST":
        _rate_limit(f"user:{user['id']}");form=dict(await request.form());_check_csrf(request,form.pop("_csrf",None));raw_form={k:v for k,v in form.items() if v not in (None,"")}
        try:payload,provenance,assessment_warnings=apply_auto_assessment(raw_form);result=predictor.predict(payload).as_dict();result["warnings"]=assessment_warnings+result.get("warnings",[])
        except (InputValidationError,ValueError) as exc:
            errors=exc.errors if isinstance(exc,InputValidationError) else [str(exc)]
            for e in errors:_flash(request,e,"error")
            return templates.TemplateResponse(request,"predict.html",_context(request,values=raw_form),status_code=400)
        pid=save_prediction(user["id"],payload,result,provenance);request.session["last_prediction"]={"id":pid,"objective":result["objective"],"probability":result["probability"],"predicted_success":int(result["predicted_success"]),"threshold":result["threshold"],"decision_strength":result["decision_strength"],"confidence_label":result.get("confidence_label",""),"model_roc_auc":result.get("model_roc_auc"),"model_balanced_accuracy":result.get("model_balanced_accuracy"),"warnings":result.get("warnings",[])[:4],"recommendations":result.get("recommendations",[])[:4],"assessment":assessment_summary(payload)};return RedirectResponse(f"/result/{pid}",status_code=303)
    return templates.TemplateResponse(request,"predict.html",_context(request,values={}))

@app.get("/result/{prediction_id}",response_class=HTMLResponse)
async def prediction_result(request:Request,prediction_id:int):
    user,redirect=_require_login(request)
    if redirect:return redirect
    with connect() as con:row=con.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",(prediction_id,user["id"])).fetchone()
    if not row:
        fallback=request.session.get("last_prediction") or {}
        if int(fallback.get("id",-1))!=prediction_id:return templates.TemplateResponse(request,"error.html",_context(request,code=404,title="غير موجود",message="لم يتم العثور على التنبؤ المطلوب."),status_code=404)
        return templates.TemplateResponse(request,"result.html",_context(request,p=fallback,inputs={},warnings=fallback.get("warnings",[]),recommendations=fallback.get("recommendations",[]),provenance={},assessment=fallback.get("assessment",[]),serverless_fallback=True))
    inputs=json.loads(row["inputs_json"]);warnings=json.loads(row["warnings_json"]);recs=json.loads(row["recommendations_json"] or "[]");provenance=json.loads(row["provenance_json"] or "{}");assessment=assessment_summary(inputs);return templates.TemplateResponse(request,"result.html",_context(request,p=row,inputs=inputs,warnings=warnings,recommendations=recs,provenance=provenance,assessment=assessment,serverless_fallback=False))

@app.get("/history",response_class=HTMLResponse)
async def history(request:Request):
    user,redirect=_require_login(request)
    if redirect:return redirect
    with connect() as con:rows=con.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 250",(user["id"],)).fetchall()
    return templates.TemplateResponse(request,"history.html",_context(request,rows=rows))

@app.get("/history.csv")
async def history_csv(request:Request):
    user,redirect=_require_login(request)
    if redirect:return redirect
    with connect() as con:rows=con.execute("SELECT id,objective,probability,predicted_success,threshold,decision_strength,confidence_label,model_roc_auc,created_at FROM predictions WHERE user_id=? ORDER BY id DESC",(user["id"],)).fetchall()
    out=io.StringIO();w=csv.writer(out);w.writerow(["id","objective","probability","predicted_success","threshold","decision_strength","confidence_label","model_roc_auc","created_at"])
    for r in rows:w.writerow(list(r))
    return StreamingResponse(iter([out.getvalue()]),media_type="text/csv; charset=utf-8",headers={"Content-Disposition":"attachment; filename=prediction_history.csv"})

@app.get("/admin",response_class=HTMLResponse)
async def admin(request:Request):
    user,redirect=_require_login(request)
    if redirect:return redirect
    if not user["is_admin"]:return templates.TemplateResponse(request,"error.html",_context(request,code=403,title="غير مصرح",message="هذه الصفحة مخصصة للإدارة."),status_code=403)
    with connect() as con:users=con.execute("SELECT u.id,u.name,u.email,u.created_at,COUNT(p.id) predictions FROM users u LEFT JOIN predictions p ON p.user_id=u.id GROUP BY u.id ORDER BY u.id DESC").fetchall();counts=con.execute("SELECT objective,COUNT(*) n,AVG(probability) avgp FROM predictions GROUP BY objective ORDER BY n DESC").fetchall()
    return templates.TemplateResponse(request,"admin.html",_context(request,users=users,counts=counts))

@app.get("/api/schema")
async def api_schema():return JSONResponse({"version":schema["version"],"objectives":schema["objectives"],"options":schema["options"],"platform_placements":schema["platform_placements"],"platform_content_types":schema["platform_content_types"],"objective_bidding_strategies":schema["objective_bidding_strategies"],"region_city":schema["region_city"],"objective_specific_required":schema["objective_specific_required"],"labels":schema.get("field_labels_ar",{})})

@app.get("/api/model-info")
async def model_info():return JSONResponse({"model_version":predictor.suite.get("dataset_version"),"objectives":sorted(predictor.objectives),"freeze_manifest_sha256":predictor.suite.get("freeze_manifest_sha256"),"final_lockbox":predictor.lockbox.get("summary",{}),"per_objective":predictor.lockbox.get("per_objective",{}),"policy":"Pre-launch only. Current-campaign post-launch fields are rejected.","public_evidence_calibration":predictor.safety.get("public_evidence_calibration")})

@app.post("/api/predict")
async def api_predict(request:Request):
    user,redirect=_require_login(request)
    if redirect:return JSONResponse({"error":"authentication_required"},status_code=401)
    _rate_limit(f"api:{user['id']}")
    if "application/json" not in request.headers.get("content-type",""):return JSONResponse({"error":"application/json required"},status_code=415)
    raw=await request.json()
    try:payload,provenance,assessment_warnings=apply_auto_assessment(dict(raw));result=predictor.predict(payload).as_dict();result["warnings"]=assessment_warnings+result.get("warnings",[])
    except (InputValidationError,ValueError) as exc:return JSONResponse({"error":"validation_failed","details":exc.errors if isinstance(exc,InputValidationError) else [str(exc)]},status_code=400)
    result["prediction_id"]=save_prediction(user["id"],payload,result,provenance);return JSONResponse(result)

@app.get("/health")
async def health():return JSONResponse({"status":"ok","model_version":"V5","objectives":len(predictor.models),"freeze":predictor.suite.get("freeze_manifest_sha256")})

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host=os.environ.get("HOST","127.0.0.1"),port=int(os.environ.get("PORT","5000")),reload=False)

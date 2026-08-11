from __future__ import annotations

import json,os,sqlite3
from pathlib import Path
from typing import Any

BASE_DIR=Path(__file__).resolve().parent
DB_PATH=Path(os.environ.get('DB_PATH') or ((Path('/tmp')/'marketpredict.db') if os.environ.get('VERCEL') else (BASE_DIR/'app.db')))

def connect():
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row;con.execute('PRAGMA foreign_keys=ON');con.execute('PRAGMA journal_mode=WAL');return con

def _columns(con:sqlite3.Connection,table:str)->set[str]:return {r[1] for r in con.execute(f'PRAGMA table_info({table})')}

def init_db():
    with connect() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT NOT NULL UNIQUE COLLATE NOCASE,password_hash TEXT NOT NULL,is_admin INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,objective TEXT NOT NULL,probability REAL NOT NULL,predicted_success INTEGER NOT NULL,threshold REAL NOT NULL,decision_strength TEXT NOT NULL,confidence_label TEXT NOT NULL DEFAULT '',model_roc_auc REAL,model_balanced_accuracy REAL,inputs_json TEXT NOT NULL,warnings_json TEXT NOT NULL,recommendations_json TEXT NOT NULL DEFAULT '[]',provenance_json TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS idx_predictions_user_created ON predictions(user_id,created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_predictions_objective ON predictions(objective);''')
        cols=_columns(con,'predictions')
        for name,ddl in {'confidence_label':"TEXT NOT NULL DEFAULT ''",'model_balanced_accuracy':'REAL','recommendations_json':"TEXT NOT NULL DEFAULT '[]'",'provenance_json':"TEXT NOT NULL DEFAULT '{}'"}.items():
            if name not in cols:con.execute(f'ALTER TABLE predictions ADD COLUMN {name} {ddl}')

def save_prediction(user_id:int,payload:dict[str,Any],result:dict[str,Any],provenance:dict[str,Any]|None=None)->int:
    with connect() as con:
        cur=con.execute('''INSERT INTO predictions(user_id,objective,probability,predicted_success,threshold,decision_strength,confidence_label,model_roc_auc,model_balanced_accuracy,inputs_json,warnings_json,recommendations_json,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(user_id,result['objective'],result['probability'],int(result['predicted_success']),result['threshold'],result['decision_strength'],result.get('confidence_label',''),result.get('model_roc_auc'),result.get('model_balanced_accuracy'),json.dumps(payload,ensure_ascii=False),json.dumps(result.get('warnings',[]),ensure_ascii=False),json.dumps(result.get('recommendations',[]),ensure_ascii=False),json.dumps(provenance or {},ensure_ascii=False)))
        return int(cur.lastrowid)

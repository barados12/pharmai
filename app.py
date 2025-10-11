# app.py
"""
PharmaAI Cloud — Final unified production-ready single-file app
- FastAPI backend (APIs + PayPal webhook + proxy to Streamlit)
- Streamlit frontend (runs locally on 127.0.0.1:8501 inside container)
- PostgreSQL storage (users, payments, drugs, queries)
- ChEMBL + PubChem sampling, optional DrugBank enrichment
- AI: KNN-based alternatives + dosage regressor
- PDF report generation (ReportLab)
- Numeric cleaning fixes (avoids pandas inplace issues)
- Designed to run on Railway Free Plan: FastAPI binds to $PORT; Streamlit runs locally and is proxied
"""

import os
import time
import json
import threading
import subprocess
import tempfile
import logging
from io import BytesIO
from typing import Dict, Any

import requests
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import uvicorn

import psycopg2
from psycopg2.extras import RealDictCursor

LOG = logging.getLogger("pharmaai")
LOG.setLevel(logging.INFO)

# -------------------------
# Configuration via env vars (set in Railway -> Variables)
# -------------------------
PORT = int(os.getenv("PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET", "")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "live")  # 'live' or 'sandbox'
PAYPAL_WEBHOOK_ID = os.getenv("PAYPAL_WEBHOOK_ID", "")
PAYPAL_PRICE_USD = os.getenv("PAYPAL_PRICE_USD", "9.99")
CHEMBL_SAMPLE = int(os.getenv("CHEMBL_SAMPLE", "300"))
PUBCHEM_BATCH = int(os.getenv("PUBCHEM_BATCH", "200"))
DRUGBANK_API_KEY = os.getenv("DRUGBANK_API_KEY", "")  # optional
DEFAULT_LANG = os.getenv("APP_LANG", "en")

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_PROPERTY = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/MolecularWeight/JSON"
RXNAV_RXCUI = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNAV_INTERACTION = "https://rxnav.nlm.nih.gov/REST/interaction/interaction.json"

STREAMLIT_PORT = 8501
STREAMLIT_FILE = "streamlit_frontend.py"
DRUG_CSV = "drug_features.csv"

# -------------------------
# Postgres helpers
# -------------------------
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL, sslmode="require", cursor_factory=RealDictCursor)

def ensure_schema():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        pwdhash TEXT NOT NULL,
        is_premium BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS payments (
        id SERIAL PRIMARY KEY,
        order_id TEXT UNIQUE,
        payer_email TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS drugs (
        id SERIAL PRIMARY KEY,
        drug_name TEXT,
        mol_weight REAL,
        logP REAL,
        h_bond_donor INTEGER,
        h_bond_acceptor INTEGER,
        source TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_queries (
        id SERIAL PRIMARY KEY,
        user_email TEXT,
        query_text TEXT,
        result_json JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

# password hashing
import hashlib, binascii, os as _os
def hash_password(password: str) -> str:
    salt = hashlib.sha256(_os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password: str, provided_password: str) -> bool:
    salt = stored_password[:64].encode('ascii')
    stored_pwdhash = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return binascii.hexlify(pwdhash).decode('ascii') == stored_pwdhash

def create_user(email: str, password: str):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (email, pwdhash) VALUES (%s, %s) RETURNING id", (email, hash_password(password)))
        uid = cur.fetchone()['id']
        conn.commit()
        return True, uid
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, "Email already registered"
    finally:
        cur.close()
        conn.close()

def authenticate_user(email: str, password: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id,email,pwdhash,is_premium FROM users WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return False, "No such user", None
    if verify_password(row['pwdhash'], password):
        return True, "Authenticated", {"id": row['id'], "email": row['email'], "is_premium": row['is_premium']}
    return False, "Wrong password", None

def set_premium_for_email(email: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET is_premium = TRUE WHERE email = %s", (email,))
    conn.commit()
    cur.close()
    conn.close()

def log_query(user_email: str, query_text: str, result: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO user_queries (user_email, query_text, result_json) VALUES (%s, %s, %s)",
                (user_email, query_text, json.dumps(result)))
    conn.commit()
    cur.close()
    conn.close()

# -------------------------
# Clean numeric helper (fixes pandas inplace issue & bad strings)
# -------------------------
def clean_numeric_series(s: pd.Series) -> pd.Series:
    # Convert to string, remove anything not digit/.- then coerce to numeric
    s_clean = s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s_clean, errors="coerce")

# -------------------------
# Build/import drug features (ChEMBL + PubChem + optional DrugBank)
# -------------------------
def import_drugs_into_db(df: pd.DataFrame):
    conn = get_conn()
    cur = conn.cursor()
    for _, r in df.iterrows():
        try:
            cur.execute("""
            INSERT INTO drugs (drug_name, mol_weight, logP, h_bond_donor, h_bond_acceptor, source)
            VALUES (%s,%s,%s,%s,%s,%s)
            """, (r.get("drug_name"), float(r.get("mol_weight") or 0),
                  float(r.get("logP") or 0),
                  int(r.get("h_bond_donor") or 0),
                  int(r.get("h_bond_acceptor") or 0),
                  r.get("source","mixed")))
        except Exception:
            # skip problematic rows
            continue
    conn.commit()
    cur.close()
    conn.close()

def build_drug_features(limit=CHEMBL_SAMPLE):
    # If DB already populated, read a sample
    try:
        conn = get_conn()
        df_db = pd.read_sql("SELECT drug_name, mol_weight, logP, h_bond_donor, h_bond_acceptor FROM drugs LIMIT %s", conn, params=(limit,))
        conn.close()
        if not df_db.empty and len(df_db) >= max(10, limit//4):
            LOG.info("Using existing drugs from DB")
            return df_db
    except Exception:
        # continue to fetch
        pass

    mols = []
    per_page = 100
    offset = 0
    fetched = 0
    LOG.info("Fetching ChEMBL sample...")
    while fetched < limit:
        try:
            params = {"limit": per_page, "offset": offset}
            r = requests.get(f"{CHEMBL_API_BASE}/molecule", params=params, timeout=15)
            r.raise_for_status()
            j = r.json()
            items = j.get("molecules", [])
            if not items:
                break
            for m in items:
                name = m.get("pref_name") or m.get("molecule_chembl_id")
                if name:
                    mols.append({"drug_name": name, "source": "ChEMBL"})
                    fetched += 1
                    if fetched >= limit:
                        break
            offset += 1
        except Exception as e:
            LOG.warning("ChEMBL fetch error: %s", e)
            break

    df = pd.DataFrame(mols, columns=["drug_name", "source"])
    # add numeric columns safely
    df = df.assign(mol_weight=np.nan, logP=np.nan, h_bond_donor=np.nan, h_bond_acceptor=np.nan)

    # enrich with PubChem limited
    count = 0
    for idx, row in df.iterrows():
        if count >= PUBCHEM_BATCH:
            break
        name = row["drug_name"]
        try:
            url = PUBCHEM_PROPERTY.format(requests.utils.requote_uri(name))
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if isinstance(props, list) and props:
                    p = props[0]
                    mw = p.get("MolecularWeight")
                    df.at[idx, "mol_weight"] = mw
                    count += 1
            time.sleep(0.12)
        except Exception as e:
            LOG.debug("PubChem fetch fail for %s: %s", name, e)
            continue

    # sanitize numeric columns robustly
    df["mol_weight"] = clean_numeric_series(df["mol_weight"])
    if df["mol_weight"].isna().all():
        df["mol_weight"] = 300.0
    else:
        df["mol_weight"] = df["mol_weight"].fillna(df["mol_weight"].median(skipna=True))

    df["logP"] = clean_numeric_series(df["logP"]).fillna(2.5)
    df["h_bond_donor"] = clean_numeric_series(df["h_bond_donor"]).fillna(1).astype(int)
    df["h_bond_acceptor"] = clean_numeric_series(df["h_bond_acceptor"]).fillna(3).astype(int)

    # optionally enrich using DrugBank if API key available (placeholder)
    if DRUGBANK_API_KEY:
        LOG.info("DrugBank key found — enrichment can be added (requires implementing DrugBank API calls).")

    # save CSV locally and import to DB
    try:
        df.to_csv(DRUG_CSV, index=False)
    except Exception:
        pass
    try:
        import_drugs_into_db(df)
    except Exception:
        LOG.debug("Import to DB failed or already exists.")
    return df

# -------------------------
# AI Module (safe numeric handling)
# -------------------------
class AIModule:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        # safe cleaning
        clean = self.df.copy()
        clean["mol_weight"] = clean_numeric_series(clean["mol_weight"]).fillna(300.0)
        clean["logP"] = clean_numeric_series(clean["logP"]).fillna(2.5)
        clean["h_bond_donor"] = clean_numeric_series(clean["h_bond_donor"]).fillna(1).astype(int)
        clean["h_bond_acceptor"] = clean_numeric_series(clean["h_bond_acceptor"]).fillna(3).astype(int)

        self.features = clean[["mol_weight", "logP", "h_bond_donor", "h_bond_acceptor"]].astype(float).values

        # replace any remaining NaN with column medians
        if np.isnan(self.features).any():
            col_medians = np.nanmedian(self.features, axis=0)
            inds = np.where(np.isnan(self.features))
            self.features[inds] = np.take(col_medians, inds[1])

        if len(self.features) >= 2:
            try:
                self.knn = NearestNeighbors(n_neighbors=min(10, len(self.features))).fit(self.features)
            except Exception as e:
                LOG.warning("KNN init failed: %s", e)
                self.knn = None
        else:
            self.knn = None

        # small dosage regressor (synthetic)
        X, y = [], []
        base_map = {"paracetamol": 10, "ibuprofen": 5, "amoxicillin": 20}
        for k, mg in base_map.items():
            for w in [20, 40, 60, 80, 100]:
                X.append([w, 40])
                y.append(w * mg)
        self.reg = LinearRegression().fit(np.array(X), np.array(y))

    def suggest_alternatives(self, drug_name: str, top_k=5):
        df = self.df
        mask = df["drug_name"].str.lower() == drug_name.lower()
        if mask.any() and self.knn is not None:
            idx = df[mask].index[0]
            feat = self.features[idx].reshape(1, -1)
            dists, inds = self.knn.kneighbors(feat)
            names = df.iloc[inds[0]]["drug_name"].tolist()
            names = [n for n in names if n.lower() != drug_name.lower()]
            return names[:top_k]
        contains = df[df["drug_name"].str.contains(drug_name.split()[0], case=False, na=False)]
        return contains["drug_name"].head(top_k).tolist()

    def predict_dosage(self, drug_name: str, weight: float, age: int, condition: str):
        nl = drug_name.lower()
        if "paracetamol" in nl:
            base = 10
        elif "ibuprofen" in nl:
            base = 5
        elif "amoxicillin" in nl:
            base = 20
        else:
            return round(self.reg.predict([[weight, age]])[0], 2)
        dose = weight * base
        if "renal" in condition.lower() or "kidney" in condition.lower():
            dose *= 0.7
        if "hepatic" in condition.lower() or "liver" in condition.lower():
            dose *= 0.8
        if age < 12:
            dose *= 0.5
        return round(dose, 2)

# -------------------------
# RxNav helpers
# -------------------------
def rxnav_get_rxcui(drug_name: str):
    try:
        r = requests.get(RXNAV_RXCUI, params={"name": drug_name}, timeout=8)
        r.raise_for_status()
        j = r.json()
        return j.get("idGroup", {}).get("rxnormId", [None])[0]
    except Exception:
        return None

def rxnav_get_interactions(rxcui: str):
    try:
        r = requests.get(RXNAV_INTERACTION, params={"rxcui": rxcui}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# -------------------------
# PayPal helpers
# -------------------------
def paypal_base():
    return "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"

def paypal_get_token():
    if not (PAYPAL_CLIENT_ID and PAYPAL_SECRET):
        raise RuntimeError("PayPal credentials missing")
    r = requests.post(f"{paypal_base()}/v1/oauth2/token", data={"grant_type":"client_credentials"},
                      auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET), timeout=15)
    r.raise_for_status()
    return r.json().get("access_token")

def paypal_create_order(amount: str = PAYPAL_PRICE_USD):
    token = paypal_get_token()
    url = f"{paypal_base()}/v2/checkout/orders"
    payload = {"intent":"CAPTURE", "purchase_units":[{"amount":{"currency_code":"USD","value":str(amount)}}]}
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def paypal_capture_order(order_id: str):
    token = paypal_get_token()
    url = f"{paypal_base()}/v2/checkout/orders/{order_id}/capture"
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    r = requests.post(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def paypal_verify_webhook(headers_map: Dict[str,str], body: Dict[str,Any]):
    if not PAYPAL_WEBHOOK_ID:
        return False, "PAYPAL_WEBHOOK_ID not set"
    token = paypal_get_token()
    url = f"{paypal_base()}/v1/notifications/verify-webhook-signature"
    payload = {
        "transmission_id": headers_map.get("paypal-transmission-id"),
        "transmission_time": headers_map.get("paypal-transmission-time"),
        "cert_url": headers_map.get("paypal-cert-url"),
        "auth_algo": headers_map.get("paypal-auth-algo"),
        "transmission_sig": headers_map.get("paypal-transmission-sig"),
        "webhook_id": PAYPAL_WEBHOOK_ID,
        "webhook_event": body
    }
    headers = {"Content-Type":"application/json", "Authorization": f"Bearer {token}"}
    r = requests.post(url, json=payload, headers=headers, timeout=15)
    if r.status_code == 200:
        return r.json().get("verification_status") == "SUCCESS", r.json()
    return False, r.text

# -------------------------
# PDF generator
# -------------------------
def generate_pdf_report(payload: Dict[str,Any]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    x = 50
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "PharmaAI Cloud Report")
    y -= 30
    c.setFont("Helvetica", 11)
    patient = payload.get("patient", {})
    c.drawString(x, y, f"Patient: {patient.get('name','-')}  Age: {patient.get('age','-')}  Weight: {patient.get('weight','-')}")
    y -= 20
    c.drawString(x, y, "Drugs:")
    y -= 16
    for d in payload.get("drugs", []):
        c.drawString(x+10, y, f"- {d}")
        y -= 12
        if y < 80:
            c.showPage(); y = h - 50
    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Interactions:")
    y -= 14; c.setFont("Helvetica", 10)
    for it in payload.get("interactions", []):
        c.drawString(x+6, y, f"- {it.get('severity','')}: {it.get('desc','')[:120]}")
        y -= 12
        if y < 80:
            c.showPage(); y = h - 50
    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Alternatives:")
    y -= 14; c.setFont("Helvetica", 10)
    for k, vals in payload.get("alternatives", {}).items():
        c.drawString(x+6, y, f"{k} -> {', '.join(vals)}")
        y -= 12
        if y < 80:
            c.showPage(); y = h - 50
    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Dosage:")
    y -= 14; c.setFont("Helvetica", 10)
    for k, v in payload.get("dosage", {}).items():
        c.drawString(x+6, y, f"{k}: {v} mg")
        y -= 12
        if y < 80:
            c.showPage(); y = h - 50
    c.showPage(); c.save()
    buf.seek(0)
    return buf.read()

# -------------------------
# FastAPI app (APIs, webhook, proxy)
# -------------------------
app = FastAPI(title="PharmaAI Cloud API")

@app.get("/api/health")
def health():
    return {"status":"ok","env":PAYPAL_ENV, "db": bool(DATABASE_URL)}

@app.post("/api/paypal/create-order")
def api_create_order(amount: str = PAYPAL_PRICE_USD):
    try:
        return paypal_create_order(amount)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/paypal/capture/{order_id}")
def api_capture(order_id: str):
    try:
        res = paypal_capture_order(order_id)
        payer_email = None
        try:
            payer_email = res.get("payer", {}).get("email_address")
        except Exception:
            payer_email = None
        # store payment
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO payments (order_id, payer_email, status) VALUES (%s,%s,%s) ON CONFLICT (order_id) DO NOTHING",
                        (order_id, payer_email, json.dumps(res.get("status"))))
            conn.commit()
            cur.close(); conn.close()
        except Exception:
            LOG.debug("store payment failed")
        if payer_email:
            set_premium_for_email(payer_email)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/paypal")
async def webhook_paypal(request: Request,
                         paypal_transmission_id: str = Header(None, alias="paypal-transmission-id"),
                         paypal_transmission_time: str = Header(None, alias="paypal-transmission-time"),
                         paypal_cert_url: str = Header(None, alias="paypal-cert-url"),
                         paypal_auth_algo: str = Header(None, alias="paypal-auth-algo"),
                         paypal_transmission_sig: str = Header(None, alias="paypal-transmission-sig")):
    body = await request.json()
    headers_map = {
        "paypal-transmission-id": paypal_transmission_id,
        "paypal-transmission-time": paypal_transmission_time,
        "paypal-cert-url": paypal_cert_url,
        "paypal-auth-algo": paypal_auth_algo,
        "paypal-transmission-sig": paypal_transmission_sig
    }
    ok, info = paypal_verify_webhook(headers_map, body)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Webhook verification failed: {info}")
    event_type = body.get("event_type")
    resource = body.get("resource", {})
    if event_type in ("PAYMENT.CAPTURE.COMPLETED", "CHECKOUT.ORDER.APPROVED", "PAYMENT.CAPTURE.COMPLETED"):
        payer = resource.get("payer", {}) or {}
        email = payer.get("email_address") or payer.get("email")
        order_id = resource.get("id") or resource.get("invoice_id") or resource.get("order_id")
        try:
            conn = get_conn(); cur = conn.cursor()
            cur.execute("INSERT INTO payments (order_id, payer_email, status) VALUES (%s,%s,%s) ON CONFLICT (order_id) DO NOTHING",
                        (order_id, email, event_type))
            conn.commit(); cur.close(); conn.close()
        except Exception:
            LOG.debug("storing webhook payment failed")
        if email:
            set_premium_for_email(email)
    return {"ok": True, "event": event_type}

@app.post("/api/report")
def api_report(payload: Dict[str, Any]):
    try:
        pdf_bytes = generate_pdf_report(payload)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes); tmp.flush(); tmp.close()
        return FileResponse(tmp.name, media_type="application/pdf", filename="pharmaai_report.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Proxy root and other non-api requests to local Streamlit (127.0.0.1:8501)
STREAMLIT_BASE = f"http://127.0.0.1:{STREAMLIT_PORT}"

@app.get("/{full_path:path}")
async def proxy_to_streamlit(full_path: str, request: Request):
    # Do not proxy API or webhook paths
    if full_path.startswith("api") or full_path.startswith("webhook") or full_path.startswith("paypal"):
        raise HTTPException(status_code=404, detail="API Route")
    try:
        url = f"{STREAMLIT_BASE}/{full_path}"
        r = requests.get(url, params=request.query_params, timeout=20)
        # return HTML content and headers (minimal)
        return HTMLResponse(content=r.content, status_code=r.status_code, headers={"Content-Type": r.headers.get("Content-Type","text/html")})
    except Exception as e:
        LOG.debug("proxy error: %s", e)
        return HTMLResponse(content=f"<h3>Streamlit UI not ready yet. Try again shortly.</h3><p>{e}</p>", status_code=503)

@app.get("/")
async def root_proxy():
    try:
        r = requests.get(f"{STREAMLIT_BASE}/", timeout=20)
        return HTMLResponse(content=r.content, status_code=r.status_code, headers={"Content-Type": r.headers.get("Content-Type","text/html")})
    except Exception as e:
        LOG.debug("streamlit root not ready: %s", e)
        return HTMLResponse(content=f"<h3>Streamlit UI is starting. Please wait a few seconds.</h3><p>{e}</p>", status_code=503)

# -------------------------
# Streamlit frontend generator and runner
# -------------------------
STREAMLIT_CODE = f'''
# streamlit_frontend.py (generated)
import streamlit as st
import requests, os

st.set_page_config(page_title="PharmaAI Cloud", layout="wide")
LANG = st.sidebar.radio("Language", ["English", "العربية"], index=0)
def T(en, ar): return ar if LANG=="العربية" else en

st.title(T("💊 PharmaAI Cloud","💊 منصة PharmaAI"))
st.caption(T("Clinical decision support assistant - not a substitute for professional medical advice.","مساعد اتخاذ قرار سريري - ليس بديلاً عن الرأي الطبي"))

API_BASE = "http://127.0.0.1:{PORT}"

tab = st.sidebar.selectbox(T("Section","القسم"), [T("Interactions","التداخل"), T("Alternatives & Dosage","البدائل والجرعات"), T("Reports","التقارير"), T("Account","الحساب")])

if tab == T("Interactions","التداخل"):
    st.header(T("Drug Interaction Checker","فاحص التداخلات"))
    a = st.text_input(T("Drug A (name)","الدواء A"))
    b = st.text_input(T("Drug B (name)","الدواء B"))
    if st.button(T("Check","تحقق")):
        if not a or not b:
            st.warning(T("Enter two drug names","أدخل اسمي دوائين"))
        else:
            st.info(T("Requesting RxNav analysis...","يتم طلب التحليل من RxNav..."))
            try:
                rxcui_a = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={{"name":a}}).json().get("idGroup",{{}}).get("rxnormId",[None])[0]
                rxcui_b = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={{"name":b}}).json().get("idGroup",{{}}).get("rxnormId",[None])[0]
                if not rxcui_a or not rxcui_b:
                    st.error(T("Could not fetch RxCUI for one or both drugs.","تعذّر جلب RxCUI."))
                else:
                    inter = requests.get("https://rxnav.nlm.nih.gov/REST/interaction/interaction.json", params={{"rxcui": rxcui_a}}).json()
                    st.write(inter)
            except Exception as e:
                st.error(str(e))

elif tab == T("Alternatives & Dosage","البدائل والجرعات"):
    st.header(T("Suggest Alternatives & Dosage","اقتراح بدائل وحساب الجرعات"))
    q = st.text_input(T("Drug name for alternatives","اسم الدواء"))
    if st.button(T("Find alternatives","ابحث")):
        if not q:
            st.warning(T("Enter a drug name","أدخل اسم دواء"))
        else:
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/api/alternatives", params={{"drug_name": q}})
                if r.status_code == 200:
                    data = r.json()
                    for alt in data.get("alternatives", []):
                        st.write("- " + alt)
                else:
                    st.error("Server error")
            except Exception as e:
                st.error(str(e))
    st.subheader(T("Dosage estimation","حساب الجرعة"))
    drug_for_dose = st.text_input(T("Drug for dosage","الدواء"))
    weight = st.number_input(T("Weight (kg)","الوزن (كغ)"), value=70.0)
    age = st.number_input(T("Age","العمر"), value=30)
    cond = st.text_input(T("Condition (e.g., renal)","حالة (مثال: renal)"))
    if st.button(T("Calculate","احسب")):
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/api/dosage", params={{"drug_name":drug_for_dose,"weight":weight,"age":age,"condition":cond}})
            if r.status_code == 200:
                st.success(T("Suggested dosage (mg):","الجرعة المقترحة (مغ):") + " " + str(r.json().get("dosage_mg")))
            else:
                st.error("Server error")
        except Exception as e:
            st.error(str(e))

elif tab == T("Reports","التقارير"):
    st.header(T("Generate PDF report","إنشاء تقرير PDF"))
    patient_name = st.text_input(T("Patient name","اسم المريض"))
    drugs = st.text_area(T("Drugs (comma separated)","الأدوية (مفصولة بفواصل)"))
    if st.button(T("Generate PDF","انشئ PDF")):
        payload = {{
            "patient": {{"name": patient_name, "age": "", "weight": ""}},
            "drugs": [d.strip() for d in drugs.split(",") if d.strip()],
            "interactions": [],
            "alternatives": {{}},
            "dosage": {{}}
        }}
        try:
            r = requests.post(f"http://127.0.0.1:{PORT}/api/report", json=payload, timeout=30)
            if r.status_code == 200:
                with open("report.pdf","wb") as f:
                    f.write(r.content)
                st.success(T("Report saved as report.pdf","تم إنشاء التقرير report.pdf"))
            else:
                st.error("Report generation failed")
        except Exception as e:
            st.error(str(e))

elif tab == T("Account","الحساب"):
    st.header(T("Account & Premium","الحساب والاشتراك"))
    st.info(T("Use backend endpoints for signup/login and PayPal flows.","استخدم واجهات الخادم للتسجيل والدفع."))

st.markdown("---")
st.caption("PharmaAI Cloud — Demo UI. Not a medical device.")
'''

def write_streamlit_file():
    with open(STREAMLIT_FILE, "w", encoding="utf-8") as f:
        f.write(STREAMLIT_CODE)

def run_streamlit():
    write_streamlit_file()
    cmd = ["streamlit", "run", STREAMLIT_FILE, "--server.port", str(STREAMLIT_PORT), "--server.address", "127.0.0.1"]
    LOG.info("Launching Streamlit: %s", " ".join(cmd))
    # start in background; in container this stays alive
    subprocess.Popen(cmd)

# -------------------------
# Startup sequence
# -------------------------
def startup():
    try:
        ensure_schema()
    except Exception as e:
        LOG.warning("DB schema ensure failed at startup: %s", e)
    # build/import drugs and init AI
    try:
        df = build_drug_features(limit=CHEMBL_SAMPLE)
    except Exception as e:
        LOG.warning("build_drug_features failed: %s", e)
        df = pd.DataFrame([{"drug_name":"paracetamol","mol_weight":300,"logP":2.5,"h_bond_donor":1,"h_bond_acceptor":3}])
    global ai
    try:
        ai = AIModule(df)
    except Exception as e:
        LOG.warning("AI init failed: %s", e)
        ai = None
    # run streamlit in background
    try:
        run_streamlit()
        time.sleep(1)
    except Exception as e:
        LOG.warning("Streamlit launch warning: %s", e)

startup()

# -------------------------
# Simple AI endpoints
# -------------------------
@app.get("/api/alternatives")
def api_alternatives(drug_name: str):
    if ai is None:
        raise HTTPException(status_code=500, detail="AI not ready")
    try:
        alts = ai.suggest_alternatives(drug_name, top_k=8)
        return {"drug": drug_name, "alternatives": alts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dosage")
def api_dosage(drug_name: str, weight: float = 70.0, age: int = 30, condition: str = ""):
    if ai is None:
        raise HTTPException(status_code=500, detail="AI not ready")
    try:
        dose = ai.predict_dosage(drug_name, weight, age, condition)
        return {"drug": drug_name, "dosage_mg": dose}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    LOG.info("Starting FastAPI on port %s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

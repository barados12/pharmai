# app.py
import os, json, csv, re, requests
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, jsonify, redirect, url_for, flash)
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional libs
import openai
from chembl_webresource_client.new_client import new_client

# -------------- Config --------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("APP_SECRET", os.urandom(24))
PORT = int(os.getenv("PORT", 5000))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required (Supabase / Railway)")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

# PayPal (sandbox by default)
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_BASE = os.getenv("PAYPAL_BASE", "https://api-m.sandbox.paypal.com")

# External APIs keys
DRUGBANK_API_KEY = os.getenv("DRUGBANK_API_KEY")  # optional, DrugBank commercial
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # optional for advanced AI
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # optional

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# -------------- Auth (Flask-Login) --------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id, username, email, role='user', telegram_chat_id=None):
        self.id = str(id)
        self.username = username
        self.email = email
        self.role = role
        self.telegram_chat_id = telegram_chat_id

@login_manager.user_loader
def load_user(uid):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, username, email, role, telegram_chat_id FROM users WHERE id=:id"), {"id": int(uid)}).fetchone()
    if row:
        return User(row.id, row.username, row.email, row.role, row.telegram_chat_id)
    return None

# -------------- DB Init --------------
def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            role TEXT DEFAULT 'user',
            telegram_chat_id TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS purchases (
            id SERIAL PRIMARY KEY,
            user_id INT REFERENCES users(id),
            order_id TEXT,
            amount NUMERIC(10,2),
            currency TEXT,
            status TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """))
init_db()

# -------------- CSV data (local) --------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DIABETES_CSV = os.path.join(DATA_DIR, "diabetes_drugs_database.csv")
ONCOLOGY_CSV = os.path.join(DATA_DIR, "oncology_drugs_database.csv")
MERGED_CSV = os.path.join(DATA_DIR, "all_drugs_database.csv")

# ensure data dir and CSVs exist (you can replace these with your originals)
os.makedirs(DATA_DIR, exist_ok=True)

def ensure_sample_csvs():
    if not os.path.exists(DIABETES_CSV):
        rows = [
            ["ميتفورمين","Metformin","خافض سكر الدم","250-500 mg مرتين يومياً","500-2000 mg يومياً","...", "سيتاجليبتين","غثيان","مراقبة وظائف الكلى"],
        ]
        with open(DIABETES_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["name_ar","name_en","classes","doses_children","doses_adults","interactions","alternatives","side_effects","precautions"])
            writer.writerows(rows)
    if not os.path.exists(ONCOLOGY_CSV):
        with open(ONCOLOGY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["name_ar","name_en","classes","doses_children","doses_adults","interactions","alternatives","side_effects","precautions"])
    # merge (simple append)
    if not os.path.exists(MERGED_CSV):
        df_list=[]
        for p in [DIABETES_CSV, ONCOLOGY_CSV]:
            try:
                df = pd.read_csv(p, encoding="utf-8-sig")
                df_list.append(df)
            except Exception:
                pass
        if df_list:
            pd.concat(df_list, ignore_index=True).to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
ensure_sample_csvs()

def load_local_drugs(csv_path=MERGED_CSV):
    df = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
    db = {}
    for _, r in df.iterrows():
        name_ar = str(r.get("name_ar","")).strip()
        db[name_ar] = r.to_dict()
    return db

LOCAL_DRUGS = load_local_drugs()

# -------------- AI local model (simple TF-IDF) --------------
class LocalDrugAI:
    def __init__(self, drugs_dict):
        self.drugs = drugs_dict
        self.names = list(drugs_dict.keys())
        docs = []
        for n in self.names:
            row = drugs_dict[n]
            text = " ".join([str(v) for k,v in row.items() if isinstance(v, str)])
            docs.append(text)
        if docs:
            self.vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
            self.matrix = self.vec.fit_transform(docs)
    def similar(self, name, top_n=5):
        if name not in self.drugs:
            possible = get_close_matches(name, self.names, n=1, cutoff=0.6)
            if possible: name = possible[0]
            else: return []
        idx = self.names.index(name)
        sims = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        top = sims.argsort()[-top_n-1:-1][::-1]
        return [self.names[i] for i in top if self.names[i] != name]

LOCAL_AI = LocalDrugAI(LOCAL_DRUGS)

# -------------- External APIs helpers --------------
# RxNav (RxNorm) simple wrappers — public endpoints. See docs. :contentReference[oaicite:4]{index=4}
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"

def rxnav_find_rxcui_by_name(name):
    url = f"{RXNAV_BASE}/rxcui.json?name={requests.utils.requote_uri(name)}"
    r = requests.get(url, timeout=10)
    if r.ok:
        j = r.json()
        return j.get("idGroup", {}).get("rxnormId", [None])[0]
    return None

def rxnav_get_properties(rxcui):
    url = f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json"
    r = requests.get(url, timeout=10)
    return r.json() if r.ok else None

# ChEMBL — official Python client used. Example: search molecule by name, then get molecule info. :contentReference[oaicite:5]{index=5}
chembl_molecule = new_client.molecule
chembl_activity = new_client.activity

def chembl_search_molecule(name, limit=5):
    res = chembl_molecule.search(name)
    out = []
    for r in res[:limit]:
        out.append({"chembl_id": r.get("molecule_chembl_id"), "pref_name": r.get("pref_name")})
    return out

def chembl_get_molecule(chembl_id):
    try:
        m = chembl_molecule.get(chembl_id)
        return m
    except Exception:
        return None

# DrugBank (optional) — requires API key. docs: dev.drugbank.com. :contentReference[oaicite:6]{index=6}
DRUGBANK_API_BASE = "https://api.drugbank.com/v1"

def drugbank_search(name):
    if not DRUGBANK_API_KEY:
        return {"error":"no_api_key"}
    headers = {"Authorization": DRUGBANK_API_KEY, "Accept":"application/json"}
    url = f"{DRUGBANK_API_BASE}/drugs?query={requests.utils.requote_uri(name)}"
    r = requests.get(url, headers=headers, timeout=10)
    if r.ok:
        return r.json()
    return {"error":"request_failed","status": r.status_code}

# -------------- PayPal helpers (same approach as before) --------------
def paypal_get_access_token():
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise RuntimeError("PayPal credentials missing")
    url = f"{PAYPAL_BASE}/v1/oauth2/token"
    r = requests.post(url, data={"grant_type":"client_credentials"}, auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET), timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]

def paypal_create_order(amount, currency="USD", metadata=None):
    token = paypal_get_access_token()
    url = f"{PAYPAL_BASE}/v2/checkout/orders"
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    body = {"intent":"CAPTURE","purchase_units":[{"amount":{"currency_code":currency,"value":str(amount)},"custom_id": json.dumps(metadata or {})}]}
    r = requests.post(url, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    return r.json()

def paypal_capture_order(order_id):
    token = paypal_get_access_token()
    url = f"{PAYPAL_BASE}/v2/checkout/orders/{order_id}/capture"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type":"application/json"}, timeout=15)
    r.raise_for_status()
    return r.json()

# -------------- Utilities --------------
def send_telegram(chat_id, text):
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        return r.ok
    except Exception:
        return False

def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated or getattr(current_user, "role", "user") != "admin":
            return ("Forbidden", 403)
        return f(*args, **kwargs)
    return wrapped

# -------------- Routes --------------
@app.route("/")
@login_required
def index():
    # sample products from local DB (or you can load products table)
    products = [
        {"id":1,"title":"تحليل دواء + تقرير PDF","description":"نظرة متقدمة على التفاعلات","price":25},
        {"id":2,"title":"استشارة AI (مميز)","description":"توليد تقارير وتوصيات","price":50},
    ]
    return render_template("index.html", products=products, paypal_client_id=PAYPAL_CLIENT_ID)

# Auth: register / login / logout
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        username = request.form.get("username","").strip()
        email = request.form.get("email","").strip()
        password = request.form.get("password","")
        telegram_id = request.form.get("telegram_chat_id","").strip() or None
        if not username or not password or not email:
            flash("الرجاء إكمال الحقول")
            return redirect(url_for("register"))
        pw = generate_password_hash(password)
        with engine.begin() as conn:
            try:
                res = conn.execute(text("INSERT INTO users(username,email,password_hash,telegram_chat_id) VALUES(:u,:e,:p,:t) RETURNING id"),
                                   {"u":username,"e":email,"p":pw,"t":telegram_id})
                uid = res.fetchone()[0]
            except IntegrityError:
                flash("اسم المستخدم أو الإيميل مستخدم")
                return redirect(url_for("register"))
        login_user(User(uid, username, email))
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        username = request.form.get("username","")
        password = request.form.get("password","")
        with engine.connect() as conn:
            row = conn.execute(text("SELECT id,username,email,password_hash,role,telegram_chat_id FROM users WHERE username=:u"), {"u":username}).fetchone()
        if row and check_password_hash(row.password_hash, password):
            login_user(User(row.id, row.username, row.email, row.role, row.telegram_chat_id))
            return redirect(url_for("index"))
        flash("خطأ في بيانات الدخول")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# AI endpoints: local + external (OpenAI)
@app.route("/api/local/similar", methods=["POST"])
@login_required
def api_local_similar():
    q = request.json.get("q","").strip()
    sims = LOCAL_AI.similar(q, top_n=6)
    return jsonify({"query": q, "similar": sims})

@app.route("/api/rxnav/search", methods=["GET"])
@login_required
def api_rxnav_search():
    q = request.args.get("q","").strip()
    if not q: return jsonify({"error":"missing q"}),400
    rxcui = rxnav_find_rxcui_by_name(q)
    props = rxnav_get_properties(rxcui) if rxcui else None
    return jsonify({"query": q, "rxcui": rxcui, "properties": props})

@app.route("/api/chembl/search", methods=["GET"])
@login_required
def api_chembl_search():
    q = request.args.get("q","").strip()
    if not q: return jsonify({"error":"missing q"}),400
    res = chembl_search_molecule(q, limit=8)
    return jsonify({"query": q, "results": res})

@app.route("/api/drugbank/search", methods=["GET"])
@login_required
def api_drugbank_search():
    q = request.args.get("q","").strip()
    if not q: return jsonify({"error":"missing q"}),400
    if not DRUGBANK_API_KEY:
        return jsonify({"error":"DrugBank API key not configured"}), 403
    res = drugbank_search(q)
    return jsonify({"query": q, "results": res})

# PayPal create & capture
@app.route("/create-order", methods=["POST"])
@login_required
def create_order_route():
    data = request.json or {}
    amount = data.get("amount")
    currency = data.get("currency","USD")
    metadata = data.get("metadata", {})
    if not amount:
        return jsonify({"error":"amount required"}), 400
    order = paypal_create_order(amount, currency, metadata)
    # store pending
    order_id = order.get("id")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO purchases(user_id,order_id,amount,currency,status,metadata) VALUES(:u,:o,:a,:c,'created',:m)"),
                     {"u": int(current_user.id), "o": order_id, "a": amount, "c": currency, "m": json.dumps(metadata)})
    return jsonify(order)

@app.route("/capture-order/<order_id>", methods=["POST"])
@login_required
def capture_order_route(order_id):
    resp = paypal_capture_order(order_id)
    with engine.begin() as conn:
        conn.execute(text("UPDATE purchases SET status='captured', metadata = COALESCE(metadata, '{}'::jsonb) || :m WHERE order_id=:o"),
                     {"o": order_id, "m": json.dumps(resp)})
    # notify user
    try:
        if current_user.telegram_chat_id:
            send_telegram(current_user.telegram_chat_id, f"تمت عملية الدفع: {order_id}")
    except Exception:
        pass
    return jsonify(resp)

# Webhook (PayPal)
@app.route("/webhook/paypal", methods=["POST"])
def paypal_webhook():
    event = request.json or {}
    # basic handling: attach event to purchase by order id if found
    try:
        res = event.get("resource", {}) or {}
        order_id = res.get("id") or res.get("order_id") or (res.get("supplementary_data") or {}).get("related_ids",{}).get("order_id")
        if order_id:
            with engine.begin() as conn:
                conn.execute(text("UPDATE purchases SET metadata = COALESCE(metadata, '{}'::jsonb) || :m, status = :s WHERE order_id=:o"),
                             {"o": order_id, "m": json.dumps(event), "s": event.get("event_type","webhook_received")})
    except Exception:
        app.logger.exception("webhook")
    return jsonify({"status":"ok"}), 200

# Dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    with engine.connect() as conn:
        users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        sales = conn.execute(text("SELECT COUNT(*) FROM purchases WHERE status='captured'")).scalar()
        revenue = conn.execute(text("SELECT COALESCE(SUM(amount),0) FROM purchases WHERE status='captured'")).scalar()
    return render_template("dashboard.html", users=users, sales=sales, revenue=revenue)

# Health
@app.route("/health")
def health():
    return jsonify({"status":"ok","ts": datetime.utcnow().isoformat()})

#if __name__=="__main__":
 #   app.run(host="0.0.0.0", port=PORT)
     


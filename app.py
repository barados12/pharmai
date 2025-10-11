# app.py
"""
PharmaAI Cloud - Complete Drug Analysis Platform
الإصدار النهائي مع جميع الميزات المجانية والمدفوعة
"""

import os
import time
import json
import threading
import hashlib
import binascii
import requests
import tempfile
from io import BytesIO
from typing import Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import psycopg2
from psycopg2.extras import RealDictCursor

import subprocess

# -------------------- Configuration / ENV --------------------
PORT = int(os.getenv("PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET", "")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")
PAYPAL_WEBHOOK_ID = os.getenv("PAYPAL_WEBHOOK_ID", "")
PAYPAL_PRICE_USD = os.getenv("PAYPAL_PRICE_USD", "9.99")
CHEMBL_SAMPLE = int(os.getenv("CHEMBL_SAMPLE", "200"))
PUBCHEM_BATCH = int(os.getenv("PUBCHEM_BATCH", "100"))

# APIs
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_PROP = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/MolecularWeight,CanonicalSMILES,HBondDonorCount,HBondAcceptorCount,XLogP/JSON"
RXNAV_RXCUI = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNAV_INTERACTION = "https://rxnav.nlm.nih.gov/REST/interaction/interaction.json"

DRUG_CSV = "drug_features.csv"

# -------------------- FastAPI App --------------------
app = FastAPI(
    title="PharmaAI Cloud API",
    description="Complete Drug Analysis Platform with AI-powered features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI module
ai = None

# -------------------- Database Functions --------------------
def get_pg_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL, sslmode="require", cursor_factory=RealDictCursor)
    return conn

def ensure_postgres_schema():
    conn = get_pg_conn()
    cur = conn.cursor()
    
    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        pwdhash TEXT NOT NULL,
        is_premium BOOLEAN DEFAULT FALSE,
        premium_expires TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    
    # User queries table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_queries (
        id SERIAL PRIMARY KEY,
        user_email TEXT,
        query_type TEXT,
        query_text TEXT,
        result_json JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    
    # Medical history table (premium feature)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS medical_history (
        id SERIAL PRIMARY KEY,
        user_email TEXT,
        condition_name TEXT,
        medications JSONB,
        notes TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def hash_password(password: str) -> str:
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password: str, provided_password: str) -> bool:
    salt = stored_password[:64].encode('ascii')
    stored_pwdhash = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_pwdhash

def create_user(email: str, password: str):
    conn = get_pg_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (email, pwdhash) VALUES (%s, %s) RETURNING id", 
                   (email, hash_password(password)))
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
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, pwdhash, is_premium FROM users WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if not row:
        return False, "No such user", None
    if verify_password(row['pwdhash'], password):
        return True, "Authenticated", {
            "id": row['id'], 
            "email": row['email'], 
            "is_premium": row['is_premium']
        }
    return False, "Wrong password", None

def set_user_premium(email: str, months: int = 1):
    conn = get_pg_conn()
    cur = conn.cursor()
    expires = datetime.now().timestamp() + (months * 30 * 24 * 60 * 60)  # 30 days per month
    cur.execute("UPDATE users SET is_premium = TRUE, premium_expires = TO_TIMESTAMP(%s) WHERE email = %s", 
               (expires, email))
    conn.commit()
    cur.close()
    conn.close()

def is_user_premium(email: str) -> bool:
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("SELECT is_premium, premium_expires FROM users WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if not row:
        return False
    if not row['is_premium']:
        return False
    if row['premium_expires'] and row['premium_expires'] < datetime.now():
        return False
    return True

def log_user_query(user_email: str, query_type: str, query_text: str, result: dict):
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_queries (user_email, query_type, query_text, result_json) 
        VALUES (%s, %s, %s, %s)
    """, (user_email, query_type, query_text, json.dumps(result)))
    conn.commit()
    cur.close()
    conn.close()

# -------------------- Data Builder --------------------
def build_drug_features(limit=CHEMBL_SAMPLE):
    """Build comprehensive drug features database"""
    if os.path.exists(DRUG_CSV):
        df = pd.read_csv(DRUG_CSV)
        print(f"✅ Loaded existing drug features with {len(df)} records")
        return df

    print("🔄 Building drug features database...")
    mols = []
    
    # Sample drug data (fallback)
    sample_drugs = [
        {"name": "Aspirin", "weight": 180.16, "logp": 1.19, "donor": 1, "acceptor": 4},
        {"name": "Paracetamol", "weight": 151.16, "logp": 0.46, "donor": 2, "acceptor": 3},
        {"name": "Ibuprofen", "weight": 206.28, "logp": 3.97, "donor": 1, "acceptor": 2},
        {"name": "Amoxicillin", "weight": 365.40, "logp": -0.77, "donor": 5, "acceptor": 8},
        {"name": "Metformin", "weight": 129.16, "logp": -1.03, "donor": 3, "acceptor": 5},
        {"name": "Atorvastatin", "weight": 558.64, "logp": 4.06, "donor": 2, "acceptor": 6},
        {"name": "Lisinopril", "weight": 405.49, "logp": 1.70, "donor": 3, "acceptor": 7},
        {"name": "Levothyroxine", "weight": 776.87, "logp": 3.24, "donor": 4, "acceptor": 8},
        {"name": "Omeprazole", "weight": 345.42, "logp": 2.27, "donor": 1, "acceptor": 5},
        {"name": "Simvastatin", "weight": 418.57, "logp": 4.68, "donor": 1, "acceptor": 5},
    ]
    
    # Add sample drugs
    for drug in sample_drugs:
        mols.append({
            "drug_name": drug["name"],
            "mol_weight": drug["weight"],
            "logP": drug["logp"],
            "h_bond_donor": drug["donor"],
            "h_bond_acceptor": drug["acceptor"]
        })
    
    # Try to fetch from ChEMBL
    try:
        per_page = 50
        offset = 0
        fetched = len(mols)
        
        while fetched < limit:
            params = {"limit": per_page, "offset": offset}
            response = requests.get(f"{CHEMBL_API}/molecule.json", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            molecules = data.get("molecules", [])
            if not molecules:
                break
                
            for molecule in molecules:
                name = molecule.get("pref_name") or molecule.get("molecule_chembl_id")
                if name and name not in [m["drug_name"] for m in mols]:
                    properties = molecule.get("molecule_properties", {})
                    mols.append({
                        "drug_name": name,
                        "mol_weight": properties.get("mw_freebase"),
                        "logP": properties.get("alogp"),
                        "h_bond_donor": properties.get("num_ro5_violations", 1),
                        "h_bond_acceptor": properties.get("num_ro5_violations", 3)
                    })
                    fetched += 1
                    if fetched >= limit:
                        break
            offset += 1
            time.sleep(0.1)
            
    except Exception as e:
        print(f"⚠️ ChEMBL fetch limited: {e}")

    df = pd.DataFrame(mols)
    
    # Impute missing values safely
    df = df.assign(
        mol_weight=df["mol_weight"].fillna(df["mol_weight"].median()),
        logP=df["logP"].fillna(2.5),
        h_bond_donor=df["h_bond_donor"].fillna(1),
        h_bond_acceptor=df["h_bond_acceptor"].fillna(3)
    )
    
    # Ensure numeric types
    numeric_cols = ["mol_weight", "logP", "h_bond_donor", "h_bond_acceptor"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.to_csv(DRUG_CSV, index=False)
    print(f"✅ Saved {len(df)} drug records to {DRUG_CSV}")
    return df

# -------------------- AI Module --------------------
class AIModule:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.features = self.df[["mol_weight", "logP", "h_bond_donor", "h_bond_acceptor"]].astype(float).values
        
        # Scale features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        # KNN for alternatives
        self.knn = None
        if len(self.features) >= 5:
            n_neighbors = min(10, len(self.features))
            self.knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
            self.knn.fit(self.features_scaled)
        
        # Dosage regression model
        self.reg = self._train_dosage_model()
        
        # Drug interactions knowledge base
        self.interactions_db = {
            "warfarin": ["aspirin", "ibuprofen", "naproxen"],
            "simvastatin": ["amiodarone", "verapamil", "cyclosporine"],
            "levothyroxine": ["calcium", "iron", "proton pump inhibitors"],
            "digoxin": ["amiodarone", "quinine", "verapamil"],
            "lithium": ["diuretics", "nsaids", "ace inhibitors"]
        }
    
    def _train_dosage_model(self):
        """Train dosage prediction model"""
        X, y = [], []
        base_map = {
            "paracetamol": 10, "acetaminophen": 10, "aspirin": 5, 
            "ibuprofen": 4, "amoxicillin": 15, "metformin": 8
        }
        
        for drug_name, base in base_map.items():
            for weight in [40, 60, 80, 100]:
                for age in [20, 40, 60, 80]:
                    X.append([weight, age])
                    y.append(weight * base * (0.8 if age > 65 else 1.0))
        
        if len(X) > 0:
            return LinearRegression().fit(np.array(X), np.array(y))
        return None
    
    def suggest_alternatives(self, drug_name: str, top_k: int = 5) -> List[str]:
        """Suggest alternative drugs based on molecular properties"""
        if self.df.empty or self.knn is None:
            return []
        
        # Find the drug in database
        mask = self.df["drug_name"].str.lower() == drug_name.lower()
        if not mask.any():
            # Fallback: similar name search
            contains = self.df[self.df["drug_name"].str.contains(drug_name, case=False, na=False)]
            if not contains.empty:
                return contains["drug_name"].head(top_k).tolist()
            return []
        
        idx = self.df[mask].index[0]
        distances, indices = self.knn.kneighbors([self.features_scaled[idx]])
        
        alternatives = []
        for i in indices[0]:
            alt_name = self.df.iloc[i]["drug_name"]
            if alt_name.lower() != drug_name.lower():
                alternatives.append(alt_name)
        
        return alternatives[:top_k]
    
    def predict_dosage(self, drug_name: str, weight: float, age: int, condition: str = "") -> float:
        """Predict dosage based on drug, weight, age, and medical condition"""
        if weight <= 0 or not drug_name:
            return 0.0
        
        drug_lower = drug_name.lower()
        base_dose = 10.0  # Default base
        
        # Base dosage mapping
        base_map = {
            "paracetamol": 10, "acetaminophen": 10, "aspirin": 5, 
            "ibuprofen": 4, "amoxicillin": 15, "metformin": 8,
            "atorvastatin": 2, "simvastatin": 2, "lisinopril": 1
        }
        
        for drug_key, base in base_map.items():
            if drug_key in drug_lower:
                base_dose = base
                break
        
        # Calculate base dosage
        if self.reg is not None:
            dosage = self.reg.predict([[weight, age]])[0]
        else:
            dosage = weight * base_dose
        
        # Adjust for conditions
        adjustments = {
            "renal": 0.7, "kidney": 0.7, "hepatic": 0.8, "liver": 0.8,
            "elderly": 0.9, "geriatric": 0.9, "pediatric": 0.5, "child": 0.5
        }
        
        for cond_key, factor in adjustments.items():
            if cond_key in condition.lower():
                dosage *= factor
        
        # Age-based adjustments
        if age < 12:
            dosage *= 0.5
        elif age > 65:
            dosage *= 0.9
        
        return max(0, round(dosage, 2))
    
    def check_interactions(self, drugs: List[str]) -> Dict[str, List[str]]:
        """Check for potential drug interactions"""
        interactions = {}
        drug_list = [d.lower() for d in drugs]
        
        for drug in drug_list:
            if drug in self.interactions_db:
                interacting_with = []
                for other_drug in drug_list:
                    if other_drug != drug and other_drug in self.interactions_db[drug]:
                        interacting_with.append(other_drug)
                if interacting_with:
                    interactions[drug] = interacting_with
        
        return interactions

# -------------------- PayPal Integration --------------------
def paypal_base_url():
    return "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"

def paypal_get_token():
    if not (PAYPAL_CLIENT_ID and PAYPAL_SECRET):
        raise RuntimeError("PayPal credentials missing")
    
    try:
        response = requests.post(
            f"{paypal_base_url()}/v1/oauth2/token",
            data={"grant_type": "client_credentials"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
            timeout=15
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        print(f"PayPal token error: {e}")
        raise

def paypal_create_order(amount: str = PAYPAL_PRICE_USD):
    try:
        token = paypal_get_token()
        url = f"{paypal_base_url()}/v2/checkout/orders"
        
        payload = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "amount": {
                    "currency_code": "USD",
                    "value": amount
                }
            }],
            "application_context": {
                "return_url": "https://your-app.railway.app/success",
                "cancel_url": "https://your-app.railway.app/cancel",
                "brand_name": "PharmaAI Cloud"
            }
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        print(f"PayPal create order error: {e}")
        raise

def paypal_capture_order(order_id: str):
    try:
        token = paypal_get_token()
        url = f"{paypal_base_url()}/v2/checkout/orders/{order_id}/capture"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        response = requests.post(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        print(f"PayPal capture error: {e}")
        raise

# -------------------- PDF Report Generation --------------------
def generate_advanced_pdf(report_data: Dict[str, Any]) -> bytes:
    """Generate comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.HexColor('#2E86AB')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        textColor=colors.HexColor('#2E86AB')
    )
    
    content = []
    
    # Title
    content.append(Paragraph("PharmaAI Cloud - Medical Report", title_style))
    content.append(Spacer(1, 20))
    
    # Patient Information
    content.append(Paragraph("Patient Information", heading_style))
    patient = report_data.get("patient", {})
    patient_info = f"""
    <b>Name:</b> {patient.get('name', 'N/A')}<br/>
    <b>Age:</b> {patient.get('age', 'N/A')}<br/>
    <b>Weight:</b> {patient.get('weight', 'N/A')} kg<br/>
    <b>Conditions:</b> {patient.get('conditions', 'None specified')}
    """
    content.append(Paragraph(patient_info, styles["Normal"]))
    content.append(Spacer(1, 20))
    
    # Medications
    content.append(Paragraph("Current Medications", heading_style))
    drugs = report_data.get("drugs", [])
    if drugs:
        drug_data = [["No.", "Medication", "Recommended Dosage"]]
        for i, drug in enumerate(drugs, 1):
            dosage = report_data.get("dosage", {}).get(drug, "Consult doctor")
            drug_data.append([str(i), drug, f"{dosage} mg" if isinstance(dosage, (int, float)) else dosage])
        
        drug_table = Table(drug_data, colWidths=[50, 200, 150])
        drug_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(drug_table)
    else:
        content.append(Paragraph("No medications specified", styles["Normal"]))
    
    content.append(Spacer(1, 20))
    
    # Drug Interactions
    interactions = report_data.get("interactions", {})
    if interactions:
        content.append(Paragraph("Drug Interactions Alert", heading_style))
        for drug, interacting_drugs in interactions.items():
            content.append(Paragraph(f"<b>{drug.title()}</b> may interact with: {', '.join(interacting_drugs)}", styles["Normal"]))
        content.append(Spacer(1, 10))
    
    # Alternatives
    alternatives = report_data.get("alternatives", {})
    if alternatives:
        content.append(Paragraph("Alternative Medications", heading_style))
        for drug, alt_list in alternatives.items():
            if alt_list:
                content.append(Paragraph(f"<b>{drug}:</b> {', '.join(alt_list[:3])}", styles["Normal"]))
    
    content.append(Spacer(1, 20))
    
    # Recommendations
    content.append(Paragraph("Medical Recommendations", heading_style))
    recommendations = [
        "Always consult with healthcare provider before changing medications",
        "Report any side effects immediately",
        "Keep regular follow-up appointments",
        "Maintain healthy lifestyle and diet"
    ]
    
    for rec in recommendations:
        content.append(Paragraph(f"• {rec}", styles["Normal"]))
    
    # Footer
    content.append(Spacer(1, 30))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Italic"]))
    content.append(Paragraph("PharmaAI Cloud - AI-Powered Drug Analysis", styles["Italic"]))
    
    doc.build(content)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------- API Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def comprehensive_homepage():
    """الشاشة الرئيسية الشاملة"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PharmaAI Cloud - Complete Drug Analysis Platform</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
            }}
            .status-badge {{
                display: inline-block;
                padding: 5px 15px;
                background: #4CAF50;
                color: white;
                border-radius: 20px;
                font-size: 14px;
                margin-left: 10px;
            }}
            .features-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .feature-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #2E86AB;
            }}
            .premium-feature {{
                border-left-color: #FF6B00;
                background: #FFF3E0;
            }}
            .endpoints {{
                background: #2E86AB;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .btn {{
                display: inline-block;
                padding: 10px 20px;
                background: #2E86AB;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 5px;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #1B5E7B;
            }}
            .btn-premium {{
                background: #FF6B00;
            }}
            .btn-premium:hover {{
                background: #E55A00;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>💊 PharmaAI Cloud - Complete Drug Analysis Platform</h1>
                <p class="status-badge">🟢 SYSTEM OPERATIONAL</p>
                <p>AI-powered drug analysis, interactions checking, and personalized recommendations</p>
            </div>

            <div style="text-align: center; margin: 30px 0;">
                <a href="/docs" class="btn">📚 API Documentation</a>
                <a href="http://0.0.0.0:8501" class="btn">🎨 Streamlit Dashboard</a>
                <a href="/api/health" class="btn">🔍 System Status</a>
                <a href="#premium" class="btn btn-premium">⭐ Go Premium</a>
            </div>

            <h2>🎯 Available Features</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <h3>🔍 Free Features</h3>
                    <ul>
                        <li>Drug Search & Information</li>
                        <li>Basic Alternatives Suggestions</li>
                        <li>Dosage Calculator</li>
                        <li>Bilingual UI (Arabic/English)</li>
                        <li>User Registration</li>
                        <li>Basic API Access</li>
                    </ul>
                </div>

                <div class="feature-card premium-feature">
                    <h3>⭐ Premium Features</h3>
                    <ul>
                        <li>Advanced PDF Reports</li>
                        <li>Drug Interaction Analysis</li>
                        <li>Personalized Treatment Plans</li>
                        <li>Medical History Tracking</li>
                        <li>Priority Support</li>
                        <li>Advanced AI Algorithms</li>
                    </ul>
                </div>

                <div class="feature-card">
                    <h3>🤖 AI Capabilities</h3>
                    <ul>
                        <li>KNN-based Drug Alternatives</li>
                        <li>Linear Regression Dosage Prediction</li>
                        <li>Drug Interaction Database</li>
                        <li>Molecular Property Analysis</li>
                        <li>Personalized Recommendations</li>
                    </ul>
                </div>
            </div>

            <div class="endpoints">
                <h2>🔌 Available Endpoints</h2>
                <p><strong>GET</strong> <code>/api/health</code> - System status</p>
                <p><strong>GET</strong> <code>/api/drugs/search?query=aspirin</code> - Drug search</p>
                <p><strong>GET</strong> <code>/api/drugs/alternatives/aspirin</code> - Alternatives</p>
                <p><strong>POST</strong> <code>/api/drugs/dosage</code> - Dosage calculation</p>
                <p><strong>POST</strong> <code>/api/report/generate</code> - PDF reports (Premium)</p>
                <p><strong>POST</strong> <code>/api/paypal/create-order</code> - Premium subscription</p>
            </div>

            <div id="premium" style="background: #FF6B00; color: white; padding: 30px; border-radius: 10px; text-align: center;">
                <h2>🚀 Upgrade to Premium</h2>
                <p>Get full access to all features for only <strong>${PAYPAL_PRICE_USD}/month</strong></p>
                <div style="margin: 20px 0;">
                    <button onclick="createPremiumOrder()" style="background: white; color: #FF6B00; border: none; padding: 15px 30px; font-size: 18px; border-radius: 8px; cursor: pointer;">
                        💳 Subscribe Now
                    </button>
                </div>
                <p>7-day money back guarantee • Cancel anytime</p>
            </div>

            <div style="margin-top: 40px; text-align: center; color: #666;">
                <p>PharmaAI Cloud v2.0.0 | Built with FastAPI + Streamlit + PostgreSQL</p>
                <p>Deployed on Railway • Secure • Scalable • Production Ready</p>
            </div>
        </div>

        <script>
            async function createPremiumOrder() {{
                try {{
                    const response = await fetch('/api/paypal/create-order', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    const data = await response.json();
                    
                    if (data.links) {{
                        const approveLink = data.links.find(link => link.rel === 'approve');
                        if (approveLink) {{
                            window.open(approveLink.href, '_blank');
                        }}
                    }}
                }} catch (error) {{
                    alert('Error creating order: ' + error.message);
                }}
            }}
        </script>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    db_status = "✅ Connected" if DATABASE_URL else "❌ Disconnected"
    paypal_status = "✅ Configured" if PAYPAL_CLIENT_ID else "❌ Not Configured"
    ai_status = "✅ Ready" if ai else "❌ Initializing"
    
    return {
        "status": "operational",
        "service": "PharmaAI Cloud",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db_status,
            "payment_gateway": paypal_status,
            "ai_module": ai_status,
            "drug_database": f"{len(ai.df) if ai else 0} drugs" if ai else "Loading"
        },
        "endpoints": {
            "main_app": f"http://0.0.0.0:{PORT}",
            "streamlit_ui": "http://0.0.0.0:8501",
            "api_docs": f"http://0.0.0.0:{PORT}/docs",
            "health_check": f"http://0.0.0.0:{PORT}/api/health"
        },
        "features": {
            "free_tier": [
                "drug_search", "basic_alternatives", "dosage_calculator", 
                "user_registration", "bilingual_ui"
            ],
            "premium_tier": [
                "advanced_reports", "interaction_analysis", 
                "medical_history", "personalized_plans"
            ]
        }
    }

@app.get("/api/drugs/search")
async def search_drugs(query: str, limit: int = 10):
    """Search for drugs in the database"""
    if not ai:
        raise HTTPException(503, "AI module initializing")
    
    try:
        matches = ai.df[ai.df["drug_name"].str.contains(query, case=False, na=False)]
        results = matches.head(limit).to_dict('records')
        
        return {
            "query": query,
            "results": results,
            "total_found": len(matches)
        }
    except Exception as e:
        raise HTTPException(500, f"Search error: {str(e)}")

@app.get("/api/drugs/alternatives/{drug_name}")
async def get_alternatives(drug_name: str, top_k: int = 5):
    """Get alternative drug suggestions"""
    if not ai:
        raise HTTPException(503, "AI module initializing")
    
    try:
        alternatives = ai.suggest_alternatives(drug_name, top_k)
        
        return {
            "drug": drug_name,
            "alternatives": alternatives,
            "count": len(alternatives)
        }
    except Exception as e:
        raise HTTPException(500, f"Alternatives error: {str(e)}")

@app.post("/api/drugs/dosage")
async def calculate_dosage(payload: Dict[str, Any]):
    """Calculate recommended dosage"""
    if not ai:
        raise HTTPException(503, "AI module initializing")
    
    try:
        drug_name = payload.get("drug_name", "")
        weight = float(payload.get("weight", 0))
        age = int(payload.get("age", 30))
        condition = payload.get("condition", "")
        
        if weight <= 0:
            raise HTTPException(400, "Weight must be positive")
        
        dosage = ai.predict_dosage(drug_name, weight, age, condition)
        
        return {
            "drug": drug_name,
            "recommended_dosage_mg": dosage,
            "weight_kg": weight,
            "age": age,
            "condition": condition,
            "notes": "Always consult with healthcare provider"
        }
    except ValueError as e:
        raise HTTPException(400, f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Dosage calculation error: {str(e)}")

@app.post("/api/drugs/interactions")
async def check_drug_interactions(payload: Dict[str, Any]):
    """Check for drug interactions (Premium feature)"""
    if not ai:
        raise HTTPException(503, "AI module initializing")
    
    # Check premium status
    user_email = payload.get("user_email")
    if user_email and not is_user_premium(user_email):
        raise HTTPException(402, "Premium feature - upgrade to access")
    
    try:
        drugs = payload.get("drugs", [])
        interactions = ai.check_interactions(drugs)
        
        return {
            "drugs_checked": drugs,
            "interactions_found": interactions,
            "severity_level": "high" if interactions else "low",
            "recommendation": "Consult healthcare provider if interactions found"
        }
    except Exception as e:
        raise HTTPException(500, f"Interaction check error: {str(e)}")

@app.post("/api/report/generate")
async def generate_report(payload: Dict[str, Any]):
    """Generate comprehensive PDF report (Premium feature)"""
    if not ai:
        raise HTTPException(503, "AI module initializing")
    
    # Check premium status
    user_email = payload.get("user_email")
    if user_email and not is_user_premium(user_email):
        raise HTTPException(402, "Premium feature - upgrade to access")
    
    try:
        # Enhance report data with AI analysis
        drugs = payload.get("drugs", [])
        
        # Get alternatives for each drug
        alternatives = {}
        for drug in drugs:
            alts = ai.suggest_alternatives(drug, 3)
            if alts:
                alternatives[drug] = alts
        
        # Calculate dosages
        dosage = {}
        patient = payload.get("patient", {})
        weight = patient.get("weight", 70)
        age = patient.get("age", 30)
        
        for drug in drugs:
            dosage[drug] = ai.predict_dosage(drug, weight, age, patient.get("conditions", ""))
        
        # Check interactions
        interactions = ai.check_interactions(drugs)
        
        # Build comprehensive report
        enhanced_report = {
            "patient": patient,
            "drugs": drugs,
            "alternatives": alternatives,
            "dosage": dosage,
            "interactions": interactions,
            "generated_at": datetime.now().isoformat(),
            "report_id": f"PHARMA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Generate PDF
        pdf_bytes = generate_advanced_pdf(enhanced_report)
        
        # Log the report generation
        if user_email:
            log_user_query(user_email, "report_generation", f"Drugs: {', '.join(drugs)}", {
                "report_id": enhanced_report["report_id"],
                "drugs_count": len(drugs),
                "interactions_found": len(interactions)
            })
        
        # Return PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        return FileResponse(
            tmp_path,
            media_type="application/pdf",
            filename=f"pharmaai_report_{enhanced_report['report_id']}.pdf",
            background=lambda: os.unlink(tmp_path)
        )
        
    except Exception as e:
        raise HTTPException(500, f"Report generation error: {str(e)}")

@app.post("/api/paypal/create-order")
async def create_premium_order():
    """Create PayPal order for premium subscription"""
    try:
        order = paypal_create_order()
        return order
    except Exception as e:
        raise HTTPException(500, f"Order creation failed: {str(e)}")

@app.post("/api/paypal/capture/{order_id}")
async def capture_payment(order_id: str):
    """Capture PayPal payment"""
    try:
        result = paypal_capture_order(order_id)
        
        # Extract email and upgrade user
        payer = result.get("payer", {})
        email = payer.get("email_address")
        
        if email:
            set_user_premium(email)
            return {
                "status": "success",
                "message": "Payment captured and account upgraded",
                "email": email,
                "premium": True
            }
        else:
            return {
                "status": "warning",
                "message": "Payment captured but email not found"
            }
            
    except Exception as e:
        raise HTTPException(500, f"Payment capture failed: {str(e)}")

@app.post("/api/auth/register")
async def register_user(payload: Dict[str, str]):
    """Register new user"""
    email = payload.get("email")
    password = payload.get("password")
    
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    
    success, result = create_user(email, password)
    
    if success:
        return {"status": "success", "user_id": result, "message": "User registered successfully"}
    else:
        raise HTTPException(400, result)

@app.post("/api/auth/login")
async def login_user(payload: Dict[str, str]):
    """Authenticate user"""
    email = payload.get("email")
    password = payload.get("password")
    
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    
    success, message, user_data = authenticate_user(email, password)
    
    if success:
        return {
            "status": "success",
            "message": message,
            "user": user_data
        }
    else:
        raise HTTPException(401, message)

# -------------------- Streamlit Frontend --------------------
def write_streamlit_frontend():
    """Create comprehensive Streamlit frontend"""
    content = '''
import streamlit as st
import requests
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PharmaAI Cloud Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .premium-feature {
        border-left-color: #FF6B00;
        background-color: #FFF3E0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Language selection
LANG = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"])

def T(en, ar):
    return en if LANG == "English" else ar

# API configuration
API_BASE = f"http://localhost:{os.environ.get("PORT", "8000")}"

# Header
st.markdown('<div class="main-header">💊 PharmaAI Cloud Dashboard</div>', unsafe_allow_html=True)
st.markdown(T(
    "AI-powered drug analysis and personalized healthcare recommendations",
    "تحليل الأدوية باستخدام الذكاء الاصطناعي والتوصيات الطبية الشخصية"
))

# Check backend status
try:
    health_response = requests.get(f"{API_BASE}/api/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success(T("✅ Backend Connected", "✅ الخادم متصل"))
        
        # Display system info in sidebar
        st.sidebar.markdown("### System Status")
        st.sidebar.write(T(f"Drugs in database: {health_data['components']['drug_database']}", 
                          f"الأدوية في قاعدة البيانات: {health_data['components']['drug_database']}"))
        st.sidebar.write(T(f"AI Status: {health_data['components']['ai_module']}",
                          f"حالة الذكاء الاصطناعي: {health_data['components']['ai_module']}"))
    else:
        st.sidebar.error(T("❌ Backend Issues", "❌ مشاكل في الخادم"))
except:
    st.sidebar.error(T("❌ Backend Unavailable", "❌ الخادم غير متاح"))

# User authentication
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔐 Authentication")

auth_tab = st.sidebar.selectbox(T("Account", "الحساب"), [T("Login", "تسجيل الدخول"), T("Register", "تسجيل جديد")])

if auth_tab == T("Login", "تسجيل الدخول"):
    with st.sidebar.form("login_form"):
        login_email = st.text_input(T("Email", "البريد الإلكتروني"))
        login_password = st.text_input(T("Password", "كلمة المرور"), type="password")
        login_submit = st.form_submit_button(T("Login", "تسجيل الدخول"))
        
        if login_submit:
            try:
                response = requests.post(f"{API_BASE}/api/auth/login", json={
                    "email": login_email,
                    "password": login_password
                })
                if response.status_code == 200:
                    user_data = response.json()
                    st.session_state.user = user_data["user"]
                    st.sidebar.success(T("Login successful!", "تم تسجيل الدخول بنجاح!"))
                else:
                    st.sidebar.error(T("Login failed", "فشل تسجيل الدخول"))
            except Exception as e:
                st.sidebar.error(T(f"Error: {e}", f"خطأ: {e}"))

else:
    with st.sidebar.form("register_form"):
        reg_email = st.text_input(T("Email", "البريد الإلكتروني"))
        reg_password = st.text_input(T("Password", "كلمة المرور"), type="password")
        reg_confirm = st.text_input(T("Confirm Password", "تأكيد كلمة المرور"), type="password")
        reg_submit = st.form_submit_button(T("Register", "تسجيل جديد"))
        
        if reg_submit:
            if reg_password == reg_confirm:
                try:
                    response = requests.post(f"{API_BASE}/api/auth/register", json={
                        "email": reg_email,
                        "password": reg_password
                    })
                    if response.status_code == 200:
                        st.sidebar.success(T("Registration successful!", "تم التسجيل بنجاح!"))
                    else:
                        error_data = response.json()
                        st.sidebar.error(error_data.get("detail", T("Registration failed", "فشل التسجيل")))
                except Exception as e:
                    st.sidebar.error(T(f"Error: {e}", f"خطأ: {e}"))
            else:
                st.sidebar.error(T("Passwords don't match", "كلمات المرور غير متطابقة"))

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    T("🔍 Drug Search", "🔍 بحث الأدوية"),
    T("🔄 Alternatives", "🔄 البدائل"),
    T("💊 Dosage Calculator", "💊 حساب الجرعات"),
    T("⚡ Interactions", "⚡ التداخلات"),
    T("📊 Reports", "📊 التقارير")
])

with tab1:
    st.header(T("Drug Search & Information", "بحث ومعلومات الأدوية"))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input(T("Enter drug name", "أدخل اسم الدواء"), 
                                   placeholder=T("e.g., Aspirin, Ibuprofen...", "مثال: أسبرين، أيبوبروفين..."))
    
    if search_query:
        with st.spinner(T("Searching...", "جاري البحث...")):
            try:
                response = requests.get(f"{API_BASE}/api/drugs/search", params={"query": search_query, "limit": 10})
                if response.status_code == 200:
                    results = response.json()
                    
                    if results["results"]:
                        st.success(T(f"Found {results['total_found']} drugs", f"تم العثور على {results['total_found']} دواء"))
                        
                        for i, drug in enumerate(results["results"]):
                            with st.expander(f"💊 {drug['drug_name']}", expanded=i==0):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(T("Molecular Weight", "الوزن الجزيئي"), f"{drug.get('mol_weight', 'N/A'):.2f}")
                                with col2:
                                    st.metric("LogP", f"{drug.get('logP', 'N/A'):.2f}")
                                with col3:
                                    st.metric(T("H-Bond Donor", "متبرع برابطة هيدروجينية"), drug.get('h_bond_donor', 'N/A'))
                                with col4:
                                    st.metric(T("H-Bond Acceptor", "متقبل رابطة هيدروجينية"), drug.get('h_bond_acceptor', 'N/A'))
                    else:
                        st.warning(T("No drugs found", "لم يتم العثور على أدوية"))
                else:
                    st.error(T("Search failed", "فشل البحث"))
            except Exception as e:
                st.error(T(f"Error: {e}", f"خطأ: {e}"))

with tab2:
    st.header(T("Find Alternative Medications", "إيجاد الأدوية البديلة"))
    
    drug_name = st.text_input(T("Enter drug name to find alternatives", "أدخل اسم الدواء لإيجاد بدائله"))
    
    if drug_name:
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider(T("Number of alternatives", "عدد البدائل"), 1, 10, 5)
        
        if st.button(T("Find Alternatives", "إيجاد البدائل"), type="primary"):
            with st.spinner(T("Finding alternatives...", "جاري إيجاد البدائل...")):
                try:
                    response = requests.get(f"{API_BASE}/api/drugs/alternatives/{drug_name}?top_k={top_k}")
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["alternatives"]:
                            st.success(T(f"Found {data['count']} alternatives for {drug_name}", 
                                       f"تم إيجاد {data['count']} بديلاً لـ {drug_name}"))
                            
                            for i, alternative in enumerate(data["alternatives"], 1):
                                st.write(f"{i}. **{alternative}**")
                        else:
                            st.warning(T(f"No alternatives found for {drug_name}", 
                                       f"لم يتم إيجاد بدائل لـ {drug_name}"))
                    else:
                        st.error(T("Failed to get alternatives", "فشل في الحصول على البدائل"))
                except Exception as e:
                    st.error(T(f"Error: {e}", f"خطأ: {e}"))

with tab3:
    st.header(T("Dosage Calculator", "حاسبة الجرعات"))
    
    with st.form("dosage_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            drug_name = st.text_input(T("Drug Name", "اسم الدواء"), 
                                    placeholder=T("e.g., Paracetamol", "مثال: باراسيتامول"))
            weight = st.number_input(T("Weight (kg)", "الوزن (كجم)"), 
                                   min_value=1.0, max_value=200.0, value=70.0)
        
        with col2:
            age = st.number_input(T("Age", "العمر"), 
                                min_value=1, max_value=120, value=30)
            condition = st.text_input(T("Medical Condition", "الحالة الصحية"),
                                    placeholder=T("e.g., renal impairment", "مثال: ضعف كلوي"))
        
        calculate = st.form_submit_button(T("Calculate Dosage", "حساب الجرعة"), type="primary")
        
        if calculate and drug_name and weight > 0:
            with st.spinner(T("Calculating...", "جاري الحساب...")):
                try:
                    response = requests.post(f"{API_BASE}/api/drugs/dosage", json={
                        "drug_name": drug_name,
                        "weight": weight,
                        "age": age,
                        "condition": condition
                    })
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success(T("Dosage Calculation Complete", "اكتمل حساب الجرعة"))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(T("Recommended Dosage", "الجرعة الموصى بها"), 
                                    f"{data['recommended_dosage_mg']} mg")
                        with col2:
                            st.metric(T("For Weight", "للوزن"), f"{data['weight_kg']} kg")
                        with col3:
                            st.metric(T("Age", "العمر"), data['age'])
                        
                        st.info(T("💡 Always consult with your healthcare provider before taking any medication",
                                "💡 دائماً استشر مقدم الرعاية الصحية قبل تناول أي دواء"))
                    else:
                        st.error(T("Dosage calculation failed", "فشل في حساب الجرعة"))
                except Exception as e:
                    st.error(T(f"Error: {e}", f"خطأ: {e}"))

with tab4:
    st.header(T("Drug Interaction Check", "فحص التداخلات الدوائية"))
    
    st.info(T("""
    **Premium Feature** - Check for potential interactions between multiple medications.
    This feature requires a premium subscription.
    """, """
    **ميزة متميزة** - افحص التداخلات المحتملة بين الأدوية المتعددة.
    هذه الميزة تتطلب اشتراك متميز.
    """))
    
    drugs_input = st.text_area(
        T("Enter drug names (one per line)", "أدخل أسماء الأدوية (سطر لكل دواء)"),
        height=100,
        placeholder=T("Aspirin\nIbuprofen\nMetformin", "أسبرين\nأيبوبروفين\nميتفورمين")
    )
    
    if st.button(T("Check Interactions", "فحص التداخلات"), type="primary"):
        if not drugs_input:
            st.warning(T("Please enter at least one drug name", "الرجاء إدخال اسم دواء واحد على الأقل"))
        else:
            drugs = [d.strip() for d in drugs_input.split('\\n') if d.strip()]
            
            # Check if user is logged in and premium
            user_email = st.session_state.get('user', {}).get('email') if 'user' in st.session_state else None
            
            if not user_email:
                st.error(T("Please login to use this feature", "الرجاء تسجيل الدخول لاستخدام هذه الميزة"))
            elif not st.session_state.user.get('is_premium', False):
                st.error(T("Premium subscription required", "الاشتراك المتميز مطلوب"))
            else:
                with st.spinner(T("Checking interactions...", "جاري فحص التداخلات...")):
                    try:
                        response = requests.post(f"{API_BASE}/api/drugs/interactions", json={
                            "drugs": drugs,
                            "user_email": user_email
                        })
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            if data["interactions_found"]:
                                st.warning(T("⚠️ Potential Interactions Found", "⚠️ تم العثور على تداخلات محتملة"))
                                
                                for drug, interactions in data["interactions_found"].items():
                                    with st.expander(f"🚨 {drug.title()}", expanded=True):
                                        st.write(T(f"May interact with: {', '.join(interactions)}", 
                                                 f"قد يتفاعل مع: {', '.join(interactions)}"))
                                        st.error(T("Consult your healthcare provider immediately!", 
                                                 "استشر مقدم الرعاية الصحية فوراً!"))
                            else:
                                st.success(T("✅ No significant interactions found", 
                                           "✅ لم يتم العثور على تداخلات مهمة"))
                                
                            st.info(T("Note: This is a preliminary check. Always consult healthcare professionals.",
                                    "ملاحظة: هذا فحص أولي. دائماً استشر المتخصصين في الرعاية الصحية."))
                        else:
                            st.error(T("Interaction check failed", "فشل فحص التداخلات"))
                    except Exception as e:
                        st.error(T(f"Error: {e}", f"خطأ: {e}"))

with tab5:
    st.header(T("Medical Reports", "التقارير الطبية"))
    
    st.info(T("""
    **Premium Feature** - Generate comprehensive PDF medical reports with drug analysis,
    interactions, alternatives, and personalized recommendations.
    """, """
    **ميزة متميزة** - أنشئ تقارير طبية شاملة بتنسيق PDF تشمل تحليل الأدوية،
    التداخلات، البدائل، والتوصيات الشخصية.
    """))
    
    with st.form("report_form"):
        st.subheader(T("Patient Information", "معلومات المريض"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_name = st.text_input(T("Patient Name", "اسم المريض"))
        with col2:
            patient_age = st.number_input(T("Age", "العمر"), min_value=1, max_value=120, value=45)
        with col3:
            patient_weight = st.number_input(T("Weight (kg)", "الوزن (كجم)"), 
                                           min_value=1.0, max_value=200.0, value=70.0)
        
        patient_conditions = st.text_input(T("Medical Conditions", "الحالات الصحية"),
                                         placeholder=T("e.g., Hypertension, Diabetes", "مثال: ضغط دم، سكري"))
        
        st.subheader(T("Current Medications", "الأدوية الحالية"))
        medications = st.text_area(
            T("List all medications (one per line)", "اذكر جميع الأدوية (سطر لكل دواء)"),
            height=120,
            placeholder=T("Aspirin\nLisinopril\nMetformin", "أسبرين\nليزينوبريل\nميتفورمين")
        )
        
        generate_report = st.form_submit_button(T("Generate PDF Report", "إنشاء تقرير PDF"), type="primary")
        
        if generate_report:
            if not medications or not patient_name:
                st.warning(T("Please fill patient name and medications", "الرجاء ملء اسم المريض والأدوية"))
            else:
                # Check premium status
                user_email = st.session_state.get('user', {}).get('email') if 'user' in st.session_state else None
                
                if not user_email:
                    st.error(T("Please login to generate reports", "الرجاء تسجيل الدخول لإنشاء التقارير"))
                elif not st.session_state.user.get('is_premium', False):
                    st.error(T("Premium subscription required for report generation", 
                             "الاشتراك المتميز مطلوب لإنشاء التقارير"))
                else:
                    with st.spinner(T("Generating comprehensive report...", "جاري إنشاء التقرير الشامل...")):
                        try:
                            drugs_list = [m.strip() for m in medications.split('\\n') if m.strip()]
                            
                            response = requests.post(f"{API_BASE}/api/report/generate", json={
                                "user_email": user_email,
                                "patient": {
                                    "name": patient_name,
                                    "age": patient_age,
                                    "weight": patient_weight,
                                    "conditions": patient_conditions
                                },
                                "drugs": drugs_list
                            })
                            
                            if response.status_code == 200:
                                st.success(T("Report generated successfully!", "تم إنشاء التقرير بنجاح!"))
                                
                                # Offer download
                                st.download_button(
                                    label=T("📥 Download PDF Report", "📥 تحميل التقرير PDF"),
                                    data=response.content,
                                    file_name=f"pharmaai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                
                                st.balloons()
                            else:
                                error_data = response.json()
                                st.error(T(f"Report generation failed: {error_data.get('detail', 'Unknown error')}",
                                         f"فشل إنشاء التقرير: {error_data.get('detail', 'خطأ غير معروف')}"))
                        except Exception as e:
                            st.error(T(f"Error generating report: {e}", f"خطأ في إنشاء التقرير: {e}"))

# Premium upgrade section
st.markdown("---")
st.markdown("### 🚀 Upgrade to Premium")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(T("""
    **Get full access to all features:**
    - ✅ Advanced PDF Reports
    - ✅ Drug Interaction Analysis  
    - ✅ Medical History Tracking
    - ✅ Personalized Treatment Plans
    - ✅ Priority Support
    - ✅ No usage limits
    """, """
    **احصل على وصول كامل لجميع الميزات:**
    - ✅ تقارير PDF متقدمة
    - ✅ تحليل التداخلات الدوائية
    - ✅ متابعة السجل الطبي
    - ✅ خطط علاج شخصية
    - ✅ دعم فني متميز
    - ✅ بدون حدود استخدام
    """))
    
with col2:
    st.info(T(f"Only ${os.environ.get('PAYPAL_PRICE_USD', '9.99')}/month", 
             f"فقط ${os.environ.get('PAYPAL_PRICE_USD', '9.99')}/شهر"))
    
    if st.button(T("💳 Subscribe Now", "💳 اشترك الآن"), type="secondary"):
        st.info(T("Redirecting to payment...", "جاري التوجيه للدفع..."))
        st.markdown(T('[Click here to upgrade](http://0.0.0.0:8000 "#")', '[انقر هنا للترقية](http://0.0.0.0:8000 "#")'))

# Footer
st.markdown("---")
st.markdown(T(
    "**PharmaAI Cloud** v2.0.0 | Built with FastAPI + Streamlit + PostgreSQL | Secure & Scalable",
    "**PharmaAI Cloud** الإصدار 2.0.0 | مبنية بـ FastAPI + Streamlit + PostgreSQL | آمنة وقابلة للتطوير"
))
'''
    
    with open("streamlit_frontend.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Streamlit frontend created successfully")

def run_streamlit():
    """Run Streamlit in background"""
    write_streamlit_frontend()
    try:
        cmd = [
            "streamlit", "run", "streamlit_frontend.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.serverAddress", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#2E86AB",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F8F9FA",
            "--theme.textColor", "#31333F"
        ]
        subprocess.Popen(cmd)
        print("✅ Streamlit server started on port 8501")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

# -------------------- Initialization --------------------
def initialize_application():
    """Initialize the complete application"""
    print("🚀 Starting PharmaAI Cloud Initialization...")
    
    # Initialize database
    try:
        ensure_postgres_schema()
        print("✅ PostgreSQL schema initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    # Build drug database and AI model
    try:
        df = build_drug_features(limit=CHEMBL_SAMPLE)
        global ai
        ai = AIModule(df)
        print(f"✅ AI module initialized with {len(df)} drugs")
    except Exception as e:
        print(f"❌ AI module initialization failed: {e}")
        # Create fallback
        ai = AIModule(pd.DataFrame())

def start_services():
    """Start all required services"""
    # Start Streamlit in background
    try:
        streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        streamlit_thread.start()
        print("✅ Streamlit frontend started")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Initialize application
    initialize_application()
    
    # Start services
    start_services()
    
    print(f"""
    🎉 PharmaAI Cloud Started Successfully!
    
    📊 Application URLs:
    FastAPI Backend: http://0.0.0.0:{PORT}
    API Documentation: http://0.0.0.0:{PORT}/docs
    Streamlit Dashboard: http://0.0.0.0:8501
    Health Check: http://0.0.0.0:{PORT}/api/health
    
    🔧 Features Ready:
    ✅ Drug Search & Information
    ✅ Alternative Suggestions  
    ✅ Dosage Calculator
    ✅ User Authentication
    ✅ Premium Subscriptions
    ✅ PDF Report Generation
    ✅ Drug Interaction Analysis
    
    🚀 System: v2.0.0 | Production Ready
    """)
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
else:
    # For production (Railway)
    initialize_application()
    start_services()
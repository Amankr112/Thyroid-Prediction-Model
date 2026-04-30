"""
HypoDetect AI v3 — Top-notch Streamlit Application
"""
import os, io, warnings, logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE_DIR, "artifacts")

# ── Clinical reference data ────────────────────────────────────────────────────
HORMONE_REF = {
    "TSH":  {"low":0.4,  "high":4.0,   "unit":"μIU/mL",  "crit_low":0.01, "crit_high":100.0,
             "pop_normal":1.89, "pop_hypo":39.23, "desc":"Thyroid Stimulating Hormone"},
    "T3":   {"low":1.1,  "high":2.8,   "unit":"nmol/L",  "crit_low":0.3,  "crit_high":8.0,
             "pop_normal":2.06, "pop_hypo":1.47,  "desc":"Triiodothyronine"},
    "TT4":  {"low":60.0, "high":150.0, "unit":"nmol/L",  "crit_low":20.0, "crit_high":300.0,
             "pop_normal":111.4,"pop_hypo":72.9,  "desc":"Total Thyroxine"},
    "T4U":  {"low":0.7,  "high":1.3,   "unit":"fraction","crit_low":0.3,  "crit_high":2.5,
             "pop_normal":0.99, "pop_hypo":1.01,  "desc":"Thyroxine Uptake"},
    "FTI":  {"low":60.0, "high":150.0, "unit":"index",   "crit_low":20.0, "crit_high":300.0,
             "pop_normal":113.6,"pop_hypo":73.2,  "desc":"Free Thyroxine Index"},
}

CLINICAL_FLAGS = {
    "on thyroxine":              ("💊","On Thyroxine",        "high"),
    "on antithyroid medication": ("💊","Antithyroid Meds",    "medium"),
    "I131 treatment":            ("☢️", "I131 Treatment",      "high"),
    "thyroid surgery":           ("🏥","Thyroid Surgery",     "high"),
    "lithium":                   ("⚗️", "Lithium Use",         "high"),
    "goitre":                    ("🦋","Goitre",              "medium"),
    "tumor":                     ("⚠️", "Tumor",               "high"),
    "hypopituitary":             ("🧠","Hypopituitary",       "high"),
    "sick":                      ("🤒","Currently Sick",      "medium"),
    "pregnant":                  ("🤰","Pregnant",            "medium"),
    "psych":                     ("🧘","Psychiatric Disorder","low"),
    "query hypothyroid":         ("🔍","Query Hypothyroid",   "medium"),
    "query hyperthyroid":        ("🔍","Query Hyperthyroid",  "medium"),
}

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HypoDetect AI · Thyroid Intelligence Platform",
    page_icon="🦋", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body { font-family: 'Inter', sans-serif !important; }

/* ── Global ── */
.main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }
[data-testid="stAppViewContainer"] { background: #f0f4f8; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2d4f 60%, #0d1b2a 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 10px 14px;
    font-size: .88rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
    display: flex !important; align-items: center; gap: 8px;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,179,237,0.18) !important;
    border-color: rgba(99,179,237,0.4) !important;
    transform: translateX(3px);
}

/* ── Page header ── */
.page-hero {
    background: linear-gradient(135deg, #dbeafe 0%, #ede9fe 50%, #ddd6fe 100%);
    border: 1px solid #bfdbfe;
    border-radius: 18px; padding: 28px 36px; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.12);
    position: relative; overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute; top: -50%; right: -10%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.page-hero h1 { color: #1e1b4b; font-size: 1.9rem; font-weight: 800; margin: 0 0 6px 0; }
.page-hero p  { color: #3730a3; margin: 0; font-size: .95rem; }

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: 16px; padding: 22px 24px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.glass-card h4 { margin: 0 0 14px 0; font-size: 1rem; font-weight: 700; color: #1a202c; }

.result-hypo {
    background: linear-gradient(135deg, #fff5f5 0%, #ffe4e4 100%);
    border: 2px solid #fc8181; border-radius: 18px; padding: 28px 32px;
    box-shadow: 0 8px 40px rgba(229,62,62,0.18); margin-bottom: 20px;
}
.result-normal {
    background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
    border: 2px solid #68d391; border-radius: 18px; padding: 28px 32px;
    box-shadow: 0 8px 40px rgba(72,187,120,0.18); margin-bottom: 20px;
}
.result-title { font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; }
.result-meta  { font-size: .88rem; color: #4a5568; display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }
.result-meta span::before { content: '•'; margin-right: 6px; opacity: .5; }
.result-meta span:first-child::before { content: ''; margin: 0; }

/* ── Score pill ── */
.score-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: .9rem;
}
.score-high   { background:#fed7d7; color:#742a2a; border:1px solid #fc8181; }
.score-medium { background:#fefcbf; color:#744210; border:1px solid #f6e05e; }
.score-low    { background:#c6f6d5; color:#22543d; border:1px solid #68d391; }

/* ── Hormone status bars ── */
.h-bar-wrap {
    background: #edf2f7; border-radius: 8px; height: 22px;
    position: relative; overflow: hidden; margin: 2px 0;
}
.h-bar-zone { position: absolute; height: 100%; opacity: 0.35; }
.h-bar-marker {
    position: absolute; top: 2px; height: calc(100% - 4px); width: 4px;
    border-radius: 3px; transform: translateX(-2px);
}
.h-bar-label { font-size: .75rem; font-weight: 600; color: #4a5568; }

/* ── Clinical flag chips ── */
.flag-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px; font-size: .78rem; font-weight: 500;
    margin: 3px; border: 1px solid transparent;
}
.flag-active-high   { background:#fed7d7; color:#742a2a; border-color:#fc8181; }
.flag-active-medium { background:#fefcbf; color:#744210; border-color:#f6e05e; }
.flag-active-low    { background:#ebf8ff; color:#2c5282; border-color:#90cdf4; }
.flag-inactive      { background:#f7fafc; color:#a0aec0; border-color:#e2e8f0; }

/* ── Recommendation items ── */
.rec-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;
    border-left: 4px solid transparent;
}
.rec-urgent   { background:#fff5f5; border-color:#e53e3e; }
.rec-warning  { background:#fffbeb; border-color:#ed8936; }
.rec-info     { background:#ebf8ff; border-color:#4299e1; }
.rec-success  { background:#f0fff4; border-color:#48bb78; }
.rec-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.rec-text { font-size: .88rem; color: #2d3748; line-height: 1.5; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
div[data-testid="stMetricLabel"] p { font-size: .78rem !important; color: #718096 !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .5px !important; }
div[data-testid="stMetricValue"]   { font-size: 1.65rem !important; font-weight: 800 !important; color: #1a202c !important; }
div[data-testid="stMetricDelta"]   { font-size: .8rem !important; }

/* ── Section divider ── */
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #1a202c;
    border-left: 4px solid #667eea; padding-left: 12px;
    margin: 24px 0 14px 0; letter-spacing: -.2px;
}

/* ── Model badge ── */
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #4338ca;
    color: #ffffff; border-radius: 20px; padding: 4px 14px;
    font-size: .78rem; font-weight: 600;
}

/* ── Form ── */
div[data-testid="stFormSubmitButton"] button {
    background: #4338ca !important;
    color: #ffffff !important; font-weight: 700 !important; font-size: 1rem !important;
    border: none !important; border-radius: 12px !important; padding: 14px !important;
    box-shadow: 0 6px 20px rgba(67,56,202,0.35) !important;
    letter-spacing: .3px !important; transition: all 0.2s !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #3730a3 !important;
    box-shadow: 0 8px 28px rgba(67,56,202,0.45) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: .88rem !important; font-weight: 600 !important;
    padding: 10px 22px !important; border-radius: 8px 8px 0 0 !important;
}

/* ── Table ── */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    border-radius: 12px !important; border: 2px dashed #cbd5e0 !important;
    background: rgba(255,255,255,0.8) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a0aec0; }

/* ── Compare table ── */
.compare-row { display:flex; gap:12px; align-items:stretch; margin-bottom:12px; }
.compare-cell {
    flex:1; background:white; border:1px solid #e2e8f0; border-radius:12px;
    padding:14px 16px; text-align:center;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.compare-cell.best { border-color:#667eea; box-shadow:0 0 0 2px rgba(102,126,234,0.2); }
.compare-val { font-size:1.3rem; font-weight:800; color:#1a202c; }
.compare-lbl { font-size:.7rem; color:#718096; font-weight:600; text-transform:uppercase; letter-spacing:.4px; margin-top:2px; }

/* ── Gradient text ── */
.grad-text {
    color: #4338ca;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Artifact loader
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    pre_art    = joblib.load(os.path.join(ART_DIR, "preprocess_artifacts.pkl"))
    scaler     = joblib.load(os.path.join(ART_DIR, "scaler.pkl"))
    meta       = joblib.load(os.path.join(ART_DIR, "best_model_meta.pkl"))
    xgb        = joblib.load(os.path.join(ART_DIR, "xgb_model.pkl"))
    rf         = joblib.load(os.path.join(ART_DIR, "rf_model.pkl"))
    lr         = joblib.load(os.path.join(ART_DIR, "lr_model.pkl"))
    shap_exp   = joblib.load(os.path.join(ART_DIR, "shap_explainer.pkl"))
    comparison = pd.read_csv(os.path.join(ART_DIR, "model_comparison.csv"), index_col=0)
    if meta["type"] == "keras":
        from tensorflow import keras
        best = keras.models.load_model(os.path.join(ART_DIR, "best_model.keras"))
    else:
        best = joblib.load(os.path.join(ART_DIR, "best_model.pkl"))
    return pre_art, scaler, meta, best, xgb, rf, lr, shap_exp, comparison


# ══════════════════════════════════════════════════════════════════════════════
# Prediction core
# ══════════════════════════════════════════════════════════════════════════════
def _run_model(model, X, X_scaled, is_keras=False):
    if is_keras:
        return float(model.predict(X_scaled, verbose=0).ravel()[0])
    elif hasattr(model, "predict_proba"):
        inp = X_scaled if isinstance(model, type(None)) else X
        try: return float(model.predict_proba(X)[:, 1][0])
        except: return float(model.predict_proba(X_scaled)[:, 1][0])
    else:
        d = model.decision_function(X_scaled)
        return float((d - d.min()) / (d.max() - d.min() + 1e-9))

def preprocess_input(raw_dict, pre_art):
    from preprocessing import preprocess
    from feature_engineering import engineer_features
    df = pd.DataFrame([raw_dict])
    X, _, _ = preprocess(df, fit=False, artifacts=pre_art)
    X = engineer_features(X)
    return X

def predict_all_models(X, scaler, rf, xgb, lr, best, meta):
    X_sc = scaler.transform(X)
    probs = {}
    try: probs["Random Forest"]        = float(rf.predict_proba(X)[:, 1][0])
    except: pass
    try: probs["XGBoost"]              = float(xgb.predict_proba(X)[:, 1][0])
    except: pass
    try: probs["Logistic Regression"]  = float(lr.predict_proba(X_sc)[:, 1][0])
    except: pass
    # best model
    m = meta["name"]
    if m not in probs:
        if meta["type"] == "keras":
            probs[m] = float(best.predict(X_sc, verbose=0).ravel()[0])
        elif hasattr(best, "predict_proba"):
            try: probs[m] = float(best.predict_proba(X)[:, 1][0])
            except: probs[m] = float(best.predict_proba(X_sc)[:, 1][0])
        else:
            d = best.decision_function(X_sc)
            probs[m] = float((d - d.min()) / (d.max() - d.min() + 1e-9))
    return probs

def predict_batch(df_raw, pre_art, scaler, meta, best):
    from preprocessing import preprocess
    from feature_engineering import engineer_features
    X, y, _ = preprocess(df_raw, fit=False, artifacts=pre_art)
    X = engineer_features(X)
    X_sc = scaler.transform(X)
    need_sc = meta["name"] in ("Logistic Regression", "SVM", "Deep Learning")
    X_in = X_sc if need_sc else X.values
    if meta["type"] == "keras":
        probs = best.predict(X_in, verbose=0).ravel()
    elif hasattr(best, "predict_proba"):
        probs = best.predict_proba(X_in)[:, 1]
    else:
        d = best.decision_function(X_in)
        probs = (d - d.min()) / (d.max() - d.min() + 1e-9)
    preds = (probs > 0.5).astype(int)
    out = df_raw.copy()
    out["Prediction"]  = ["Hypothyroid" if p else "Normal" for p in preds]
    out["Probability"] = np.round(probs, 4)
    out["Risk Level"]  = out["Probability"].apply(risk_label)
    return out, y


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════
def risk_label(p):
    return "High" if p >= 0.65 else ("Moderate" if p >= 0.30 else "Low")

def risk_color(p):
    return "#e53e3e" if p >= 0.65 else ("#ed8936" if p >= 0.30 else "#48bb78")

def risk_bg(p):
    return "#fff5f5" if p >= 0.65 else ("#fffbeb" if p >= 0.30 else "#f0fff4")

def score_pill_cls(p):
    return "score-high" if p >= 0.65 else ("score-medium" if p >= 0.30 else "score-low")

def hormone_status(val, ref_key):
    r = HORMONE_REF[ref_key]
    if val < r["low"]:   return "↓ Low",    "#ed8936"
    elif val > r["high"]:return "↑ High",   "#e53e3e"
    else:                return "✓ Normal", "#48bb78"


# ══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════════════
def build_gauge(prob):
    col = risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix":"%","font":{"size":52,"color":col,"family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#718096",
                    "tickvals":[0,25,50,75,100],"tickfont":{"size":11}},
            "bar":{"color":col,"thickness":0.3},
            "bgcolor":"white","borderwidth":0,
            "steps":[
                {"range":[0,30],  "color":"rgba(72,187,120,0.12)"},
                {"range":[30,65], "color":"rgba(237,137,54,0.12)"},
                {"range":[65,100],"color":"rgba(229,62,62,0.12)"},
            ],
            "threshold":{"line":{"color":"#2d3748","width":3},"thickness":0.85,"value":50},
        },
        title={"text":"<b>Hypothyroid Risk Score</b>","font":{"size":14,"color":"#718096","family":"Inter"}},
    ))
    fig.update_layout(
        height=260, margin=dict(l=30,r=30,t=60,b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"},
    )
    return fig


def build_hormone_radar(vals_patient, measured):
    """Radar chart: patient vs normal-mean vs hypo-mean."""
    hormones = ["TSH","T3","TT4","T4U","FTI"]
    # Normalise each to 0-1 within crit range
    def norm(v, k):
        r = HORMONE_REF[k]
        return min(1.0, max(0.0, (v - r["crit_low"]) / (r["crit_high"] - r["crit_low"] + 1e-9)))

    p_vals = [norm(vals_patient.get(h, HORMONE_REF[h]["pop_normal"]) if measured.get(h, True) else HORMONE_REF[h]["pop_normal"], h) for h in hormones]
    n_vals = [norm(HORMONE_REF[h]["pop_normal"], h) for h in hormones]
    y_vals = [norm(HORMONE_REF[h]["pop_hypo"],   h) for h in hormones]

    fig = go.Figure()
    for vals, name, color, fill in [
        (n_vals, "Normal Population",    "#48bb78", "rgba(72,187,120,0.1)"),
        (y_vals, "Hypothyroid Population","#e53e3e","rgba(229,62,62,0.1)"),
        (p_vals, "This Patient",         "#667eea", "rgba(102,126,234,0.2)"),
    ]:
        v2 = vals + [vals[0]]
        l2 = hormones + [hormones[0]]
        fig.add_trace(go.Scatterpolar(
            r=v2, theta=l2, fill="toself", fillcolor=fill,
            line=dict(color=color, width=2.5),
            name=name,
            hovertemplate="<b>%{theta}</b>: %{r:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickformat=".0%",
                           gridcolor="rgba(0,0,0,0.08)", tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=12, color="#2d3748")),
            bgcolor="rgba(248,250,252,0.5)",
        ),
        showlegend=True, legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                                     font=dict(size=11)),
        height=340, margin=dict(l=40,r=40,t=20,b=60),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        title=dict(text="Hormone Profile vs Population", font=dict(size=13, color="#4a5568"), x=0.5),
    )
    return fig


def build_hormone_bars(vals, measured):
    """Beautiful horizontal bar chart for hormone levels."""
    hormones = ["TSH","T3","TT4","T4U","FTI"]
    fig = go.Figure()
    y_labels, bar_vals, bar_colors, hover_texts, annotations = [], [], [], [], []

    for h in hormones:
        r = HORMONE_REF[h]
        v = vals.get(h, r["pop_normal"])
        meas = measured.get(h, True)
        pct = min(100, max(0, (v - 0) / (r["crit_high"] * 1.1 + 1e-9) * 100))
        if not meas:
            bar_colors.append("#cbd5e0")
            hover_texts.append(f"<b>{h}</b>: Not measured")
        elif v < r["low"]:
            bar_colors.append("#ed8936")
            hover_texts.append(f"<b>{h}</b>: {v:.2f} {r['unit']}<br>↓ Below normal ({r['low']}–{r['high']})")
        elif v > r["high"]:
            bar_colors.append("#e53e3e")
            hover_texts.append(f"<b>{h}</b>: {v:.2f} {r['unit']}<br>↑ Above normal ({r['low']}–{r['high']})")
        else:
            bar_colors.append("#48bb78")
            hover_texts.append(f"<b>{h}</b>: {v:.2f} {r['unit']}<br>✓ Normal range ({r['low']}–{r['high']})")

        y_labels.append(f"{h}<br><sub>{r['desc']}</sub>")
        bar_vals.append(pct)

    fig.add_trace(go.Bar(
        y=y_labels, x=bar_vals, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0),
                    cornerradius=6),
        width=0.55,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))
    # Normal range shading (as shapes)
    shapes = []
    for i, h in enumerate(hormones):
        r = HORMONE_REF[h]
        lo_pct = r["low"] / (r["crit_high"] * 1.1 + 1e-9) * 100
        hi_pct = r["high"]/ (r["crit_high"] * 1.1 + 1e-9) * 100
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=lo_pct, x1=hi_pct, y0=i-0.3, y1=i+0.3,
            fillcolor="rgba(72,187,120,0.12)", line=dict(width=0),
            layer="below",
        ))
    fig.update_layout(
        shapes=shapes, height=300,
        xaxis=dict(title="% of critical range", range=[0,105],
                   gridcolor="rgba(0,0,0,0.06)", tickformat=".0f",
                   ticksuffix="%", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=10,r=10,t=10,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter", size=11),
        showlegend=False,
    )
    return fig


def build_shap_chart(shap_exp, X_proc):
    try:
        sv = shap_exp.shap_values(X_proc)
        shap_arr = sv[0] if isinstance(sv, list) else sv[0]
        feat_names = X_proc.columns.tolist()
        pairs = sorted(zip(feat_names, shap_arr), key=lambda x: abs(x[1]), reverse=True)[:18]
        names  = [p[0] for p in pairs][::-1]
        values = [p[1] for p in pairs][::-1]
        colors = ["rgba(229,62,62,0.85)" if v > 0 else "rgba(72,187,120,0.85)" for v in values]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values, y=names, orientation="h",
            marker=dict(color=colors, line=dict(width=0), cornerradius=4),
            text=[f"{'+'if v>0 else ''}{v:.4f}" for v in values],
            textposition="outside", textfont=dict(size=9.5, color="#4a5568"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            width=0.65,
        ))
        fig.add_vline(x=0, line_color="#2d3748", line_width=1.5,
                      line_dash="dot", opacity=0.6)
        fig.update_layout(
            height=500,
            xaxis=dict(title="SHAP value (impact on prediction)",
                       gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                       tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10.5)),
            margin=dict(l=10,r=50,t=10,b=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter"),
        )
        return fig, True
    except Exception as e:
        return None, str(e)


def build_multi_model_chart(model_probs, best_name):
    """Horizontal bar showing each model's prediction probability."""
    names  = list(model_probs.keys())
    probs  = [model_probs[n] for n in names]
    colors = [("rgba(229,62,62,0.85)" if p > 0.5 else "rgba(72,187,120,0.85)") for p in probs]
    is_best = [n == best_name for n in names]
    border_colors = ["#667eea" if b else "rgba(0,0,0,0)" for b in is_best]

    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color=border_colors, width=2.5),
                    cornerradius=6),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside", textfont=dict(size=11, color="#2d3748"),
        width=0.55,
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#2d3748", line_width=1.5,
                  annotation_text="Decision threshold",
                  annotation_font=dict(size=10, color="#718096"))
    fig.update_layout(
        height=240, xaxis=dict(range=[0,1.2], tickformat=".0%",
                               gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=10,r=60,t=10,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter"), showlegend=False,
    )
    return fig


def build_population_compare(vals, measured):
    """Bar chart: patient vs normal vs hypo population means."""
    hormones = ["TSH","T3","TT4","FTI"]
    patient_vals = [vals.get(h, HORMONE_REF[h]["pop_normal"]) if measured.get(h,True) else None for h in hormones]
    normal_vals  = [HORMONE_REF[h]["pop_normal"] for h in hormones]
    hypo_vals    = [HORMONE_REF[h]["pop_hypo"]   for h in hormones]
    units        = [HORMONE_REF[h]["unit"]        for h in hormones]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Normal Population", x=hormones, y=normal_vals,
                         marker_color="rgba(72,187,120,0.75)", marker_cornerradius=5,
                         text=[f"{v:.1f}" for v in normal_vals], textposition="outside",
                         textfont=dict(size=9.5),
                         hovertemplate="<b>%{x}</b> Normal mean: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Hypothyroid Population", x=hormones, y=hypo_vals,
                         marker_color="rgba(229,62,62,0.75)", marker_cornerradius=5,
                         text=[f"{v:.1f}" for v in hypo_vals], textposition="outside",
                         textfont=dict(size=9.5),
                         hovertemplate="<b>%{x}</b> Hypo mean: %{y:.2f}<extra></extra>"))
    p_clean = [v if v is not None else 0 for v in patient_vals]
    fig.add_trace(go.Bar(name="This Patient", x=hormones, y=p_clean,
                         marker_color="rgba(102,126,234,0.9)", marker_cornerradius=5,
                         marker_line=dict(color="#667eea", width=2),
                         text=[f"{v:.1f}" if v else "N/M" for v in patient_vals],
                         textposition="outside", textfont=dict(size=9.5),
                         hovertemplate="<b>%{x}</b> Patient: %{y:.2f}<extra></extra>"))
    fig.update_layout(
        barmode="group", height=300,
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
        margin=dict(l=0,r=0,t=40,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter"),
        title=dict(text="Patient vs Population Benchmarks", font=dict(size=13,color="#4a5568"), x=0.5),
    )
    return fig


def build_batch_charts(results):
    # Donut
    counts = results["Prediction"].value_counts()
    vals = [counts.get("Hypothyroid",0), counts.get("Normal",0)]
    fig_donut = go.Figure(go.Pie(
        labels=["Hypothyroid","Normal"], values=vals,
        marker_colors=["#fc8181","#68d391"],
        hole=0.55, textfont_size=13,
        hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>",
        pull=[0.04, 0],
    ))
    fig_donut.update_layout(
        height=280, margin=dict(l=0,r=0,t=30,b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        legend=dict(orientation="h", y=-0.08),
        title=dict(text="Diagnosis Split", font=dict(size=13,color="#4a5568"), x=0.5),
        annotations=[dict(text=f"<b>{len(results)}</b><br>patients", x=0.5, y=0.5,
                          font=dict(size=15, color="#2d3748"), showarrow=False)],
    )
    # Prob distribution
    fig_hist = go.Figure()
    for label, col in [("Hypothyroid","#fc8181"),("Normal","#68d391")]:
        sub = results[results["Prediction"]==label]["Probability"]
        if len(sub):
            fig_hist.add_trace(go.Histogram(
                x=sub, name=label, marker_color=col, opacity=0.75, nbinsx=20,
                hovertemplate=f"{label}<br>Prob: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
            ))
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#2d3748",
                       annotation_text="Threshold")
    fig_hist.update_layout(
        height=280, barmode="overlay",
        xaxis=dict(title="Risk Score", gridcolor="rgba(0,0,0,0.06)", tickformat=".0%"),
        yaxis=dict(title="Count", gridcolor="rgba(0,0,0,0.06)"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0,r=0,t=50,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter"),
        title=dict(text="Risk Score Distribution", font=dict(size=13,color="#4a5568"),x=0.5),
    )
    # Risk tier
    rc = results["Risk Level"].value_counts().reindex(["Low","Moderate","High"]).fillna(0)
    fig_risk = go.Figure(go.Bar(
        x=rc.index, y=rc.values, marker_color=["#68d391","#ed8936","#e53e3e"],
        text=rc.values.astype(int), textposition="outside", textfont=dict(size=12),
        marker_cornerradius=6, width=0.5,
        hovertemplate="<b>%{x} Risk</b>: %{y}<extra></extra>",
    ))
    fig_risk.update_layout(
        height=280, xaxis_tickfont=dict(size=12),
        yaxis=dict(title="Count", gridcolor="rgba(0,0,0,0.06)"),
        margin=dict(l=0,r=0,t=50,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter"),
        title=dict(text="Risk Stratification", font=dict(size=13,color="#4a5568"), x=0.5),
    )
    return fig_donut, fig_hist, fig_risk


def build_scatter_timeline(results):
    fig = px.scatter(
        results.reset_index(), x="index", y="Probability",
        color="Prediction", size_max=14,
        color_discrete_map={"Hypothyroid":"#e53e3e","Normal":"#48bb78"},
        hover_data={"Probability":":.3f","Risk Level":True},
        labels={"index":"Patient #"},
        title="Patient-level Risk Scores",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#2d3748", line_width=1.5,
                  annotation_text="Decision boundary", annotation_font=dict(size=10))
    fig.update_traces(marker=dict(size=9, line=dict(color="white",width=1.5)))
    fig.update_layout(
        height=300, legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(range=[0,1.05], tickformat=".0%", gridcolor="rgba(0,0,0,0.06)"),
        margin=dict(l=0,r=0,t=50,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter"),
    )
    return fig


def build_model_radar(comp):
    metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    comp_c = comp[metrics].clip(0,1)
    COLORS = ["#667eea","#48bb78","#ed8936","#e53e3e","#9f7aea"]
    fig = go.Figure()
    for i, (model, row) in enumerate(comp_c.iterrows()):
        v = row[metrics].tolist(); v.append(v[0])
        m2 = metrics + [metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=v, theta=m2, fill="toself", name=model,
            line=dict(color=COLORS[i%5], width=2.5),
            fillcolor=f"rgba({int(COLORS[i%5][1:3],16)},{int(COLORS[i%5][3:5],16)},{int(COLORS[i%5][5:],16)},0.07)",
            hovertemplate="%{theta}: %{r:.4f}<extra>" + model + "</extra>",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickformat=".0%",
                           gridcolor="rgba(0,0,0,0.1)"),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=True, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        height=400, margin=dict(l=50,r=50,t=20,b=80),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        title=dict(text="Model Performance Radar", font=dict(size=14,color="#4a5568"), x=0.5),
    )
    return fig


def build_metric_bar(comp):
    metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    COLORS  = ["#667eea","#48bb78","#ed8936","#e53e3e","#9f7aea"]
    comp_c  = comp[metrics].clip(0,1)
    fig = go.Figure()
    for i, (model, row) in enumerate(comp_c.iterrows()):
        fig.add_trace(go.Bar(
            name=model, x=metrics, y=[row[m] for m in metrics],
            marker_color=COLORS[i%5], marker_cornerradius=5, opacity=0.88,
            text=[f"{row[m]*100:.1f}%" for m in metrics],
            textposition="outside", textfont=dict(size=9),
            hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=380,
        yaxis=dict(range=[0,1.15], tickformat=".0%", gridcolor="rgba(0,0,0,0.06)"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=0,r=0,t=20,b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter", size=11),
    )
    return fig


def clinical_recs(pred, prob, raw_dict):
    recs = []  # (style, icon, title, body)
    tsh = raw_dict.get("TSH")
    fti = raw_dict.get("FTI")
    t3  = raw_dict.get("T3")
    tt4 = raw_dict.get("TT4")

    if isinstance(tsh, (int,float)):
        if tsh > 10:
            recs.append(("rec-urgent","🚨","Critical TSH Elevation",
                f"TSH of {tsh:.1f} μIU/mL is significantly above normal (0.4–4.0). This strongly suggests primary hypothyroidism. Immediate endocrinology referral is warranted."))
        elif tsh > 4:
            recs.append(("rec-warning","⚠️","Elevated TSH",
                f"TSH of {tsh:.1f} μIU/mL exceeds the upper normal limit. Primary hypothyroidism is likely. Confirm with free T4 measurement."))
        elif tsh < 0.4:
            recs.append(("rec-info","ℹ️","Suppressed TSH",
                f"TSH of {tsh:.1f} μIU/mL is below normal. May indicate hyperthyroidism or over-replacement. Consider FT4/T3 measurement."))

    if isinstance(fti, (int,float)) and fti < 60:
        recs.append(("rec-warning","⚗️","Low Free Thyroxine Index",
            f"FTI of {fti:.0f} (normal: 60–150) confirms reduced free thyroxine. Supportive of hypothyroid diagnosis."))

    if isinstance(t3, (int,float)) and t3 < 1.1:
        recs.append(("rec-warning","🧪","Low T3",
            f"T3 of {t3:.1f} nmol/L (normal: 1.1–2.8). Low T3 is associated with severe or long-standing hypothyroidism."))

    if isinstance(tt4, (int,float)) and tt4 < 60:
        recs.append(("rec-warning","🧫","Low Total T4",
            f"TT4 of {tt4:.0f} nmol/L is below the normal range (60–150). Confirms insufficient thyroid hormone production."))

    if raw_dict.get("thyroid surgery") == "t":
        recs.append(("rec-info","🏥","Post-Thyroidectomy",
            "Prior thyroid surgery significantly increases hypothyroid risk. Lifelong replacement therapy should be evaluated."))

    if raw_dict.get("I131 treatment") == "t":
        recs.append(("rec-info","☢️","Radioiodine History",
            "I131 treatment destroys thyroid tissue and frequently causes hypothyroidism. Regular TSH monitoring every 6–12 months is essential."))

    if raw_dict.get("pregnant") == "t":
        recs.append(("rec-urgent","🤰","Pregnancy — Stricter Targets",
            "TSH targets during pregnancy are stricter: <2.5 μIU/mL in the first trimester. Untreated hypothyroidism risks fetal neurodevelopment."))

    if raw_dict.get("lithium") == "t":
        recs.append(("rec-warning","⚗️","Lithium-Induced Risk",
            "Lithium interferes with thyroid hormone synthesis and release, causing hypothyroidism in 20–50% of long-term users. Annual thyroid screening is recommended."))

    if raw_dict.get("hypopituitary") == "t":
        recs.append(("rec-info","🧠","Secondary Hypothyroidism Risk",
            "Hypopituitary conditions can cause central (secondary) hypothyroidism where TSH may appear falsely normal despite low T4. Free T4 is a more reliable marker."))

    if raw_dict.get("on thyroxine") == "t":
        recs.append(("rec-info","💊","Currently on Thyroxine",
            "Patient is already on thyroid replacement. Evaluate whether current dose is adequate based on TSH target (0.5–2.5 μIU/mL typically)."))

    # Final verdict
    if pred == 1 and prob >= 0.80:
        recs.append(("rec-urgent","🔴","High-Confidence Positive Screen",
            f"Model confidence: {prob*100:.1f}%. High-priority referral to endocrinology. Initiate levothyroxine therapy discussion pending further workup."))
    elif pred == 1:
        recs.append(("rec-warning","🟡","Positive Screen — Confirm",
            f"Model confidence: {prob*100:.1f}%. Repeat thyroid panel in 6–8 weeks. Consider free T4 and anti-TPO antibody testing to confirm autoimmune etiology."))
    elif prob >= 0.30:
        recs.append(("rec-warning","🟡","Borderline — Monitor",
            f"Borderline score ({prob*100:.1f}%). Retest in 3–6 months, especially if symptomatic (fatigue, weight gain, cold intolerance)."))
    else:
        recs.append(("rec-success","🟢","Low Risk — Routine Monitoring",
            f"Risk score {prob*100:.1f}%. No immediate action required. Annual thyroid screening is sufficient for asymptomatic patients without risk factors."))

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 16px;'>
        <div style='font-size:3.2rem;'>🦋</div>
        <h1 style='font-size:1.45rem;font-weight:800;margin:8px 0 2px;letter-spacing:-.3px;'>HypoDetect AI</h1>
        <p style='font-size:.78rem;opacity:.55;margin:0;letter-spacing:.8px;'>THYROID INTELLIGENCE PLATFORM</p>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1);margin:4px 0 16px;'>
    """, unsafe_allow_html=True)

    mode = st.radio("nav", [
        "🏠  Dashboard",
        "🩺  Patient Screening",
        "📂  Batch Analysis",
        "📊  Model Analytics",
        "📋  Session History",
    ], label_visibility="collapsed")

    try:
        with st.spinner(""):
            pre_art, scaler, meta, best, xgb, rf, lr, shap_exp, comparison = load_artifacts()
        st.markdown(f"""
        <div style='background:rgba(72,187,120,0.12);border:1px solid rgba(72,187,120,0.3);
             border-radius:10px;padding:12px 14px;margin-top:16px;'>
            <div style='font-size:.7rem;opacity:.6;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;'>Active Model</div>
            <div style='font-weight:700;font-size:.95rem;'>{meta["name"]}</div>
            <div style='font-size:.75rem;opacity:.5;margin-top:2px;'>ROC-AUC: {comparison.loc[meta["name"],"ROC-AUC"] if meta["name"] in comparison.index else "–"}</div>
        </div>""", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("⚠️ Run `python train.py` first")
        st.stop()

    if "history" in st.session_state and st.session_state.history:
        h = st.session_state.history
        n_h = sum(1 for r in h if r["pred"]==1)
        n_n = len(h) - n_h
        st.markdown(f"""
        <div style='margin-top:14px;'>
            <div style='font-size:.7rem;opacity:.55;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;'>Session Stats</div>
            <div style='display:flex;gap:8px;'>
                <div style='flex:1;background:rgba(229,62,62,0.12);border-radius:8px;padding:10px;text-align:center;'>
                    <div style='font-size:1.5rem;font-weight:800;color:#fc8181;'>{n_h}</div>
                    <div style='font-size:.68rem;opacity:.6;'>Positive</div>
                </div>
                <div style='flex:1;background:rgba(72,187,120,0.12);border-radius:8px;padding:10px;text-align:center;'>
                    <div style='font-size:1.5rem;font-weight:800;color:#68d391;'>{n_n}</div>
                    <div style='font-size:.68rem;opacity:.6;'>Negative</div>
                </div>
                <div style='flex:1;background:rgba(102,126,234,0.12);border-radius:8px;padding:10px;text-align:center;'>
                    <div style='font-size:1.5rem;font-weight:800;color:#667eea;'>{len(h)}</div>
                    <div style='font-size:.68rem;opacity:.6;'>Total</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='position:absolute;bottom:20px;left:0;right:0;text-align:center;padding:0 16px;'>
        <p style='font-size:.68rem;opacity:.35;line-height:1.6;'>
            ⚕️ Clinical decision support only.<br>Not a substitute for medical advice.
        </p>
    </div>""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🏠  Dashboard":
    st.markdown("""
    <div class="page-hero">
        <h1>🦋 HypoDetect AI Platform</h1>
        <p>Intelligent thyroid function screening powered by ensemble machine learning.<br>
           Trained on 3,772 clinical records · 5-model ensemble · SHAP explainability</p>
    </div>""", unsafe_allow_html=True)

    # KPI row
    comp_c = comparison.drop(columns=["score"], errors="ignore")
    best_row = comp_c.loc[meta["name"]] if meta["name"] in comp_c.index else comp_c.iloc[0]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Best Model",     meta["name"].split()[0])
    k2.metric("ROC-AUC",        f"{best_row.get('ROC-AUC',0)*100:.1f}%")
    k3.metric("Recall",         f"{best_row.get('Recall',0)*100:.1f}%")
    k4.metric("F1-Score",       f"{best_row.get('F1',0)*100:.1f}%")
    k5.metric("Training Set",   "3,772 records")

    st.markdown("<br>", unsafe_allow_html=True)

    # Model performance quick view
    c_left, c_right = st.columns([3,2], gap="large")
    with c_left:
        st.markdown('<div class="section-title">📊 Model Performance Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Heatmap-style table
        metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        comp_display = comp_c[metrics].copy()
        def color_val(v):
            g = int(v * 200)
            return f"background-color:rgba(99,102,241,{v:.2f});color:{'#ffffff' if v>0.65 else '#1e1b4b'}"
        st.dataframe(
            comp_display.style.format("{:.4f}").applymap(color_val),
            use_container_width=True, height=220,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="section-title">🦠 Dataset Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_ds = go.Figure(go.Pie(
            labels=["Normal (P)","Hypothyroid (N)"], values=[3481, 291],
            marker_colors=["#68d391","#fc8181"], hole=0.5,
            textfont_size=12, pull=[0,0.06],
            hovertemplate="<b>%{label}</b>: %{value}<br>%{percent}<extra></extra>",
        ))
        fig_ds.update_layout(
            height=200, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
            legend=dict(orientation="h", y=-0.1),
            annotations=[dict(text="<b>3,772</b><br>records",x=0.5,y=0.5,
                              font=dict(size=13,color="#2d3748"),showarrow=False)],
        )
        st.plotly_chart(fig_ds, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature guide
    st.markdown('<div class="section-title">📌 Key Biomarkers Reference</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (h, r) in enumerate(HORMONE_REF.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:16px;">
                <div style="font-size:1.3rem;font-weight:800;color:#667eea;">{h}</div>
                <div style="font-size:.72rem;color:#718096;margin:3px 0 8px;">{r['desc']}</div>
                <div style="font-size:.8rem;color:#2d3748;">
                    <b>{r['low']}–{r['high']}</b><br>
                    <span style="color:#718096;">{r['unit']}</span>
                </div>
                <div style="font-size:.72rem;margin-top:6px;color:#718096;">
                    Normal avg: {r['pop_normal']}<br>Hypo avg: {r['pop_hypo']}
                </div>
            </div>""", unsafe_allow_html=True)

    # Quick start
    st.markdown('<div class="section-title">🚀 Quick Start</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    for col, icon, title, desc, page in [
        (q1,"🩺","Screen a Patient","Enter hormone values for instant AI prediction","🩺  Patient Screening"),
        (q2,"📂","Batch Upload","Analyse hundreds of patients via CSV","📂  Batch Analysis"),
        (q3,"📊","Model Analytics","Explore model performance & explainability","📊  Model Analytics"),
        (q4,"📋","View History","Review all predictions this session","📋  Session History"),
    ]:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;cursor:pointer;transition:all .2s;">
                <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700;font-size:.95rem;color:#1a202c;">{title}</div>
                <div style="font-size:.8rem;color:#718096;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PATIENT
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🩺  Patient Screening":
    st.markdown("""
    <div class="page-hero">
        <h1>🩺 Patient Thyroid Screening</h1>
        <p>Enter clinical data below. The AI will analyse hormone levels, clinical flags, and risk factors to generate a comprehensive prediction report.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("patient_form", clear_on_submit=False):
        # Demographics
        st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
        d1,d2,d3,d4,d5 = st.columns([1.2,0.8,0.8,1.2,2])
        patient_id = d1.text_input("Patient ID", placeholder="PT-001", help="Optional identifier")
        age        = d2.number_input("Age", 1, 120, 45, help="Patient age in years")
        sex        = d3.selectbox("Sex", ["F","M"])
        referral   = d4.selectbox("Referral Source", ["SVHC","other","SVI","STMW","SVHD"])
        notes      = d5.text_input("Clinical Notes (optional)", placeholder="Additional context…")

        st.markdown("<br>", unsafe_allow_html=True)

        # Hormone panel
        st.markdown('<div class="section-title">🧪 Thyroid Hormone Panel</div>', unsafe_allow_html=True)
        st.caption("Enable each test you have results for. Untick = 'not measured'.")
        h1,h2,h3,h4,h5 = st.columns(5)
        hormone_inputs = {}
        hormone_measured = {}
        for col_w, hname, defval, step, hi, tip in [
            (h1,"TSH", 2.0,0.1, 500.0,"Normal: 0.4–4.0 μIU/mL. PRIMARY marker — elevated in hypothyroid."),
            (h2,"T3",  2.0,0.1, 15.0, "Normal: 1.1–2.8 nmol/L. Reduced in severe hypothyroid."),
            (h3,"TT4",100.0,1.0,500.0,"Normal: 60–150 nmol/L. Total thyroxine level."),
            (h4,"T4U", 1.0,0.01,3.0,  "Normal: 0.7–1.3 fraction. Resin T4 uptake."),
            (h5,"FTI",100.0,1.0,500.0,"Normal: 60–150. Free Thyroxine Index = TT4 × T4U/100."),
        ]:
            with col_w:
                m = st.checkbox(f"{hname} measured", value=True, key=f"m_{hname}")
                v = st.number_input(f"{hname}", 0.0, hi, defval, step=step,
                                    disabled=not m, help=tip, key=f"v_{hname}")
                hormone_inputs[hname]  = v
                hormone_measured[hname] = m

                # Live status indicator
                if m:
                    status_txt, status_col = hormone_status(v, hname)
                    st.markdown(f'<div style="font-size:.75rem;color:{status_col};font-weight:600;">{status_txt}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:.75rem;color:#a0aec0;">— not measured</div>',
                                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Clinical flags
        st.markdown('<div class="section-title">🏥 Clinical History & Risk Factors</div>', unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns(4)
        flag_inputs = {}
        with f1:
            st.caption("**🔵 Treatment**")
            flag_inputs["on thyroxine"]              = st.checkbox("On Thyroxine")
            flag_inputs["on antithyroid medication"] = st.checkbox("Antithyroid Medication")
            flag_inputs["I131 treatment"]            = st.checkbox("I131 Radioiodine")
            flag_inputs["lithium"]                   = st.checkbox("Lithium Use")
        with f2:
            st.caption("**🟠 Structural**")
            flag_inputs["thyroid surgery"] = st.checkbox("Thyroid Surgery")
            flag_inputs["goitre"]          = st.checkbox("Goitre Present")
            flag_inputs["tumor"]           = st.checkbox("Thyroid Tumor")
            flag_inputs["hypopituitary"]   = st.checkbox("Hypopituitary")
        with f3:
            st.caption("**🔴 Current Status**")
            flag_inputs["sick"]      = st.checkbox("Currently Sick")
            flag_inputs["pregnant"]  = st.checkbox("Pregnant")
            flag_inputs["psych"]     = st.checkbox("Psychiatric Dx")
        with f4:
            st.caption("**🟣 Clinical Suspicion**")
            flag_inputs["query on thyroxine"]   = st.checkbox("Query on Thyroxine")
            flag_inputs["query hypothyroid"]    = st.checkbox("Query Hypothyroid")
            flag_inputs["query hyperthyroid"]   = st.checkbox("Query Hyperthyroid")

        st.markdown("<br>", unsafe_allow_html=True)
        sub = st.form_submit_button("🔬  Run Full Analysis", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if sub:
        def tf(v): return "t" if v else "f"
        raw = {
            "age": age, "sex": sex, "referral source": referral,
            "TSH measured": tf(hormone_measured["TSH"]),
            "TSH": hormone_inputs["TSH"] if hormone_measured["TSH"] else "?",
            "T3 measured":  tf(hormone_measured["T3"]),
            "T3":  hormone_inputs["T3"]  if hormone_measured["T3"]  else "?",
            "TT4 measured": tf(hormone_measured["TT4"]),
            "TT4": hormone_inputs["TT4"] if hormone_measured["TT4"] else "?",
            "T4U measured": tf(hormone_measured["T4U"]),
            "T4U": hormone_inputs["T4U"] if hormone_measured["T4U"] else "?",
            "FTI measured": tf(hormone_measured["FTI"]),
            "FTI": hormone_inputs["FTI"] if hormone_measured["FTI"] else "?",
            "TBG measured": "f", "TBG": "?",
        }
        for k, v in flag_inputs.items():
            raw[k] = tf(v)

        with st.spinner("🤖 Running AI analysis across all models…"):
            X_proc = preprocess_input(raw, pre_art)
            all_probs = predict_all_models(X_proc, scaler, rf, xgb, lr, best, meta)
            best_prob = all_probs.get(meta["name"], list(all_probs.values())[0])
            pred = int(best_prob > 0.5)
            prob = best_prob

        # Save history
        pid = patient_id.strip() or f"PT-{len(st.session_state.history)+1:03d}"
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "id":   pid, "age": age, "sex": sex,
            "pred": pred, "prob": round(prob,4), "risk": risk_label(prob),
            "tsh":  hormone_inputs["TSH"] if hormone_measured["TSH"] else None,
        })

        st.divider()

        # ── Verdict banner ─────────────────────────────────────────────────────
        card_cls = "result-hypo" if pred==1 else "result-normal"
        icon_big = "🔴" if pred==1 else "🟢"
        verdict  = "HYPOTHYROID DETECTED" if pred==1 else "NORMAL — NO HYPOTHYROIDISM"
        col_txt  = "#c53030" if pred==1 else "#276749"
        rl       = risk_label(prob)
        sc_cls   = score_pill_cls(prob)

        st.markdown(f"""
        <div class="{card_cls}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
                <div>
                    <div class="result-title" style="color:{col_txt};">{icon_big} {verdict}</div>
                    <div class="result-meta">
                        <span>Patient: <b>{pid}</b></span>
                        <span>Age: <b>{age}</b></span>
                        <span>Sex: <b>{sex}</b></span>
                        <span>{datetime.now().strftime('%d %b %Y · %H:%M')}</span>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                    <span class="score-pill {sc_cls}">{rl} Risk · {prob*100:.1f}%</span>
                    <span class="model-badge">🤖 {meta['name']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Row 1: Gauge + Hormone bars ────────────────────────────────────────
        r1a, r1b = st.columns([1, 1.4], gap="large")
        with r1a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(build_gauge(prob), use_container_width=True, config={"displayModeBar":False})
            m1,m2,m3 = st.columns(3)
            m1.metric("Risk Score",  f"{prob*100:.1f}%")
            m2.metric("Verdict",     "Positive" if pred==1 else "Negative")
            m3.metric("Confidence",  "High" if abs(prob-.5)>.3 else "Med" if abs(prob-.5)>.15 else "Low")
            st.markdown("</div>", unsafe_allow_html=True)
        with r1b:
            st.markdown('<div class="glass-card" style="padding:16px 20px;">', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:700;font-size:.95rem;color:#1a202c;margin-bottom:12px;">🧪 Hormone Panel Status</div>', unsafe_allow_html=True)
            st.plotly_chart(build_hormone_bars(hormone_inputs, hormone_measured),
                            use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: Radar + Population compare ─────────────────────────────────
        r2a, r2b = st.columns([1,1], gap="large")
        with r2a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(build_hormone_radar(hormone_inputs, hormone_measured),
                            use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)
        with r2b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(build_population_compare(hormone_inputs, hormone_measured),
                            use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 3: Multi-model + SHAP ──────────────────────────────────────────
        r3a, r3b = st.columns([1, 1.6], gap="large")
        with r3a:
            st.markdown('<div class="section-title">🤖 All Models Consensus</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            agree_count = sum(1 for p in all_probs.values() if (p>0.5)==(pred==1))
            agree_pct   = agree_count / len(all_probs) * 100
            st.markdown(f"""
            <div style="background:{'rgba(229,62,62,0.08)' if pred==1 else 'rgba(72,187,120,0.08)'};
                 border-radius:10px;padding:12px 16px;margin-bottom:12px;text-align:center;">
                <div style="font-size:1.5rem;font-weight:800;color:{'#e53e3e' if pred==1 else '#48bb78'};">
                    {agree_count}/{len(all_probs)} models agree
                </div>
                <div style="font-size:.82rem;color:#718096;">Ensemble consensus: {agree_pct:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(build_multi_model_chart(all_probs, meta["name"]),
                            use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        with r3b:
            st.markdown('<div class="section-title">🔍 AI Explanation — SHAP Values</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.caption("🔴 Red = pushes toward Hypothyroid &nbsp; 🟢 Green = pushes toward Normal")
            fig_shap, ok = build_shap_chart(shap_exp, X_proc)
            if fig_shap:
                st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info(f"SHAP unavailable: {ok}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Clinical recommendations ───────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Clinical Interpretation & Recommendations</div>', unsafe_allow_html=True)
        recs = clinical_recs(pred, prob, raw)
        rec_cols = st.columns(2)
        for i, (style, icon, title, body) in enumerate(recs):
            with rec_cols[i % 2]:
                st.markdown(f"""
                <div class="rec-item {style}">
                    <span class="rec-icon">{icon}</span>
                    <div><div style="font-weight:700;font-size:.88rem;color:#1a202c;margin-bottom:3px;">{title}</div>
                    <div class="rec-text">{body}</div></div>
                </div>""", unsafe_allow_html=True)

        # ── Active flags + Hormone ratios ─────────────────────────────────────
        fl_col, rat_col = st.columns([3, 2], gap="large")
        with fl_col:
            st.markdown('<div class="section-title">🚩 Active Clinical Flags</div>', unsafe_allow_html=True)
            flag_html = ""
            active_count = 0
            for key, (emoji, label_f, sev) in CLINICAL_FLAGS.items():
                active = raw.get(key) == "t"
                if active:
                    active_count += 1
                    cls = f"flag-active-{sev}"
                else:
                    cls = "flag-inactive"
                flag_html += f'<span class="flag-chip {cls}">{emoji} {label_f}</span>'
            if active_count:
                st.markdown(f'<div style="margin-bottom:8px;font-size:.8rem;color:#718096;">{active_count} active risk flag(s)</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="line-height:2.2;">{flag_html}</div>', unsafe_allow_html=True)

        with rat_col:
            if all(hormone_measured.values()):
                st.markdown('<div class="section-title">⚗️ Derived Ratios</div>', unsafe_allow_html=True)
                tsh_v = hormone_inputs["TSH"]; tt4_v = hormone_inputs["TT4"]
                t3_v  = hormone_inputs["T3"];  t4u_v = hormone_inputs["T4U"]
                fti_v = hormone_inputs["FTI"]
                rr1,rr2,rr3 = st.columns(3)
                rr1.metric("T3/TT4",    f"{t3_v/(tt4_v+1e-6):.4f}", help="Low → poor T4→T3 conversion")
                rr2.metric("TSH/TT4",   f"{tsh_v/(tt4_v+1e-6):.4f}", help="High → pituitary compensation")
                rr3.metric("FTI/T4U",   f"{fti_v/(t4u_v+1e-6):.2f}", help="Adjusted free hormone index")

        # ── Notes ──────────────────────────────────────────────────────────────
        if notes.strip():
            st.markdown(f"""
            <div class="glass-card" style="margin-top:8px;">
                <div style="font-size:.78rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">📝 Clinical Notes</div>
                <div style="font-size:.9rem;color:#2d3748;line-height:1.6;">{notes}</div>
            </div>""", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div style="background:rgba(102,126,234,0.06);border:1px solid rgba(102,126,234,0.2);
             border-radius:10px;padding:12px 16px;margin-top:16px;">
            <span style="font-size:.82rem;color:#4a5568;">
                ⚕️ <b>Disclaimer:</b> HypoDetect is a clinical decision <i>support</i> tool.
                All results must be interpreted by a qualified healthcare professional in conjunction with the full clinical picture.
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📂  Batch Analysis":
    st.markdown("""
    <div class="page-hero">
        <h1>📂 Batch Patient Analysis</h1>
        <p>Upload a CSV to screen an entire cohort at once. Supports up to thousands of patients.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4>📌 CSV Format Guide</h4>
        <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:.87rem;color:#4a5568;">
            <div>• Same column names as training data</div>
            <div>• Missing values → <code>?</code></div>
            <div>• Boolean columns → <code>t</code> / <code>f</code></div>
            <div>• Numerical columns → numeric values</div>
            <div>• <code>binaryClass</code> column optional (enables accuracy evaluation)</div>
        </div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        with st.expander(f"📋 Data Preview — {len(df_raw)} rows × {len(df_raw.columns)} columns", expanded=False):
            st.dataframe(df_raw.head(8), use_container_width=True)

        if st.button("▶️  Run Batch Screening", use_container_width=True, type="primary"):
            prog = st.progress(0, "Preprocessing…")
            with st.spinner(""):
                results, y_true = predict_batch(df_raw, pre_art, scaler, meta, best)
                prog.progress(100, "Complete!")

            st.success(f"✅ Screened **{len(results)}** patients successfully.")
            st.divider()

            # KPIs
            n_hypo  = (results["Prediction"]=="Hypothyroid").sum()
            n_norm  = len(results) - n_hypo
            n_high  = (results["Risk Level"]=="High").sum()
            n_mod   = (results["Risk Level"]=="Moderate").sum()
            avg_p   = results["Probability"].mean()
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Total",      len(results))
            k2.metric("Hypothyroid",n_hypo, delta=f"{n_hypo/len(results)*100:.1f}%")
            k3.metric("Normal",     n_norm, delta=f"{n_norm/len(results)*100:.1f}%")
            k4.metric("High Risk",  n_high)
            k5.metric("Moderate Risk",n_mod)
            k6.metric("Avg Risk Score",f"{avg_p*100:.1f}%")

            if y_true is not None and len(y_true) > 0:
                try:
                    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                    yp = (results["Probability"]>0.5).astype(int).values
                    acc = accuracy_score(y_true, yp)
                    f1v = f1_score(y_true, yp, zero_division=0)
                    auc = roc_auc_score(y_true, results["Probability"].values)
                    st.markdown(f"""
                    <div style="background:rgba(72,187,120,0.08);border:1px solid rgba(72,187,120,0.3);
                         border-radius:10px;padding:12px 20px;margin:8px 0;display:flex;gap:32px;flex-wrap:wrap;align-items:center;">
                        <span style="font-size:.85rem;">🎯 <b>Ground truth detected</b></span>
                        <span style="font-size:.85rem;">Accuracy: <b>{acc*100:.1f}%</b></span>
                        <span style="font-size:.85rem;">F1-Score: <b>{f1v:.4f}</b></span>
                        <span style="font-size:.85rem;">ROC-AUC: <b>{auc:.4f}</b></span>
                    </div>""", unsafe_allow_html=True)
                except: pass

            # Charts
            fd, fh, fr = build_batch_charts(results)
            c1,c2,c3 = st.columns(3)
            with c1: st.plotly_chart(fd, use_container_width=True, config={"displayModeBar":False})
            with c2: st.plotly_chart(fh, use_container_width=True, config={"displayModeBar":False})
            with c3: st.plotly_chart(fr, use_container_width=True, config={"displayModeBar":False})

            # Scatter timeline
            st.plotly_chart(build_scatter_timeline(results),
                            use_container_width=True, config={"displayModeBar":False})

            # Results table
            st.markdown('<div class="section-title">📋 Full Patient Results</div>', unsafe_allow_html=True)
            display_cols = ["Prediction","Probability","Risk Level"] + \
                           [c for c in ["age","sex","TSH","T3","TT4","FTI","T4U"] if c in results.columns]

            def style_rows(row):
                c = "#fff5f5" if row.get("Prediction")=="Hypothyroid" else "#f0fff4"
                return [f"background-color:{c}"]*len(row)

            filtered_pred = st.selectbox("Filter by prediction", ["All","Hypothyroid","Normal"], index=0)
            filtered_risk = st.selectbox("Filter by risk level",  ["All","High","Moderate","Low"], index=0)
            filt = results.copy()
            if filtered_pred != "All": filt = filt[filt["Prediction"]==filtered_pred]
            if filtered_risk != "All": filt = filt[filt["Risk Level"]==filtered_risk]
            st.caption(f"Showing {len(filt)} of {len(results)} patients")
            st.dataframe(
                filt[display_cols].style.apply(style_rows, axis=1).format({"Probability":"{:.3f}"}),
                use_container_width=True, height=400,
            )

            # Download
            buf = io.StringIO()
            results.to_csv(buf, index=False)
            st.download_button(
                "⬇️  Download Full Results CSV",
                data=buf.getvalue(),
                file_name=f"hypodetect_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#a0aec0;">
            <div style="font-size:3rem;margin-bottom:12px;">📂</div>
            <div style="font-size:1rem;font-weight:600;">Drop your CSV above to begin</div>
            <div style="font-size:.85rem;margin-top:6px;">Supports the standard hypothyroid dataset format</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📊  Model Analytics":
    st.markdown("""
    <div class="page-hero">
        <h1>📊 Model Performance Analytics</h1>
        <p>Deep-dive into how each model was trained, evaluated, and selected.</p>
    </div>""", unsafe_allow_html=True)

    comp_c = comparison.drop(columns=["score"], errors="ignore")

    # Best model hero
    best_name = meta["name"]
    br = comp_c.loc[best_name] if best_name in comp_c.index else comp_c.iloc[0]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#eef2ff,#ede9fe);border:2px solid #a5b4fc;border-radius:16px;
         padding:24px 32px;margin-bottom:24px;display:flex;justify-content:space-between;
         align-items:center;flex-wrap:wrap;gap:16px;">
        <div>
            <div style="font-size:.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.6px;font-weight:700;margin-bottom:6px;">🏆 Selected Best Model</div>
            <div style="font-size:2rem;font-weight:800;color:#1e1b4b;">{best_name}</div>
            <div style="color:#4338ca;font-size:.85rem;margin-top:4px;font-weight:500;">
                Selection criterion: 0.5 × ROC-AUC + 0.5 × Recall
            </div>
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            {"".join([f'<div style="text-align:center;background:rgba(99,102,241,0.1);border-radius:10px;padding:10px 18px;"><div style="font-size:2rem;font-weight:800;color:#4338ca;">{br[m]*100:.1f}%</div><div style="font-size:.72rem;color:#6366f1;text-transform:uppercase;letter-spacing:.5px;font-weight:700;">{m}</div></div>' for m in ["Accuracy","Precision","Recall","F1","ROC-AUC"]])}
        </div>
    </div>""", unsafe_allow_html=True)

    # Model cards
    COLORS = ["#667eea","#48bb78","#ed8936","#e53e3e","#9f7aea"]
    model_cols = st.columns(len(comp_c))
    for i, (mn, row) in enumerate(comp_c.iterrows()):
        is_best = mn == best_name
        with model_cols[i]:
            border = "2px solid #667eea" if is_best else "1px solid #e2e8f0"
            bg = "linear-gradient(135deg,#f0f4ff,#e8edff)" if is_best else "white"
            badge = "<div style='font-size:.65rem;background:#4338ca;color:#ffffff;border-radius:10px;padding:3px 10px;display:inline-block;margin-bottom:6px;font-weight:700;'>⭐ BEST</div>" if is_best else ""
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:14px;padding:18px 14px;
                 text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
                {badge}
                <div style="font-weight:800;font-size:.88rem;color:#1a202c;margin-bottom:12px;line-height:1.3;">{mn}</div>
                {"".join([f'<div style="margin:6px 0;"><div style="font-size:1.1rem;font-weight:800;color:{COLORS[i%5]};">{row[m]*100:.1f}%</div><div style="font-size:.68rem;color:#718096;text-transform:uppercase;letter-spacing:.4px;">{m}</div></div>' for m in ["Accuracy","Recall","F1","ROC-AUC"]])}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    ch1, ch2 = st.columns([1.1,1], gap="large")
    with ch1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(build_model_radar(comp_c), use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)
    with ch2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(build_metric_bar(comp_c), use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Tabs: CM + importances + SHAP + design notes
    t1,t2,t3,t4,t5 = st.tabs([
        "🔲 Confusion Matrix",
        "🌳 Random Forest",
        "⚡ XGBoost",
        "🔍 SHAP Summary",
        "📐 Design Notes",
    ])
    with t1:
        cols = st.columns([1,2,1])
        cm_path = os.path.join(ART_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            with cols[1]: st.image(cm_path, caption=f"Best model: {best_name}", use_container_width=True)
        with st.expander("How to read this matrix"):
            st.markdown("""
| Cell | Meaning | Clinical Impact |
|---|---|---|
| **True Negative** (top-left) | Normal → correctly Normal | ✅ No harm |
| **False Positive** (top-right) | Normal → flagged Hypothyroid | ⚠️ Unnecessary follow-up |
| **False Negative** (bottom-left) | Hypothyroid → missed | 🚨 Most costly — missed diagnosis |
| **True Positive** (bottom-right) | Hypothyroid → correctly caught | ✅ Early treatment |
""")
    with t2:
        p = os.path.join(ART_DIR, "rf_feature_importance.png")
        if os.path.exists(p): st.image(p, use_container_width=True)
        st.info("Random Forest importance = mean decrease in impurity across all trees. Higher = more influential.")
    with t3:
        p = os.path.join(ART_DIR, "xgb_feature_importance.png")
        if os.path.exists(p): st.image(p, use_container_width=True)
        st.info("XGBoost importance = number of times feature is used to split across all trees.")
    with t4:
        p = os.path.join(ART_DIR, "shap_summary.png")
        if os.path.exists(p): st.image(p, use_container_width=True)
        st.markdown("""
        **Reading SHAP:**
        - Each dot = one patient from the test set
        - **Red** = high feature value · **Blue** = low feature value
        - Right of center = pushes toward **Hypothyroid** · Left = pushes toward **Normal**
        - Top features have the largest effect on predictions
        """)
    with t5:
        st.markdown("""
        ### Why these 5 models?
        | Model | Strengths | Limitations |
        |---|---|---|
        | Logistic Regression | Interpretable, calibrated probabilities | Assumes linearity |
        | Random Forest | Robust, handles non-linearity | Slower, large memory |
        | XGBoost | SOTA on tabular data, regularized | Needs hyperparameter tuning |
        | SVM (RBF) | Effective in high-dim, robust to outliers | Poor probability calibration |
        | Deep Learning | Learns complex patterns | Needs more data, black-box |

        ### Why SMOTE over class_weight?
        SMOTE synthesises realistic minority-class examples in feature space rather than simply
        reweighting loss — this produces more generalisable decision boundaries on imbalanced medical data.

        ### Why Recall matters more than Accuracy?
        In a population where only 7.7% have hypothyroidism, a model that predicts everyone as
        "Normal" achieves 92.3% accuracy. Recall measures what fraction of actual hypothyroid
        cases are caught — missing a case delays treatment for months.

        ### Model selection formula
        ```
        score = 0.5 × ROC-AUC + 0.5 × Recall
        ```
        Both are threshold-independent (ROC-AUC) and clinically critical (Recall).
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📋  Session History":
    st.markdown("""
    <div class="page-hero">
        <h1>📋 Session Screening History</h1>
        <p>All patients screened during this session. Data is cleared when the app restarts.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#a0aec0;">
            <div style="font-size:3rem;margin-bottom:16px;">📋</div>
            <div style="font-size:1.1rem;font-weight:600;color:#718096;">No screenings yet this session</div>
            <div style="font-size:.88rem;margin-top:8px;">Go to <b>🩺 Patient Screening</b> to begin</div>
        </div>""", unsafe_allow_html=True)
    else:
        h   = st.session_state.history
        df_h = pd.DataFrame(h)
        n_total = len(df_h)
        n_hypo  = (df_h["pred"]==1).sum()
        n_norm  = n_total - n_hypo
        n_high  = (df_h["risk"]=="High").sum()

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Screened",   n_total)
        k2.metric("Hypothyroid",      n_hypo,  delta=f"{n_hypo/n_total*100:.1f}%")
        k3.metric("Normal",           n_norm)
        k4.metric("High Risk",        n_high)

        # Timeline chart
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=list(range(1,n_total+1)), y=df_h["prob"],
            mode="lines+markers",
            line=dict(color="#cbd5e0", width=1.5),
            marker=dict(color=[risk_color(p) for p in df_h["prob"]],
                        size=13, line=dict(color="#f8fafc",width=2)),
            text=df_h["id"],
            hovertemplate="<b>%{text}</b><br>Risk: %{y:.3f}<extra></extra>",
        ))
        fig_tl.add_hrect(y0=0, y1=0.3, fillcolor="rgba(72,187,120,0.06)", line_width=0)
        fig_tl.add_hrect(y0=0.3, y1=0.65, fillcolor="rgba(237,137,54,0.06)", line_width=0)
        fig_tl.add_hrect(y0=0.65, y1=1.0, fillcolor="rgba(229,62,62,0.06)", line_width=0)
        fig_tl.add_hline(y=0.5, line_dash="dash", line_color="#2d3748", line_width=1.5,
                         annotation_text="Decision threshold")
        fig_tl.update_layout(
            height=280, title="Risk Score Timeline",
            xaxis=dict(title="Screening Order", gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(range=[0,1.05], tickformat=".0%", gridcolor="rgba(0,0,0,0.06)"),
            margin=dict(l=0,r=0,t=50,b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar":False})

        # History table
        df_display = df_h.copy()
        df_display["Diagnosis"] = df_display["pred"].map({1:"🔴 Hypothyroid", 0:"🟢 Normal"})
        df_display["Risk Score (%)"] = (df_display["prob"]*100).round(1)
        disp_cols = ["time","id","age","sex","Diagnosis","Risk Score (%)","risk"]
        rename_map = {"time":"Time","id":"Patient ID","age":"Age","sex":"Sex","risk":"Risk Level"}
        df_show = df_display[disp_cols].rename(columns=rename_map)

        def row_color(row):
            c = "#fff5f5" if "Hypo" in str(row.get("Diagnosis","")) else "#f0fff4"
            return [f"background-color:{c}"]*len(row)

        st.dataframe(df_show.style.apply(row_color, axis=1),
                     use_container_width=True, hide_index=True)

        ecol, ccol = st.columns([4,1])
        with ecol:
            buf = io.StringIO(); df_show.to_csv(buf, index=False)
            st.download_button("⬇️  Export Session Report CSV",
                               data=buf.getvalue(),
                               file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", use_container_width=True)
        with ccol:
            if st.button("🗑️  Clear History", use_container_width=True):
                st.session_state.history = []; st.rerun()

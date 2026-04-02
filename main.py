"""
🍊 JuiceSpace Pro v2 — Explainable AI Space Allocation Agent
With Ollama local LLM, XAI layer, and improved data management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import json, io, time, requests, textwrap
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JuiceSpace Pro",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background: #FAFAF8; }

  /* KPI Cards */
  .kpi-card { background:white; border-radius:12px; padding:18px 20px; border:1px solid #E8E4DC; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.05); }
  .kpi-label { font-size:11px; color:#999; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:5px; }
  .kpi-value { font-size:26px; font-weight:700; color:#1A1A1A; line-height:1.1; }
  .kpi-delta { font-size:11px; margin-top:4px; }
  .kpi-good { color:#2E7D32; } .kpi-bad { color:#C62828; } .kpi-neutral { color:#888; }

  /* Alert boxes */
  .alert-box    { background:#FFF3E0; border-left:4px solid #F57C00; border-radius:4px; padding:10px 16px; margin:6px 0; font-size:13px; color:#5D4037; }
  .success-box  { background:#E8F5E9; border-left:4px solid #43A047; border-radius:4px; padding:10px 16px; margin:6px 0; font-size:13px; color:#1B5E20; }
  .info-box     { background:#E3F2FD; border-left:4px solid #1976D2; border-radius:4px; padding:10px 16px; margin:6px 0; font-size:13px; color:#0D47A1; }
  .warning-box  { background:#FFF8E1; border-left:4px solid #FFA000; border-radius:4px; padding:10px 16px; margin:6px 0; font-size:13px; color:#6D4C00; }

  /* XAI specific */
  .xai-card     { background:#F3E5F5; border-left:4px solid #7B1FA2; border-radius:8px; padding:14px 18px; margin:10px 0; font-size:13px; color:#4A148C; }
  .xai-header   { font-size:13px; font-weight:600; letter-spacing:0.05em; color:#7B1FA2; margin-bottom:6px; text-transform:uppercase; }
  .xai-rule     { background:#EDE7F6; border-radius:6px; padding:8px 12px; margin:5px 0; font-size:12.5px; color:#311B92; font-family:'DM Mono',monospace; }
  .ollama-card  { background:#E8F5E9; border-left:4px solid #2E7D32; border-radius:8px; padding:14px 18px; margin:10px 0; }
  .score-bar-bg { background:#E0E0E0; border-radius:6px; height:8px; width:100%; margin:2px 0; }
  .score-bar    { background:linear-gradient(90deg,#F57C00,#E65100); border-radius:6px; height:8px; }

  /* Buttons */
  .stButton>button { background:#F57C00; color:white; border:none; border-radius:8px; font-weight:600; padding:10px 24px; font-size:14px; font-family:'DM Sans',sans-serif; transition:background 0.2s; }
  .stButton>button:hover { background:#E65100; color:white; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif; font-size:14px; }
  .stTabs [aria-selected="true"] { color:#F57C00 !important; }

  /* Section label */
  .section-header { font-size:12px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#F57C00; margin:10px 0 4px 0; }

  /* Data badge */
  .badge-synth { background:#E3F2FD; color:#1565C0; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; display:inline-block; letter-spacing:0.05em; }
  .badge-real  { background:#E8F5E9; color:#2E7D32; font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; display:inline-block; letter-spacing:0.05em; }

  div[data-testid="stMetricValue"] { font-size:22px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

FLAVORS = ["Mixed Fruit","Orange","Mango","Apple","Guava","Litchi","Pineapple","Pomegranate","Grape","Cranberry"]
SIZE_ORDER = ["200ml","500ml","1L","1.5L"]

SIZES = {
    "200ml": {"pack_width_cm":5.5,  "pack_height_cm":10.0, "mrp":25,  "cost":16, "shelf_life_days":180},
    "500ml": {"pack_width_cm":7.0,  "pack_height_cm":14.5, "mrp":55,  "cost":36, "shelf_life_days":180},
    "1L":    {"pack_width_cm":9.5,  "pack_height_cm":19.0, "mrp":99,  "cost":65, "shelf_life_days":180},
    "1.5L":  {"pack_width_cm":11.5, "pack_height_cm":22.0, "mrp":139, "cost":92, "shelf_life_days":180},
}

DISPLAY_NORMS = {
    "min_facings_per_sku": 2,
    "max_facings_per_sku": 8,
    "industry_sales_per_sqft_inr": 18000,
    "target_gmroi": 3.5,
    "min_gmroi": 2.5,
    "shelf_levels": 5,
    "shelf_depth_cm": 45,
    "gondola_width_cm": 120,
    "near_expiry_threshold_days": 30,
    "reorder_days_cover": 7,
}

SHELF_LEVEL_NAMES = {5:"Top (reach)", 4:"Eye Level ⭐", 3:"Speed Zone", 2:"Waist Level", 1:"Floor Level"}
SHELF_LEVEL_MULTIPLIERS = {5:0.85, 4:1.40, 3:1.15, 2:1.00, 1:0.75}

FLAVOR_COLORS = {
    "Mixed Fruit":"#FF7043","Orange":"#FF9800","Mango":"#FFC107","Apple":"#66BB6A",
    "Guava":"#EC407A","Litchi":"#E91E63","Pineapple":"#CDDC39","Pomegranate":"#D32F2F",
    "Grape":"#7B1FA2","Cranberry":"#AD1457",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "sales_data": generate_sample_sales(),
        "data_source": "synthetic",
        "store_config": {},
        "allocation": None,
        "schedule_log": [],
        "xai_log": [],
        "ollama_available": False,
        "ollama_model": "llama3",
        "last_ollama_response": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample_sales(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    flavor_weights = [0.18,0.17,0.15,0.12,0.08,0.07,0.07,0.06,0.05,0.05]
    size_weights   = [0.30,0.35,0.25,0.10]
    for flavor, fw in zip(FLAVORS, flavor_weights):
        for size, sw in zip(SIZE_ORDER, size_weights):
            base_units = max(int(rng.normal(500*fw*sw*30, 40)), 20)
            mrp = SIZES[size]["mrp"]; cost = SIZES[size]["cost"]
            rows.append({
                "flavor": flavor, "size": size,
                "monthly_units": base_units, "mrp": mrp, "cost": cost,
                "gross_margin": mrp - cost,
                "monthly_sales_inr": base_units * mrp,
                "monthly_margin_inr": base_units * (mrp - cost),
                "stock_on_hand": int(rng.integers(30, 150)),
                "days_to_expiry": int(rng.integers(20, 180)),
                "current_facings": int(rng.integers(1, 6)),
                "current_shelf_level": int(rng.integers(1, 6)),
            })
    return pd.DataFrame(rows)

def get_upload_template():
    rows = []
    for flavor in FLAVORS:
        for size in SIZE_ORDER:
            rows.append({
                "flavor": flavor,
                "size": size,
                "monthly_units": 200,
                "mrp": SIZES[size]["mrp"],
                "cost": SIZES[size]["cost"],
                "stock_on_hand": 80,
                "days_to_expiry": 120,
                "current_facings": 3,
                "current_shelf_level": 3,
            })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df, store_config):
    shelf_depth_ft = DISPLAY_NORMS["shelf_depth_cm"] / 30.48
    results = df.copy()
    if "gross_margin" not in results.columns:
        results["gross_margin"] = results["mrp"] - results["cost"]
    if "monthly_sales_inr" not in results.columns:
        results["monthly_sales_inr"] = results["monthly_units"] * results["mrp"]
    if "monthly_margin_inr" not in results.columns:
        results["monthly_margin_inr"] = results["monthly_units"] * results["gross_margin"]

    results["pack_width_ft"] = results["size"].map(lambda s: SIZES[s]["pack_width_cm"] / 30.48)
    results["sqft_allocated"] = (results["current_facings"] * results["pack_width_ft"] * shelf_depth_ft).clip(lower=0.05)
    results["sales_per_sqft"]  = (results["monthly_sales_inr"] / results["sqft_allocated"]).round(0)
    results["avg_inventory_cost"] = (results["stock_on_hand"] * results["cost"]).clip(lower=1)
    results["gmroi"] = (results["monthly_margin_inr"] * 12 / results["avg_inventory_cost"]).round(2)
    results["vs_benchmark_pct"] = ((results["sales_per_sqft"] - DISPLAY_NORMS["industry_sales_per_sqft_inr"]) / DISPLAY_NORMS["industry_sales_per_sqft_inr"] * 100).round(1)
    results["gross_margin_pct"] = (results["gross_margin"] / results["mrp"] * 100).round(1)
    results["near_expiry"] = results["days_to_expiry"] <= DISPLAY_NORMS["near_expiry_threshold_days"]
    results["low_stock"]   = results["stock_on_hand"] < (results["monthly_units"] / 30 * DISPLAY_NORMS["reorder_days_cover"])
    results["level_multiplier"] = results["current_shelf_level"].map(SHELF_LEVEL_MULTIPLIERS)

    # Priority score (XAI-transparent weighted formula)
    s_sales  = results["monthly_sales_inr"] / results["monthly_sales_inr"].max()
    s_gmroi  = results["gmroi"].clip(0, 10) / 10
    s_spft   = results["sales_per_sqft"] / results["sales_per_sqft"].max()
    s_margin = results["gross_margin"] / results["gross_margin"].max()
    s_fresh  = 1 - (results["days_to_expiry"] / 180).clip(0, 1)

    results["score_sales"]  = (s_sales  * 0.35).round(4)
    results["score_gmroi"]  = (s_gmroi  * 0.25).round(4)
    results["score_spft"]   = (s_spft   * 0.20).round(4)
    results["score_margin"] = (s_margin * 0.10).round(4)
    results["score_fresh"]  = (s_fresh  * 0.10).round(4)
    results["priority_score"] = (
        results["score_sales"] + results["score_gmroi"] +
        results["score_spft"]  + results["score_margin"] + results["score_fresh"]
    ).round(4)
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  ALLOCATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_allocation(metrics_df, store_config):
    df = metrics_df.copy().sort_values("priority_score", ascending=False).reset_index(drop=True)
    xai_decisions = []

    # Rule 1: 1.5L always on level 1-2 (heavy pack rule)
    large = df["size"] == "1.5L"
    df.loc[large, "recommended_level"] = np.where(
        df.loc[large, "priority_score"] > df["priority_score"].quantile(0.5), 2, 1
    )
    df.loc[large, "decision_rule"] = "Rule R1: 1.5L pack → Floor/Waist (weight & size norm)"

    # Rule 2: Priority-based level for all other sizes
    non_large = ~large
    ni = df.index[non_large]
    n_nl = len(ni)
    for i, idx in enumerate(ni):
        pct = i / max(n_nl - 1, 1)
        if pct < 0.25:
            df.loc[idx, "recommended_level"] = 4
            df.loc[idx, "decision_rule"] = f"Rule R2: Top 25% priority → Eye Level (score={df.loc[idx,'priority_score']:.3f})"
        elif pct < 0.55:
            df.loc[idx, "recommended_level"] = 3
            df.loc[idx, "decision_rule"] = f"Rule R2: Mid-high priority → Speed Zone (score={df.loc[idx,'priority_score']:.3f})"
        elif pct < 0.80:
            df.loc[idx, "recommended_level"] = 2
            df.loc[idx, "decision_rule"] = f"Rule R2: Mid priority → Waist Level (score={df.loc[idx,'priority_score']:.3f})"
        else:
            df.loc[idx, "recommended_level"] = 5
            df.loc[idx, "decision_rule"] = f"Rule R2: Lower priority → Top shelf (score={df.loc[idx,'priority_score']:.3f})"

    # Rule 3: Near-expiry override → Eye Level for rapid sell-through
    ne_mask = df["near_expiry"] & (df["recommended_level"] < 3)
    df.loc[ne_mask, "recommended_level"] = 4
    df.loc[ne_mask, "decision_rule"] = (
        df.loc[ne_mask, "decision_rule"] +
        " → OVERRIDDEN to Eye Level [Rule R3: Near-expiry (<30d) sell-through priority]"
    )

    # Rule 4: Facings proportional to priority, clipped to norms
    score_norm = (df["priority_score"] - df["priority_score"].min()) / \
                 (df["priority_score"].max() - df["priority_score"].min() + 1e-9)
    df["recommended_facings"] = (2 + score_norm * 6).round().astype(int).clip(
        DISPLAY_NORMS["min_facings_per_sku"], DISPLAY_NORMS["max_facings_per_sku"]
    )
    df.loc[df["near_expiry"],  "recommended_facings"] = df.loc[df["near_expiry"],  "recommended_facings"].clip(lower=3)
    df.loc[df["low_stock"],    "recommended_facings"] = df.loc[df["low_stock"],    "recommended_facings"].clip(upper=2)

    # Rule 5: Position on shelf (left = higher priority = better traffic)
    df["position_x"] = 0
    for level in [1,2,3,4,5]:
        mask = df["recommended_level"] == level
        x = 0
        for idx in df[mask].sort_values("priority_score", ascending=False).index:
            df.loc[idx, "position_x"] = x
            x += df.loc[idx, "recommended_facings"]

    # Projected uplift
    old_m = df["current_shelf_level"].map(SHELF_LEVEL_MULTIPLIERS)
    new_m = df["recommended_level"].map(SHELF_LEVEL_MULTIPLIERS)
    df["projected_sales_uplift_pct"] = ((new_m / old_m) - 1) * 100

    shelf_depth_ft = DISPLAY_NORMS["shelf_depth_cm"] / 30.48
    pack_w_ft = df["size"].map(lambda s: SIZES[s]["pack_width_cm"] / 30.48)
    df["new_sqft"] = (df["recommended_facings"] * pack_w_ft * shelf_depth_ft).clip(lower=0.05)
    df["projected_sales_inr"]    = df["monthly_sales_inr"] * (1 + df["projected_sales_uplift_pct"] / 100)
    df["projected_sales_per_sqft"] = (df["projected_sales_inr"] / df["new_sqft"]).round(0)

    return df

# ═══════════════════════════════════════════════════════════════════════════════
#  XAI ENGINE  — Explainable AI layer
# ═══════════════════════════════════════════════════════════════════════════════

SCORING_RULES = [
    {"id":"R1","weight":"—",    "factor":"Pack Size Override",   "description":"1.5L packs always assigned to Floor (L1) or Waist (L2). Safety & ergonomics norm."},
    {"id":"R2","weight":"35%",  "factor":"Monthly Sales (INR)",  "description":"Higher revenue contribution → higher priority score → premium shelf level."},
    {"id":"R3","weight":"25%",  "factor":"GMROI",                "description":"Return on inventory investment. Target ≥3.5x. Low GMROI SKUs lose shelf share."},
    {"id":"R4","weight":"20%",  "factor":"Sales per Sqft",       "description":"Space productivity vs ₹18,000/sqft benchmark. Underperformers get reduced facings."},
    {"id":"R5","weight":"10%",  "factor":"Gross Margin %",       "description":"Higher-margin SKUs get slight uplift to maximise shelf profitability."},
    {"id":"R6","weight":"10%",  "factor":"Days to Expiry",       "description":"Fresher stock penalised (FIFO OK). Near-expiry (<30d) → Eye Level override for sell-through."},
    {"id":"R7","weight":"—",    "factor":"Low Stock Override",   "description":"SKUs with <7-day stock cover get facings capped at 2 to avoid phantom inventory."},
    {"id":"R8","weight":"—",    "factor":"Shelf Position",       "description":"Within each level, highest-priority SKUs placed left (primary traffic flow)."},
]

def build_xai_explanation(row):
    """Build a human-readable explanation for a single SKU's placement decision."""
    reasons = []

    reasons.append(f"**Priority Score: {row['priority_score']:.3f} / 1.000**")
    reasons.append(f"• Sales contribution (35%): ₹{row['monthly_sales_inr']:,.0f}/mo → score {row['score_sales']:.3f}")
    reasons.append(f"• GMROI (25%): {row['gmroi']:.2f}x → score {row['score_gmroi']:.3f}")
    reasons.append(f"• Sales/Sqft (20%): ₹{row['sales_per_sqft']:,.0f} → score {row['score_spft']:.3f}")
    reasons.append(f"• Gross Margin (10%): {row['gross_margin_pct']:.1f}% → score {row['score_margin']:.3f}")
    reasons.append(f"• Freshness (10%): {row['days_to_expiry']}d to expiry → score {row['score_fresh']:.3f}")
    reasons.append("")
    reasons.append(f"**Decision:** {row.get('decision_rule', 'Standard allocation')}")

    if row["near_expiry"]:
        reasons.append("⚠️ **Near-expiry override applied** — moved to Eye Level for sell-through.")
    if row["low_stock"]:
        reasons.append("📦 **Low-stock cap applied** — facings reduced to 2 to avoid empty shelf.")
    if row["size"] == "1.5L":
        reasons.append("📏 **Pack-size norm applied** — 1.5L always at Floor/Waist per display guidelines.")

    reasons.append("")
    reasons.append(f"**Expected outcome:** {row['projected_sales_uplift_pct']:+.1f}% uplift vs current position")
    reasons.append(f"({SHELF_LEVEL_NAMES[int(row['current_shelf_level'])]} → {SHELF_LEVEL_NAMES[int(row['recommended_level'])]})")

    return "\n".join(reasons)


def build_portfolio_summary(df):
    """Return a structured data dict for Ollama prompt / display."""
    top3    = df.nlargest(3, "priority_score")[["flavor","size","priority_score","monthly_sales_inr"]].to_dict("records")
    bottom3 = df.nsmallest(3, "priority_score")[["flavor","size","priority_score","monthly_sales_inr"]].to_dict("records")
    ne_skus = df[df["near_expiry"]][["flavor","size","days_to_expiry"]].to_dict("records")
    ls_skus = df[df["low_stock"]][["flavor","size","stock_on_hand"]].to_dict("records")
    return {
        "total_skus":      len(df),
        "total_monthly_sales": int(df["monthly_sales_inr"].sum()),
        "avg_gmroi":       round(df["gmroi"].mean(), 2),
        "avg_sales_per_sqft": int(df["sales_per_sqft"].mean()),
        "benchmark_sales_per_sqft": DISPLAY_NORMS["industry_sales_per_sqft_inr"],
        "avg_uplift_pct":  round(df["projected_sales_uplift_pct"].mean(), 1),
        "near_expiry_count": len(ne_skus),
        "low_stock_count": len(ls_skus),
        "top3_skus":    top3,
        "bottom3_skus": bottom3,
        "near_expiry_skus": ne_skus,
        "low_stock_skus":   ls_skus,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  OLLAMA INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE = "http://localhost:11434"

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []

def ollama_chat(prompt, model="llama3", system=None):
    """Call Ollama generate API; returns streamed text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 600},
    }
    if system:
        payload["system"] = system
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return f"[Ollama error {r.status_code}]"
    except requests.exceptions.ConnectionError:
        return "[Ollama not reachable. Make sure 'ollama serve' is running on localhost:11434]"
    except Exception as e:
        return f"[Error: {str(e)}]"

SYSTEM_PROMPT = """You are JuiceSpace Pro — an expert retail space planning analyst specialising in FMCG juice brands in Indian hypermarkets.
You have access to sales data, GMROI calculations, and shelf allocation decisions made by the space allocation engine.
Your job is to provide clear, actionable, business-friendly explanations of WHY the system made specific decisions.
Always be concise, use retail/trade language, and quantify your points with numbers from the data.
Never make up numbers — only use what is given to you.
Respond in plain English with bullet points where helpful. Max 250 words."""

def build_portfolio_prompt(summary, store_name):
    return f"""Analyse this juice brand shelf allocation for {store_name} and provide a strategic summary:

DATA:
- Total SKUs: {summary['total_skus']} (10 flavors × 4 sizes)
- Monthly portfolio sales: ₹{summary['total_monthly_sales']:,}
- Average GMROI: {summary['avg_gmroi']}x (target: 3.5x)
- Avg Sales/Sqft: ₹{summary['avg_sales_per_sqft']:,} (industry benchmark: ₹{summary['benchmark_sales_per_sqft']:,})
- Projected uplift from new allocation: +{summary['avg_uplift_pct']}%
- Near-expiry SKUs requiring attention: {summary['near_expiry_count']}
- Low stock SKUs needing reorder: {summary['low_stock_count']}

TOP 3 PRIORITY SKUs: {json.dumps(summary['top3_skus'], indent=2)}
BOTTOM 3 PRIORITY SKUs: {json.dumps(summary['bottom3_skus'], indent=2)}
NEAR-EXPIRY: {json.dumps(summary['near_expiry_skus'], indent=2)}
LOW STOCK: {json.dumps(summary['low_stock_skus'], indent=2)}

Provide:
1. Overall portfolio health assessment
2. Key actions for store team (3 bullet points)
3. Revenue risk alerts
4. One strategic recommendation for the brand team"""

def build_sku_prompt(row):
    return f"""Explain in plain English why the space allocation engine placed {row['flavor']} {row['size']} on {SHELF_LEVEL_NAMES[int(row['recommended_level'])]} with {int(row['recommended_facings'])} facings.

SKU DATA:
- Monthly Sales: ₹{row['monthly_sales_inr']:,.0f}
- GMROI: {row['gmroi']:.2f}x
- Sales/Sqft: ₹{row['sales_per_sqft']:,.0f} (benchmark ₹{DISPLAY_NORMS['industry_sales_per_sqft_inr']:,})
- Priority Score: {row['priority_score']:.3f}/1.000
- Gross Margin: {row['gross_margin_pct']:.1f}%
- Days to Expiry: {row['days_to_expiry']}
- Near Expiry: {row['near_expiry']}
- Low Stock: {row['low_stock']}
- Current Level: {SHELF_LEVEL_NAMES[int(row['current_shelf_level'])]}
- Recommended Level: {SHELF_LEVEL_NAMES[int(row['recommended_level'])]}
- Projected Uplift: {row['projected_sales_uplift_pct']:+.1f}%

Explain the decision in 3-4 concise bullet points a store manager would understand."""

# ═══════════════════════════════════════════════════════════════════════════════
#  PLANOGRAM DRAWING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def draw_planogram(allocation_df, store_name="Store", gondolas=2):
    n_levels    = DISPLAY_NORMS["shelf_levels"]
    gondola_w   = DISPLAY_NORMS["gondola_width_cm"]
    total_units = 20 * gondolas

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#FAFAF8")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[10, 2.5], width_ratios=[4, 1],
                           hspace=0.22, wspace=0.04)
    ax_main   = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_kpi    = fig.add_subplot(gs[1, :])

    for ax in [ax_main, ax_legend, ax_kpi]:
        ax.set_facecolor("#FAFAF8")
        for spine in ax.spines.values(): spine.set_visible(False)

    shelf_h, gap = 1.0, 0.18
    max_y = n_levels * (shelf_h + gap) + 1.8

    ax_main.set_xlim(-2, total_units + 2)
    ax_main.set_ylim(-0.6, max_y)
    ax_main.axis("off")

    # Title
    ax_main.text(total_units / 2, max_y - 0.1,
                 f"🍊  JUICESPACE PRO — PLANOGRAM",
                 ha="center", va="top", fontsize=14, fontweight="bold", color="#E65100")
    ax_main.text(total_units / 2, max_y - 0.6,
                 f"{store_name}   |   {gondolas} Gondola{'s' if gondolas>1 else ''}   |   "
                 f"Generated {datetime.now().strftime('%d %b %Y %H:%M')}",
                 ha="center", va="top", fontsize=8.5, color="#9E9E9E", style="italic")

    # Gondola backgrounds
    for g in range(gondolas):
        gx = g * 20
        ax_main.add_patch(FancyBboxPatch(
            (gx - 0.3, -0.25), 20.4, n_levels*(shelf_h+gap)+0.2,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor="#BDBDBD", facecolor="#F5F5F5", zorder=0
        ))
        ax_main.text(gx + 10, -0.45, f"Gondola {g+1}",
                     ha="center", fontsize=8, color="#9E9E9E", style="italic")

    # Shelf boards + level labels
    for level in range(1, n_levels + 1):
        y = (level - 1) * (shelf_h + gap)
        name = SHELF_LEVEL_NAMES[level]
        # Shelf board
        ax_main.plot([-0.5, total_units + 0.5], [y - 0.06, y - 0.06],
                     color="#795548", linewidth=3.5, zorder=2, solid_capstyle="round")
        ax_main.text(-0.7, y + shelf_h/2, name,
                     ha="right", va="center", fontsize=8,
                     color="#E65100" if level == 4 else "#795548",
                     fontweight="bold" if level == 4 else "normal")
        # Eye-level highlight band
        if level == 4:
            ax_main.add_patch(plt.Rectangle(
                (-0.5, y - 0.05), total_units + 1, shelf_h + 0.05,
                facecolor="#FFF3E0", alpha=0.4, zorder=0
            ))

    # Product blocks
    level_x = {l: 0 for l in range(1, 6)}
    for _, row in allocation_df.sort_values(["recommended_level","position_x"]).iterrows():
        level   = int(row["recommended_level"])
        facings = int(row["recommended_facings"])
        y_base  = (level - 1) * (shelf_h + gap)

        pack_w_norm = (SIZES[row["size"]]["pack_width_cm"] / gondola_w) * 20
        block_w = min(facings * pack_w_norm * 0.65, 20 / 3)
        block_w = max(block_w, 0.55)

        x_pos = level_x[level]
        if x_pos + block_w > total_units:
            continue

        col = FLAVOR_COLORS.get(row["flavor"], "#90A4AE")
        pack_h = SIZES[row["size"]]["pack_height_cm"] / 25

        # Main block
        rect = FancyBboxPatch(
            (x_pos + 0.06, y_base + 0.06), block_w - 0.12, min(pack_h, shelf_h - 0.12),
            boxstyle="round,pad=0.04",
            linewidth=0.8, edgecolor="white", facecolor=col, alpha=0.88, zorder=3
        )
        ax_main.add_patch(rect)

        # Near-expiry warning border
        if row.get("near_expiry", False):
            warn = FancyBboxPatch(
                (x_pos + 0.04, y_base + 0.04), block_w - 0.08, min(pack_h, shelf_h - 0.08),
                boxstyle="round,pad=0.04",
                linewidth=2.2, edgecolor="#FF1744", facecolor="none", zorder=4
            )
            ax_main.add_patch(warn)

        # Low-stock diagonal hatch
        if row.get("low_stock", False):
            hatch = FancyBboxPatch(
                (x_pos + 0.06, y_base + 0.06), block_w - 0.12, min(pack_h, shelf_h - 0.12),
                boxstyle="round,pad=0.04",
                linewidth=0, edgecolor="#FF6F00", facecolor="none",
                hatch="///", alpha=0.4, zorder=5
            )
            ax_main.add_patch(hatch)

        # Label
        if block_w > 0.75:
            short_f = row["flavor"][:4].upper()
            short_s = row["size"]
            ax_main.text(
                x_pos + block_w / 2, y_base + min(pack_h, shelf_h - 0.12) / 2 + 0.06,
                f"{short_f}\n{short_s}",
                ha="center", va="center", fontsize=6.2, fontweight="500",
                color="white", zorder=6
            )

        level_x[level] += block_w + 0.08

    # Gondola dividers
    for g in range(1, gondolas):
        ax_main.axvline(x=g*20, color="#9E9E9E", linewidth=1.8, linestyle="--", zorder=1)

    # ── LEGEND ─────────────────────────────────────────────────────────────────
    ax_legend.axis("off")
    ax_legend.text(0.05, 0.99, "LEGEND", fontsize=10, fontweight="bold",
                   color="#E65100", va="top", transform=ax_legend.transAxes)

    ax_legend.text(0.05, 0.93, "FLAVORS", fontsize=8, color="#aaa",
                   va="top", transform=ax_legend.transAxes)
    for i, (flavor, col) in enumerate(FLAVOR_COLORS.items()):
        yp = 0.88 - i * 0.072
        ax_legend.add_patch(plt.Rectangle((0.05, yp - 0.005), 0.10, 0.046,
                             facecolor=col, edgecolor="white", linewidth=0.5,
                             transform=ax_legend.transAxes))
        ax_legend.text(0.20, yp + 0.018, flavor, fontsize=7.5, va="center",
                       transform=ax_legend.transAxes, color="#333")

    # Symbols
    ax_legend.text(0.05, 0.21, "SYMBOLS", fontsize=8, color="#aaa",
                   va="top", transform=ax_legend.transAxes)

    ax_legend.add_patch(plt.Rectangle((0.05, 0.12), 0.10, 0.04, facecolor="none",
                         edgecolor="#FF1744", linewidth=2.2, transform=ax_legend.transAxes))
    ax_legend.text(0.20, 0.14, "Near Expiry (<30d)", fontsize=7, va="center",
                   transform=ax_legend.transAxes, color="#FF1744")

    ax_legend.add_patch(plt.Rectangle((0.05, 0.065), 0.10, 0.04, facecolor="none",
                         edgecolor="#FF6F00", linewidth=0, hatch="///",
                         alpha=0.6, transform=ax_legend.transAxes))
    ax_legend.text(0.20, 0.085, "Low Stock (<7d cover)", fontsize=7, va="center",
                   transform=ax_legend.transAxes, color="#FF6F00")

    # Shelf levels
    ax_legend.text(0.05, 0.005, "SHELF LEVELS", fontsize=8, color="#aaa",
                   va="bottom", transform=ax_legend.transAxes)

    # ── KPI BAR ────────────────────────────────────────────────────────────────
    ax_kpi.axis("off")
    kpis = _summary_kpis(allocation_df)
    kpi_items = [
        ("Total SKUs",       str(kpis["total_skus"]),            "10 flavors × 4 sizes"),
        ("Avg ₹/Sqft",       f"₹{kpis['avg_spf']:,.0f}",         f"Benchmark ₹{DISPLAY_NORMS['industry_sales_per_sqft_inr']:,}"),
        ("Portfolio GMROI",  f"{kpis['avg_gmroi']:.2f}x",         "Target ≥3.5x"),
        ("Eye-Level SKUs",   str(kpis["eye_level_count"]),        "premium position"),
        ("Near-Expiry",      str(kpis["near_expiry"]),            "⚠ immediate action"),
        ("Proj. Uplift",     f"+{kpis['uplift']:.1f}%",           "vs current placement"),
    ]
    n = len(kpi_items)
    ax_kpi.plot([0, 1], [0.95, 0.95], color="#E0E0E0", linewidth=0.8, transform=ax_kpi.transAxes)
    for i, (label, value, sub) in enumerate(kpi_items):
        xp = i / n + 0.5/n
        ax_kpi.text(xp, 0.88, label, ha="center", va="top", fontsize=8,
                    color="#999", transform=ax_kpi.transAxes)
        ax_kpi.text(xp, 0.58, value, ha="center", va="top", fontsize=17,
                    fontweight="bold", color="#1A1A1A", transform=ax_kpi.transAxes)
        ax_kpi.text(xp, 0.18, sub, ha="center", va="top", fontsize=7.5,
                    color="#BDBDBD", transform=ax_kpi.transAxes)

    plt.tight_layout(pad=0.4)
    return fig


def _summary_kpis(df):
    return {
        "total_skus":      len(df),
        "avg_spf":         df["projected_sales_per_sqft"].mean(),
        "avg_gmroi":       df["gmroi"].mean(),
        "eye_level_count": (df["recommended_level"] == 4).sum(),
        "near_expiry":     int(df["near_expiry"].sum()),
        "uplift":          df["projected_sales_uplift_pct"].mean(),
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION TEXT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(allocation_df):
    recs = []
    ne = allocation_df[allocation_df["near_expiry"]]
    ls = allocation_df[allocation_df["low_stock"]]
    if len(ne):
        skus = ", ".join([f"{r.flavor} {r.size}" for _, r in ne.iterrows()])
        recs.append(("⚠️ Near-Expiry Alert",
                     f"{len(ne)} SKUs expire within 30 days ({skus}). "
                     "Moved to Eye Level. Consider markdown / BOGOF promotion.",
                     "alert"))
    if len(ls):
        recs.append(("📦 Low Stock Warning",
                     f"{len(ls)} SKUs below 7-day cover. Raise PO immediately. "
                     "Facings capped at 2 to prevent empty shelf showing.",
                     "alert"))
    top = allocation_df.nlargest(3,"priority_score")[["flavor","size","priority_score"]]
    recs.append(("⭐ Star SKUs",
                 "Eye-level priority: " + ", ".join([f"{r.flavor} {r.size}" for _, r in top.iterrows()]) +
                 ". Never let these go OOS — they drive 35%+ of revenue.",
                 "success"))
    low_g = allocation_df[allocation_df["gmroi"] < DISPLAY_NORMS["min_gmroi"]]
    if len(low_g):
        recs.append(("📉 GMROI Below Threshold",
                     f"{len(low_g)} SKUs below {DISPLAY_NORMS['min_gmroi']}x GMROI. "
                     "Review pricing, reduce space, or delist.",
                     "alert"))
    avg_up = allocation_df["projected_sales_uplift_pct"].mean()
    recs.append(("📈 Projected Revenue Uplift",
                 f"Recommended placement projects +{avg_up:.1f}% sales vs current "
                 "arrangement, based on shelf-level conversion multipliers.",
                 "info"))
    return recs

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("## 🍊 JuiceSpace Pro")
        st.caption("Explainable AI · Space Allocation Agent")
        st.divider()

        st.markdown('<div class="section-header">Store Profile</div>', unsafe_allow_html=True)
        store_name = st.text_input("Store Name", value="HyperFresh — Andheri")
        store_city = st.selectbox("City", ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad"])
        store_type = st.selectbox("Format", ["Hypermarket","Supermarket","Cash & Carry"])
        gondolas   = st.slider("Gondolas", 1, 6, 2)
        total_sqft = st.number_input("Juice Section (sqft)", 50, 2000, 300)

        st.divider()
        st.markdown('<div class="section-header">Ollama (Local AI)</div>', unsafe_allow_html=True)
        ollama_ok, ollama_models = check_ollama()
        st.session_state.ollama_available = ollama_ok

        if ollama_ok:
            st.success(f"✅ Ollama connected — {len(ollama_models)} model(s)")
            if ollama_models:
                chosen = st.selectbox("Model", ollama_models,
                                      index=0 if "llama3" not in ollama_models else ollama_models.index("llama3"))
                st.session_state.ollama_model = chosen
        else:
            st.warning("⚠️ Ollama not detected")
            st.caption("Start with: `ollama serve`")
            st.caption("Then pull a model: `ollama pull llama3`")
            st.info("Rule-based XAI explanations will still work without Ollama.")

        st.divider()
        st.markdown('<div class="section-header">Update Schedule</div>', unsafe_allow_html=True)
        frequency = st.selectbox("Replanning", ["Daily","Weekly","Monthly"])
        next_run = _next_run(frequency)
        st.info(f"⏱ Next run: **{next_run.strftime('%d %b %Y')}**")

        st.divider()
        st.markdown('<div class="section-header">Display Norms</div>', unsafe_allow_html=True)
        DISPLAY_NORMS["min_facings_per_sku"] = st.slider("Min Facings", 1, 4, 2)
        DISPLAY_NORMS["max_facings_per_sku"] = st.slider("Max Facings", 4, 12, 8)
        DISPLAY_NORMS["target_gmroi"]        = st.number_input("Target GMROI", 1.0, 10.0, 3.5, 0.1)
        DISPLAY_NORMS["near_expiry_threshold_days"] = st.slider("Near-Expiry Threshold (days)", 10, 60, 30)

        st.divider()
        run = st.button("🚀 Run Allocation Agent", use_container_width=True)

        return {
            "store_name": store_name, "store_city": store_city,
            "store_type": store_type, "gondolas": gondolas,
            "total_sqft": total_sqft, "frequency": frequency, "run": run,
        }

def _next_run(freq):
    now = datetime.now()
    return now + (timedelta(days=1) if freq=="Daily" else timedelta(weeks=1) if freq=="Weekly" else timedelta(days=30))

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def tab_dashboard(df, store_config):
    # Data source badge
    source = st.session_state.data_source
    badge  = "SYNTHETIC DATA" if source == "synthetic" else "UPLOADED DATA"
    badge_cls = "badge-synth" if source == "synthetic" else "badge-real"
    st.markdown(
        f'<span class="{badge_cls}">{badge}</span>&nbsp; '
        '<span style="font-size:12px;color:#999;">Switch to the 📥 Data tab to upload real data</span>',
        unsafe_allow_html=True
    )
    st.subheader("📊 Performance Dashboard")

    total_sales = df["monthly_sales_inr"].sum()
    avg_spf     = df["sales_per_sqft"].mean()
    avg_gmroi   = df["gmroi"].mean()
    bench       = DISPLAY_NORMS["industry_sales_per_sqft_inr"]
    avg_uplift  = df["projected_sales_uplift_pct"].mean()

    kc = st.columns(5)
    _kpi(kc[0], "Monthly Sales",     f"₹{total_sales/1e5:.1f}L",     "All 40 SKUs", "neutral")
    _kpi(kc[1], "Avg Sales/Sqft",    f"₹{avg_spf:,.0f}",
         f"{'▲' if avg_spf>=bench else '▼'} vs ₹{bench:,} benchmark",
         "good" if avg_spf >= bench else "bad")
    _kpi(kc[2], "Portfolio GMROI",   f"{avg_gmroi:.2f}x",
         f"Target {DISPLAY_NORMS['target_gmroi']}x",
         "good" if avg_gmroi >= DISPLAY_NORMS["target_gmroi"] else "bad")
    _kpi(kc[3], "Proj. Uplift",      f"+{avg_uplift:.1f}%",            "new vs current", "good")
    ne_count = int(df["near_expiry"].sum())
    _kpi(kc[4], "Near-Expiry SKUs",  str(ne_count),                    "≤30 days", "bad" if ne_count else "good")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Monthly Sales by Flavor**")
        flavor_sales = df.groupby("flavor")["monthly_sales_inr"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#FAFAF8")
        bars = ax.barh(flavor_sales.index, flavor_sales.values/1000,
                       color=[FLAVOR_COLORS[f] for f in flavor_sales.index],
                       edgecolor="white", linewidth=0.5)
        ax.set_xlabel("₹ (thousands)", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        for b, v in zip(bars, flavor_sales.values):
            ax.text(v/1000 + 0.5, b.get_y()+b.get_height()/2,
                    f"₹{v/1000:.0f}K", va="center", fontsize=7.5, color="#555")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.markdown("**GMROI vs Sales/Sqft (bubble = monthly sales)**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#FAFAF8")
        sa = df.groupby("size").agg(spf=("sales_per_sqft","mean"), gmroi=("gmroi","mean"),
                                     vol=("monthly_sales_inr","sum")).reset_index()
        colors_map = {"200ml":"#B3E5FC","500ml":"#29B6F6","1L":"#0288D1","1.5L":"#01579B"}
        for _, r in sa.iterrows():
            ax.scatter(r["spf"], r["gmroi"], s=r["vol"]/500, alpha=0.75,
                       color=colors_map[r["size"]], edgecolors="#555", linewidth=0.5)
            ax.annotate(r["size"], (r["spf"], r["gmroi"]),
                        textcoords="offset points", xytext=(7,4), fontsize=9, fontweight="bold")
        ax.axvline(bench, color="#F57C00", linestyle="--", linewidth=1, alpha=0.7, label=f"Benchmark ₹{bench:,}")
        ax.axhline(DISPLAY_NORMS["target_gmroi"], color="#43A047", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Target GMROI {DISPLAY_NORMS['target_gmroi']}x")
        ax.set_xlabel("Sales per Sqft (₹)", fontsize=9); ax.set_ylabel("GMROI (x)", fontsize=9)
        ax.legend(fontsize=7.5); ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**SKU Detail**")
    show = df[["flavor","size","monthly_units","monthly_sales_inr","sales_per_sqft",
               "gmroi","vs_benchmark_pct","gross_margin_pct","days_to_expiry","near_expiry","low_stock"]].copy()
    show = show.rename(columns={
        "monthly_units":"Units/Mo","monthly_sales_inr":"Sales ₹","sales_per_sqft":"₹/Sqft",
        "gmroi":"GMROI","vs_benchmark_pct":"vs Bench %","gross_margin_pct":"Margin %",
        "days_to_expiry":"Expiry Days","near_expiry":"⚠ Expiry","low_stock":"📦 Low Stock"
    })
    st.dataframe(
        show.style
            .format({"Sales ₹":"₹{:,.0f}","₹/Sqft":"₹{:,.0f}","GMROI":"{:.2f}x",
                     "vs Bench %":"{:+.1f}%","Margin %":"{:.1f}%"})
            .background_gradient(subset=["₹/Sqft","GMROI"], cmap="YlGn"),
        use_container_width=True, height=340
    )

def _kpi(col, label, value, delta, kind):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta kpi-{kind}">{delta}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: XAI — EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════════════

def tab_xai(allocation_df, store_config):
    st.subheader("🧠 Explainable AI — Why These Decisions?")
    st.markdown(
        '<div class="info-box">This tab shows <b>exactly why</b> the allocation engine placed each SKU '
        'where it did. Every decision is traceable to a rule, weight, or override. '
        'No black boxes — every number is auditable.</div>',
        unsafe_allow_html=True
    )

    # ── SCORING RULES TABLE ─────────────────────────────────────────────────
    st.markdown("### 📋 Scoring Rules & Weights")
    st.markdown("The priority score (0–1) drives shelf level, facings, and position. Here's every factor:")

    cols = st.columns([1, 1.5, 2.5, 1])
    for hdr, w in zip(["Rule","Weight","Factor","Description"], [1,1.5,2.5,1]):
        pass
    rule_df = pd.DataFrame(SCORING_RULES)
    st.dataframe(
        rule_df.rename(columns={"id":"Rule","weight":"Weight","factor":"Factor","description":"What it means"}),
        use_container_width=True, hide_index=True, height=320
    )

    # ── SCORE DECOMPOSITION CHART ───────────────────────────────────────────
    st.markdown("### 📊 Score Decomposition — Top 10 SKUs")
    top10 = allocation_df.nlargest(10, "priority_score")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#FAFAF8")

    labels = [f"{r.flavor[:5]}\n{r.size}" for _, r in top10.iterrows()]
    x = np.arange(len(labels))
    w = 0.15

    components = [
        ("score_sales",  "35% Sales",  "#F57C00"),
        ("score_gmroi",  "25% GMROI",  "#1976D2"),
        ("score_spft",   "20% ₹/Sqft", "#43A047"),
        ("score_margin", "10% Margin", "#7B1FA2"),
        ("score_fresh",  "10% Fresh",  "#E91E63"),
    ]
    for i, (col, label, color) in enumerate(components):
        ax.bar(x + i*w, top10[col].values, w*0.85, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + w*2); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score Contribution", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=5)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── DECISION RULES APPLIED ──────────────────────────────────────────────
    st.markdown("### 🔍 Decision Rules Applied per SKU")
    dr_df = allocation_df[["flavor","size","priority_score","recommended_level",
                            "recommended_facings","decision_rule"]].copy()
    dr_df["Shelf"] = dr_df["recommended_level"].map(SHELF_LEVEL_NAMES)
    dr_df = dr_df.rename(columns={
        "flavor":"Flavor","size":"Size","priority_score":"Score",
        "recommended_facings":"Facings","decision_rule":"Decision Rule"
    })
    st.dataframe(
        dr_df[["Flavor","Size","Score","Shelf","Facings","Decision Rule"]]
            .sort_values("Score", ascending=False)
            .style.format({"Score":"{:.3f}"})
            .background_gradient(subset=["Score"], cmap="YlOrRd"),
        use_container_width=True, height=380
    )

    # ── SKU DEEP-DIVE ───────────────────────────────────────────────────────
    st.markdown("### 🔬 SKU Deep-Dive Explanation")
    c1, c2 = st.columns([1, 2])
    with c1:
        sel_flavor = st.selectbox("Flavor", FLAVORS)
        sel_size   = st.selectbox("Size", SIZE_ORDER)
    with c2:
        sku_row = allocation_df[(allocation_df["flavor"]==sel_flavor) & (allocation_df["size"]==sel_size)]
        if len(sku_row) == 0:
            st.warning("SKU not found in current allocation."); return

        row = sku_row.iloc[0]

        # Rule-based explanation always shown
        st.markdown("**📐 Rule-Based Explanation (always available)**")
        explanation = build_xai_explanation(row)
        st.markdown(f'<div class="xai-card">{explanation.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)

        # Score bars
        st.markdown("**Score Components**")
        for comp, label, pct in [
            ("score_sales", "Sales (35%)", 0.35),
            ("score_gmroi", "GMROI (25%)", 0.25),
            ("score_spft",  "₹/Sqft (20%)", 0.20),
            ("score_margin","Margin (10%)", 0.10),
            ("score_fresh", "Freshness (10%)", 0.10),
        ]:
            val = row[comp]; filled = int(val / pct * 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:3px 0;">'
                f'<span style="width:120px;font-size:12px;color:#555;">{label}</span>'
                f'<div class="score-bar-bg"><div class="score-bar" style="width:{min(filled,100)}%;"></div></div>'
                f'<span style="width:50px;font-size:12px;font-family:monospace;color:#333;">{val:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Ollama LLM explanation
        if st.session_state.ollama_available:
            st.markdown("**🤖 AI Narrative Explanation (Ollama)**")
            if st.button(f"Ask {st.session_state.ollama_model} to explain this SKU"):
                with st.spinner(f"Asking {st.session_state.ollama_model}..."):
                    prompt = build_sku_prompt(row)
                    response = ollama_chat(prompt, st.session_state.ollama_model, SYSTEM_PROMPT)
                    st.session_state.last_ollama_response = response
                    st.session_state.xai_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "sku": f"{sel_flavor} {sel_size}",
                        "response": response[:120] + "..."
                    })

            if st.session_state.last_ollama_response:
                st.markdown(f'<div class="ollama-card">'
                            f'<div style="font-size:11px;color:#555;margin-bottom:6px;">🤖 {st.session_state.ollama_model.upper()} says:</div>'
                            f'{st.session_state.last_ollama_response}'
                            f'</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">🤖 Start Ollama to get AI narrative explanations.<br>'
                        '<code>ollama serve</code> → <code>ollama pull llama3</code></div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════════

def tab_allocation(allocation_df, store_config):
    st.subheader("🗂 Recommended Shelf Allocation")

    # Ollama portfolio summary
    if st.session_state.ollama_available:
        if st.button(f"🤖 Generate AI Portfolio Commentary ({st.session_state.ollama_model})"):
            summary = build_portfolio_summary(allocation_df)
            prompt  = build_portfolio_prompt(summary, store_config.get("store_name",""))
            with st.spinner("Generating AI commentary..."):
                response = ollama_chat(prompt, st.session_state.ollama_model, SYSTEM_PROMPT)
            st.markdown(f'<div class="ollama-card">'
                        f'<div style="font-size:11px;color:#555;margin-bottom:6px;">'
                        f'🤖 {st.session_state.ollama_model.upper()} PORTFOLIO ANALYSIS</div>'
                        f'{response}</div>', unsafe_allow_html=True)

    recs = generate_recommendations(allocation_df)
    for title, msg, kind in recs:
        st.markdown(f'<div class="{kind}-box"><b>{title}</b><br>{msg}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Shelf Level Distribution**")
    lc = st.columns(5)
    for i, (level, name) in enumerate(sorted(SHELF_LEVEL_NAMES.items(), reverse=True)):
        cnt = (allocation_df["recommended_level"] == level).sum()
        mult = SHELF_LEVEL_MULTIPLIERS[level]
        lc[i].metric(name, f"{cnt} SKUs", delta=f"{mult}x conversion")

    st.markdown("---")
    show_cols = ["flavor","size","recommended_level","recommended_facings",
                 "priority_score","projected_sales_per_sqft","projected_sales_uplift_pct",
                 "near_expiry","low_stock"]
    ad = allocation_df[show_cols].copy()
    ad["recommended_level"] = ad["recommended_level"].map(SHELF_LEVEL_NAMES)
    ad = ad.rename(columns={
        "recommended_level":"Shelf Level","recommended_facings":"Facings",
        "priority_score":"Priority","projected_sales_per_sqft":"Proj ₹/Sqft",
        "projected_sales_uplift_pct":"Uplift %","near_expiry":"⚠ Expiry","low_stock":"📦 Low Stock"
    })
    st.dataframe(
        ad.sort_values("Priority", ascending=False)
          .style.format({"Priority":"{:.3f}","Proj ₹/Sqft":"₹{:,.0f}","Uplift %":"{:+.1f}%"})
          .background_gradient(subset=["Proj ₹/Sqft","Priority"], cmap="YlOrRd"),
        use_container_width=True, height=400
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: PLANOGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def tab_planogram(allocation_df, store_config):
    st.subheader("📐 Planogram — Line Drawing for Store Teams")
    st.markdown('<div class="info-box">Print this and hand to the shelf-filler team. '
                '<b>Red border</b> = near-expiry (sell first). '
                '<b>Hatch</b> = low stock (do not spread facings). '
                '<b>Orange band</b> = Eye Level (prime zone).</div>',
                unsafe_allow_html=True)

    gondolas = store_config.get("gondolas", 2)
    fig = draw_planogram(allocation_df, store_config.get("store_name","Store"), gondolas)
    st.pyplot(fig, use_container_width=True); plt.close()

    buf = io.BytesIO()
    fig2 = draw_planogram(allocation_df, store_config.get("store_name","Store"), gondolas)
    fig2.savefig(buf, format="pdf", bbox_inches="tight", dpi=150, facecolor="#FAFAF8")
    plt.close(fig2); buf.seek(0)
    st.download_button("⬇️ Download Planogram (PDF)", data=buf,
                       file_name=f"planogram_{store_config.get('store_name','store').replace(' ','_')}.pdf",
                       mime="application/pdf")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def tab_data():
    st.subheader("📥 Sales Data Management")

    # ── FORMAT GUIDE ─────────────────────────────────────────────────────────
    with st.expander("📖 File Format Guide — Read Before Uploading", expanded=True):
        st.markdown("""
**Required CSV columns** (the engine cannot run without these):

| Column | Type | Valid Values | Example |
|---|---|---|---|
| `flavor` | text | Mixed Fruit, Orange, Mango, Apple, Guava, Litchi, Pineapple, Pomegranate, Grape, Cranberry | Orange |
| `size` | text | 200ml, 500ml, 1L, 1.5L | 500ml |
| `monthly_units` | integer | > 0 | 450 |
| `mrp` | integer | > 0 | 55 |
| `cost` | integer | > 0, < mrp | 36 |
| `stock_on_hand` | integer | ≥ 0 | 120 |
| `days_to_expiry` | integer | > 0 | 95 |
| `current_facings` | integer | 1–12 | 3 |
| `current_shelf_level` | integer | 1–5 (1=Floor, 5=Top) | 3 |

**Rules:**
- One row per SKU (flavor + size combination). Max 40 rows (10 flavors × 4 sizes).
- Missing `mrp`/`cost` columns → engine will use default price list.
- Missing `current_facings`/`current_shelf_level` → defaults to 3 (waist level).
- Flavor names must match exactly (case-insensitive accepted).
- File must be `.csv` (comma-separated). Excel → Save As CSV before uploading.
        """)

        col1, col2 = st.columns(2)
        with col1:
            template = get_upload_template()
            st.download_button(
                "⬇️ Download Blank Template (CSV)",
                template.to_csv(index=False),
                "juicespace_upload_template.csv", "text/csv",
                use_container_width=True
            )
        with col2:
            sample = generate_sample_sales()
            st.download_button(
                "⬇️ Download Sample Data (CSV)",
                sample.to_csv(index=False),
                "juicespace_sample_data.csv", "text/csv",
                use_container_width=True
            )

    st.markdown("---")

    # ── UPLOAD ───────────────────────────────────────────────────────────────
    st.markdown("### Upload Your Data")
    uploaded = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        help="Use the template above to ensure correct format"
    )

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            raw.columns = [c.strip().lower().replace(" ","_") for c in raw.columns]

            # Validate required columns
            required = ["flavor","size","monthly_units"]
            missing = [c for c in required if c not in raw.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.stop()

            # Normalise flavor names (case-insensitive match)
            flavor_map = {f.lower().strip(): f for f in FLAVORS}
            raw["flavor"] = raw["flavor"].str.strip().str.lower().map(flavor_map)
            invalid_flavors = raw[raw["flavor"].isna()]
            if len(invalid_flavors):
                st.error(f"❌ Unrecognised flavors in rows: {invalid_flavors.index.tolist()}. "
                         f"Valid: {FLAVORS}")
                st.stop()

            # Normalise sizes
            raw["size"] = raw["size"].str.strip()
            invalid_sizes = raw[~raw["size"].isin(SIZE_ORDER)]
            if len(invalid_sizes):
                st.error(f"❌ Invalid sizes: {invalid_sizes['size'].unique()}. Valid: {SIZE_ORDER}")
                st.stop()

            # Fill optional columns with defaults
            for col, default in [("mrp", None), ("cost", None),
                                  ("stock_on_hand", 80), ("days_to_expiry", 120),
                                  ("current_facings", 3), ("current_shelf_level", 3)]:
                if col not in raw.columns:
                    if col == "mrp":
                        raw["mrp"] = raw["size"].map({s: SIZES[s]["mrp"] for s in SIZE_ORDER})
                    elif col == "cost":
                        raw["cost"] = raw["size"].map({s: SIZES[s]["cost"] for s in SIZE_ORDER})
                    else:
                        raw[col] = default

            raw["gross_margin"]        = raw["mrp"] - raw["cost"]
            raw["monthly_sales_inr"]   = raw["monthly_units"] * raw["mrp"]
            raw["monthly_margin_inr"]  = raw["monthly_units"] * raw["gross_margin"]

            # Preview
            st.success(f"✅ Loaded {len(raw)} SKUs successfully")
            st.dataframe(raw, use_container_width=True, height=250)

            if st.button("✅ Use This Data for Allocation"):
                st.session_state.sales_data  = raw
                st.session_state.data_source = "uploaded"
                st.session_state.allocation  = None  # force re-run
                st.success("Data applied! Click 'Run Allocation Agent' in the sidebar.")

        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")

    # ── CURRENT DATA EDITOR ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Edit Current Data Inline")
    source_badge = "SYNTHETIC" if st.session_state.data_source == "synthetic" else "UPLOADED"
    st.markdown(f'<span class="badge-{"synth" if source_badge=="SYNTHETIC" else "real"}">{source_badge}</span>',
                unsafe_allow_html=True)

    edit_cols = ["flavor","size","monthly_units","stock_on_hand","days_to_expiry",
                 "current_facings","current_shelf_level"]
    available = [c for c in edit_cols if c in st.session_state.sales_data.columns]
    edited = st.data_editor(
        st.session_state.sales_data[available],
        use_container_width=True, num_rows="fixed", height=480
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save Edits"):
            merged = edited.copy()
            merged["mrp"]  = merged["size"].map({s: SIZES[s]["mrp"]  for s in SIZE_ORDER})
            merged["cost"] = merged["size"].map({s: SIZES[s]["cost"] for s in SIZE_ORDER})
            merged["gross_margin"]       = merged["mrp"] - merged["cost"]
            merged["monthly_sales_inr"]  = merged["monthly_units"] * merged["mrp"]
            merged["monthly_margin_inr"] = merged["monthly_units"] * merged["gross_margin"]
            st.session_state.sales_data  = merged
            st.session_state.allocation  = None
            st.success("Saved! Re-run the agent to update allocation.")
    with c2:
        if st.button("🔄 Reset to Synthetic Data"):
            st.session_state.sales_data  = generate_sample_sales()
            st.session_state.data_source = "synthetic"
            st.session_state.allocation  = None
            st.success("Reset to synthetic data.")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def tab_schedule(store_config):
    st.subheader("📅 Scheduling & Audit Log")
    c1, c2 = st.columns(2)
    with c1:
        freq = store_config.get("frequency","Weekly")
        st.markdown(f"""
        <div class="kpi-card" style="text-align:left;">
          <div class="kpi-label">Active Schedule</div>
          <div class="kpi-value" style="font-size:20px;">{freq} Replanning</div>
          <div class="kpi-delta kpi-neutral">Next run: {_next_run(freq).strftime('%d %b %Y')}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
**Scheduling logic:**
- **Daily** — runs every morning at 06:00. Best for promo periods or high-churn stores.
- **Weekly** — runs every Monday. Recommended default for hypermarkets.
- **Monthly** — runs on 1st of month. Suitable for stable portfolios.

On each scheduled run, the agent re-ingests latest sales, stock, and expiry data,
recomputes all metrics, and outputs a new planogram automatically.
        """)
        if st.session_state.xai_log:
            st.markdown("**XAI Query Log**")
            st.dataframe(pd.DataFrame(st.session_state.xai_log), use_container_width=True)
    with c2:
        st.markdown("**Allocation Run Log**")
        log = st.session_state.schedule_log
        if log:
            st.dataframe(pd.DataFrame(log).tail(15), use_container_width=True)
        else:
            st.info("No runs yet. Click 'Run Allocation Agent' in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    store_config = sidebar()

    if store_config["run"]:
        with st.spinner("🍊 Running space allocation agent..."):
            metrics_df    = compute_metrics(st.session_state.sales_data, store_config)
            allocation_df = run_allocation(metrics_df, store_config)
            st.session_state.allocation = allocation_df
            st.session_state.store_config = store_config
            st.session_state.schedule_log.append({
                "Run Time":  datetime.now().strftime("%d %b %Y %H:%M"),
                "Store":     store_config["store_name"],
                "Data":      st.session_state.data_source.upper(),
                "SKUs":      len(allocation_df),
                "Avg Uplift":f"+{allocation_df['projected_sales_uplift_pct'].mean():.1f}%",
                "Frequency": store_config["frequency"],
            })
        st.success(f"✅ Allocation complete for **{store_config['store_name']}**")

    # Always compute on load so app shows results immediately
    if st.session_state.allocation is None:
        metrics_df    = compute_metrics(st.session_state.sales_data, store_config)
        allocation_df = run_allocation(metrics_df, store_config)
        st.session_state.allocation = allocation_df
        st.session_state.store_config = store_config
    else:
        allocation_df = st.session_state.allocation

    # Header
    source_label = "SYNTHETIC DATA" if st.session_state.data_source == "synthetic" else "LIVE DATA"
    st.markdown(f"""
    <h2 style="margin-bottom:2px;color:#E65100;">
      🍊 JuiceSpace Pro
      <span style="font-size:13px;font-weight:400;color:#aaa;margin-left:12px;">
        v2 · Explainable AI Edition
      </span>
    </h2>
    <p style="color:#9E9E9E;margin-top:2px;font-size:12px;">
      {store_config['store_name']} · {store_config['store_city']} · {store_config['store_type']} ·
      {len(allocation_df)} SKUs · <b style="color:#1976D2">{source_label}</b> ·
      {datetime.now().strftime('%d %b %Y')} · Next update: {_next_run(store_config['frequency']).strftime('%d %b')}
    </p>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📊 Dashboard",
        "🧠 Explainable AI",
        "🗂 Allocation",
        "📐 Planogram",
        "📥 Data",
        "📅 Schedule",
    ])
    with tabs[0]: tab_dashboard(allocation_df, store_config)
    with tabs[1]: tab_xai(allocation_df, store_config)
    with tabs[2]: tab_allocation(allocation_df, store_config)
    with tabs[3]: tab_planogram(allocation_df, store_config)
    with tabs[4]: tab_data()
    with tabs[5]: tab_schedule(store_config)


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import base64
import math

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Carbon Footprint Calculator",
    page_icon="🌍",
    initial_sidebar_state="collapsed"
)

# ── Sample Feature Template ─────────────────────────────────────────────────
sample = {
    'Body Type': 2, 'Sex': 0, 'How Often Shower': 1, 'Social Activity': 2,
    'Monthly Grocery Bill': 230, 'Frequency of Traveling by Air': 2,
    'Vehicle Monthly Distance Km': 210, 'Waste Bag Size': 2,
    'Waste Bag Weekly Count': 4, 'How Long TV PC Daily Hour': 7,
    'How Many New Clothes Monthly': 26, 'How Long Internet Daily Hour': 1,
    'Energy efficiency': 0, 'Do You Recyle_Paper': 0, 'Do You Recyle_Plastic': 0,
    'Do You Recyle_Glass': 0, 'Do You Recyle_Metal': 1, 'Cooking_with_stove': 1,
    'Cooking_with_oven': 1, 'Cooking_with_microwave': 0, 'Cooking_with_grill': 0,
    'Cooking_with_airfryer': 1, 'Diet_omnivore': 0, 'Diet_pescatarian': 1,
    'Diet_vegan': 0, 'Diet_vegetarian': 0, 'Heating Energy Source_coal': 1,
    'Heating Energy Source_electricity': 0, 'Heating Energy Source_natural gas': 0,
    'Heating Energy Source_wood': 0, 'Transport_private': 0, 'Transport_public': 1,
    'Transport_walk/bicycle': 0, 'Vehicle Type_None': 1, 'Vehicle Type_diesel': 0,
    'Vehicle Type_electric': 0, 'Vehicle Type_hybrid': 0, 'Vehicle Type_lpg': 0,
    'Vehicle Type_petrol': 0
}

# ── Preprocessing ────────────────────────────────────────────────────────────
def input_preprocessing(data):
    data["Body Type"] = data["Body Type"].map({'underweight':0, 'normal':1, 'overweight':2, 'obese':3})
    data["Sex"] = data["Sex"].map({'female':0, 'male':1})
    data = pd.get_dummies(data, columns=["Diet","Heating Energy Source","Transport","Vehicle Type"], dtype=int)
    data["How Often Shower"] = data["How Often Shower"].map({'less frequently':0, 'daily':1, "twice a day":2, "more frequently":3})
    data["Social Activity"] = data["Social Activity"].map({'never':0, 'sometimes':1, "often":2})
    data["Frequency of Traveling by Air"] = data["Frequency of Traveling by Air"].map({'never':0, 'rarely':1, "frequently":2, "very frequently":3})
    data["Waste Bag Size"] = data["Waste Bag Size"].map({'small':0, 'medium':1, "large":2, "extra large":3})
    data["Energy efficiency"] = data["Energy efficiency"].map({'No':0, 'Sometimes':1, "Yes":2})
    return data

def hesapla(model, ss, sample_df):
    copy_df = sample_df.copy()
    travels = copy_df[["Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
                        'Transport_private', 'Transport_public', 'Transport_walk/bicycle',
                        'Vehicle Type_None', 'Vehicle Type_diesel', 'Vehicle Type_electric',
                        'Vehicle Type_hybrid', 'Vehicle Type_lpg', 'Vehicle Type_petrol']]
    copy_df[list(set(copy_df.columns) - set(travels.columns))] = 0
    travel = np.exp(model.predict(ss.transform(copy_df)))

    copy_df = sample_df.copy()
    energys = copy_df[['Heating Energy Source_coal','How Often Shower','How Long TV PC Daily Hour',
                        'Heating Energy Source_electricity','How Long Internet Daily Hour',
                        'Heating Energy Source_natural gas', 'Cooking_with_stove',
                        'Cooking_with_oven', 'Cooking_with_microwave', 'Cooking_with_grill',
                        'Cooking_with_airfryer', 'Heating Energy Source_wood','Energy efficiency']]
    copy_df[list(set(copy_df.columns) - set(energys.columns))] = 0
    energy = np.exp(model.predict(ss.transform(copy_df)))

    copy_df = sample_df.copy()
    wastes = copy_df[['Do You Recyle_Paper','How Many New Clothes Monthly',
                       'Waste Bag Size','Waste Bag Weekly Count',
                       'Do You Recyle_Plastic','Do You Recyle_Glass',
                       'Do You Recyle_Metal','Social Activity']]
    copy_df[list(set(copy_df.columns) - set(wastes.columns))] = 0
    waste = np.exp(model.predict(ss.transform(copy_df)))

    copy_df = sample_df.copy()
    diets = copy_df[['Diet_omnivore','Diet_pescatarian','Diet_vegan','Diet_vegetarian',
                      'Monthly Grocery Bill','Transport_private','Transport_public',
                      'Transport_walk/bicycle','Heating Energy Source_coal',
                      'Heating Energy Source_electricity','Heating Energy Source_natural gas',
                      'Heating Energy Source_wood']]
    copy_df[list(set(copy_df.columns) - set(diets.columns))] = 0
    diet = np.exp(model.predict(ss.transform(copy_df)))

    return {"🚗 Travel": travel[0], "⚡ Energy": energy[0], "🗑️ Waste": waste[0], "🥗 Diet": diet[0]}


# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --bg-glass: rgba(255, 255, 255, 0.03);
    --border-glass: rgba(255, 255, 255, 0.08);
    --text-primary: #f0f4f8;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-green: #10b981;
    --accent-green-light: #34d399;
    --accent-emerald: #059669;
    --accent-teal: #14b8a6;
    --accent-blue: #3b82f6;
    --gradient-primary: linear-gradient(135deg, #10b981 0%, #14b8a6 50%, #3b82f6 100%);
    --gradient-card: linear-gradient(145deg, rgba(16, 185, 129, 0.08), rgba(20, 184, 166, 0.04), rgba(59, 130, 246, 0.08));
    --shadow-glow: 0 0 40px rgba(16, 185, 129, 0.15);
    --radius: 16px;
    --radius-lg: 24px;
}

/* ── Global Reset ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(16, 185, 129, 0.08), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(59, 130, 246, 0.06), transparent),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(20, 184, 166, 0.04), transparent);
    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Sidebar hide ── */
[data-testid="stSidebar"] { display: none !important; }

/* ── Tab Container Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    padding: 6px !important;
    gap: 4px !important;
    backdrop-filter: blur(20px) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(16, 185, 129, 0.1) !important;
    color: var(--accent-green-light) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-primary) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Input Styling ── */
[data-testid="stNumberInput"] > div,
.stSelectbox > div,
.stMultiSelect > div,
.stSlider > div {
    font-family: 'Inter', sans-serif !important;
}

[data-baseweb="input"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease !important;
}

[data-baseweb="input"]:focus-within {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
}

[data-baseweb="select"] > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent-green) !important;
}

[data-baseweb="popover"] > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
}

[data-baseweb="menu"] {
    background: var(--bg-secondary) !important;
}

[data-baseweb="menu"] li {
    color: var(--text-primary) !important;
}

[data-baseweb="menu"] li:hover {
    background: rgba(16, 185, 129, 0.1) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-green) !important;
    border-color: var(--accent-green) !important;
}

[data-baseweb="slider"] > div > div > div {
    background: var(--accent-green) !important;
}

/* ── Labels ── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label,
[data-testid="stWidgetLabel"] label, .stMarkdown label {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* ── Button Styling ── */
.stButton > button {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 14px 32px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Markdown Text ── */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

p, li, span, div {
    color: var(--text-secondary) !important;
}

.stMarkdown a {
    color: var(--accent-green-light) !important;
    text-decoration: none !important;
}

/* ── Metric Styling ── */
[data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    backdrop-filter: blur(10px) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent-green-light) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: rgba(16, 185, 129, 0.15) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 8px !important;
    color: var(--accent-green-light) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-glass) !important;
}

/* ── Tooltip ── */
[data-baseweb="tooltip"] > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(10px) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
}

/* ── Custom Classes ── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #10b981, #14b8a6, #3b82f6, #10b981);
    background-size: 300% 300%;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: gradientShift 6s ease infinite;
    line-height: 1.2 !important;
    margin-bottom: 0 !important;
    text-align: center;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-subtitle {
    color: #94a3b8 !important;
    font-size: 1.15rem !important;
    font-weight: 400 !important;
    text-align: center !important;
    line-height: 1.7 !important;
    max-width: 600px;
    margin: 12px auto 0 auto;
}

.glass-card {
    background: var(--gradient-card);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 28px;
    backdrop-filter: blur(20px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), transparent);
}

.glass-card:hover {
    border-color: rgba(16, 185, 129, 0.2);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}

.stat-number {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: var(--accent-green-light) !important;
    line-height: 1 !important;
}

.stat-label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-top: 8px !important;
}

.section-header {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 8px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-desc {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    margin-bottom: 24px !important;
}

.result-card {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.12), rgba(20, 184, 166, 0.06));
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: var(--radius-lg);
    padding: 40px;
    text-align: center;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(16, 185, 129, 0.1);
}

.result-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 4rem !important;
    font-weight: 800 !important;
    background: var(--gradient-primary);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.result-unit {
    color: var(--text-muted) !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
}

.tree-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 999px;
    padding: 10px 24px;
    color: var(--accent-green-light) !important;
    font-weight: 600;
    font-size: 1rem;
    margin-top: 16px;
}

.breakdown-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 0;
    border-bottom: 1px solid var(--border-glass);
}

.breakdown-label {
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

.breakdown-value {
    color: var(--accent-green-light) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

.info-banner {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(16, 185, 129, 0.08));
    border: 1px solid rgba(59, 130, 246, 0.15);
    border-radius: var(--radius);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

.info-banner-text {
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

.footer-bar {
    background: rgba(17, 24, 39, 0.8);
    border-top: 1px solid var(--border-glass);
    padding: 16px 32px;
    text-align: center;
    backdrop-filter: blur(20px);
    margin-top: 60px;
}

.footer-text {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.5px !important;
}

/* Hide image fullscreen button */
button[title="View fullscreen"] { display: none !important; }

/* Animated separator line */
.gradient-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-green), var(--accent-teal), var(--accent-blue), transparent);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    border-radius: 2px;
    margin: 24px 0;
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

/* Animated floating dots decoration */
.floating-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    opacity: 0.4;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

/* Columns gap fix */
[data-testid="column"] {
    padding: 0 8px !important;
}

/* Number input arrows */
[data-testid="stNumberInput"] button {
    background: var(--bg-glass) !important;
    border-color: var(--border-glass) !important;
    color: var(--text-secondary) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    scaler_path = os.path.join(BASE_DIR, "models", "scale.sav")
    model_path = os.path.join(BASE_DIR, "models", "model.sav")

    ss = pickle.load(open(scaler_path, "rb"))
    model = pickle.load(open(model_path, "rb"))

    return ss, model

ss, model = load_models()

# ── Did-You-Know Facts ───────────────────────────────────────────────────────
facts = [
    "Each year, human activities release over **40 billion tonnes** of CO₂ into the atmosphere.",
    "The production of **1 kg of beef** generates approximately **26 kgCO₂** emissions.",
    "Transportation accounts for nearly **25%** of global CO₂ emissions.",
    "Deforestation contributes to about **10%** of global carbon emissions.",
    "Driving an EV can cut your carbon footprint by around **50%** compared to petrol cars.",
    "The fashion industry emits around **3.3 billion tonnes** of CO₂ annually.",
    "Buildings are responsible for approximately **36%** of total energy use globally.",
    "The ocean absorbs about **30%** of atmospheric CO₂, causing acidification.",
    "Approximately **1.3 billion tonnes** of food are wasted globally each year.",
    "The average American generates over **16 metric tonnes** of CO₂ annually.",
]

# ═══════════════════════════════════════════════════════════════════
#  HERO SECTION
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align: center; padding: 40px 0 10px 0;">
    <div style="font-size: 4rem; margin-bottom: 8px;">🌍</div>
    <h1 class="hero-title">Carbon Footprint<br>Calculator</h1>
    <p class="hero-subtitle">
        Measure your environmental impact and discover actionable ways to reduce
        your carbon footprint. Every small change matters.
    </p>
</div>
<div class="gradient-line"></div>
""", unsafe_allow_html=True)

# ── Quick Stats Row ──────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <div class="stat-number">40B+</div>
        <div class="stat-label">Tonnes CO₂ / Year</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <div class="stat-number">1.2°C</div>
        <div class="stat-label">Global Temp Rise</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <div class="stat-number">25%</div>
        <div class="stat-label">From Transport</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Did You Know Banner ─────────────────────────────────────────
import random
fact = random.choice(facts)
st.markdown(f"""
<div class="info-banner">
    <span style="font-size: 1.4rem;">💡</span>
    <span class="info-banner-text"><strong>Did you know?</strong> {fact}</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  INPUT FORM
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">📋 Your Lifestyle Data</div>
<div class="section-desc">Fill in the details below across each category to calculate your monthly carbon emissions.</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 Personal", "🚗 Travel", "🗑️ Waste", "⚡ Energy", "💳 Consumption"])

# ── Personal Tab ─────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        height = st.number_input("📏 Height (cm)", 0, 251, value=None, placeholder="e.g. 170", help="Your height in centimeters")
    with p2:
        weight = st.number_input("⚖️ Weight (kg)", 0, 250, value=None, placeholder="e.g. 70", help="Your weight in kilograms")

    if (weight is None) or (weight == 0): weight = 1
    if (height is None) or (height == 0): height = 1
    bmi = weight / (height/100)**2
    body_type = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"

    sex = st.selectbox('🚻 Gender', ["female", "male"])
    diet = st.selectbox('🥗 Diet Type', ['omnivore', 'pescatarian', 'vegetarian', 'vegan'],
                        help="Omnivore: Eats plants & animals · Pescatarian: Plants & seafood · Vegetarian: No meat · Vegan: No animal products")
    social = st.selectbox('🎉 Social Activity', ['never', 'often', 'sometimes'], help="How frequently do you go out socially?")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Travel Tab ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    transport = st.selectbox('🚌 Primary Transportation', ['public', 'private', 'walk/bicycle'],
                             help="Which mode of transport do you use most?")
    if transport == "private":
        vehicle_type = st.selectbox('⛽ Vehicle Fuel Type', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric'])
    else:
        vehicle_type = "None"

    if transport == "walk/bicycle":
        vehicle_km = 0
    else:
        vehicle_km = st.slider('📍 Monthly Vehicle Distance (km)', 0, 5000, 0)

    air_travel = st.selectbox('✈️ Monthly Air Travel Frequency',
                              ['never', 'rarely', 'frequently', 'very frequently'],
                              help="Never · Rarely: 1–4 hrs · Frequently: 5–10 hrs · Very Frequently: 10+ hrs")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Waste Tab ────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    waste_bag = st.selectbox('🛍️ Waste Bag Size', ['small', 'medium', 'large', 'extra large'])
    waste_count = st.slider('🗑️ Weekly Waste Bags', 0, 10, 0)
    recycle = st.multiselect('♻️ Materials You Recycle', ['Plastic', 'Paper', 'Metal', 'Glass'])
    st.markdown('</div>', unsafe_allow_html=True)

# ── Energy Tab ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    heating_energy = st.selectbox('🔥 Heating Energy Source', ['natural gas', 'electricity', 'wood', 'coal'])
    for_cooking = st.multiselect('🍳 Cooking Appliances', ['microwave', 'oven', 'grill', 'airfryer', 'stove'])
    energy_efficiency = st.selectbox('💡 Consider Energy Efficiency?', ['No', 'Yes', 'Sometimes'],
                                     help="Do you consider energy efficiency when buying electronics?")
    daily_tv_pc = st.slider('🖥️ Daily Screen Time (hours)', 0, 24, 0)
    internet_daily = st.slider('🌐 Daily Internet Usage (hours)', 0, 24, 0)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Consumption Tab ──────────────────────────────────────────────
with tab5:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    shower = st.selectbox('🚿 Shower Frequency', ['daily', 'twice a day', 'more frequently', 'less frequently'])
    grocery_bill = st.slider('🛒 Monthly Grocery Bill ($)', 0, 500, 0)
    clothes_monthly = st.slider('👕 New Clothes Bought Monthly', 0, 30, 0)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

# Assemble dataframe
data_dict = {
    'Body Type': body_type, "Sex": sex, 'Diet': diet,
    "How Often Shower": shower, "Heating Energy Source": heating_energy,
    "Transport": transport, "Social Activity": social,
    'Monthly Grocery Bill': grocery_bill,
    "Frequency of Traveling by Air": air_travel,
    "Vehicle Monthly Distance Km": vehicle_km,
    "Waste Bag Size": waste_bag, "Waste Bag Weekly Count": waste_count,
    "How Long TV PC Daily Hour": daily_tv_pc,
    "Vehicle Type": vehicle_type,
    "How Many New Clothes Monthly": clothes_monthly,
    "How Long Internet Daily Hour": internet_daily,
    "Energy efficiency": energy_efficiency
}
data_dict.update({f"Cooking_with_{x}": y for x, y in dict(zip(for_cooking, np.ones(len(for_cooking)))).items()})
data_dict.update({f"Do You Recyle_{x}": y for x, y in dict(zip(recycle, np.ones(len(recycle)))).items()})

df = pd.DataFrame(data_dict, index=[0])
data = input_preprocessing(df)

sample_df = pd.DataFrame(data=sample, index=[0])
sample_df[sample_df.columns] = 0
sample_df[data.columns] = data

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    calculate = st.button("🌿  Calculate My Carbon Footprint", type="primary", use_container_width=True)

if calculate:
    prediction = round(np.exp(model.predict(ss.transform(sample_df))[0]))
    tree_count = round(prediction / 411.4)
    breakdown = hesapla(model, ss, sample_df)
    total_bd = sum(breakdown.values())

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

    # ── Result Card ──
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 3rem; margin-bottom: 12px;">🌍</div>
        <div style="color: var(--text-muted); font-size: 0.95rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">
            Your Monthly Carbon Emission
        </div>
        <div class="result-value">{prediction:,}</div>
        <div class="result-unit">kgCO₂e per month</div>
        <div class="tree-badge">
            🌳 You owe nature <strong>&nbsp;{tree_count}&nbsp;</strong> tree{'s' if tree_count != 1 else ''} monthly to offset
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Breakdown ──
    st.markdown('<div class="section-header">📊 Emission Breakdown</div>', unsafe_allow_html=True)

    bd_cols = st.columns(4)
    colors = ["#10b981", "#14b8a6", "#3b82f6", "#8b5cf6"]
    for i, (label, value) in enumerate(breakdown.items()):
        pct = (value / total_bd * 100) if total_bd > 0 else 0
        with bd_cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{label.split(' ')[0]}</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: {colors[i]};">
                    {value:.0f}
                </div>
                <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 4px;">kgCO₂e</div>
                <div style="margin-top: 12px; background: rgba(255,255,255,0.05); border-radius: 999px; height: 6px; overflow: hidden;">
                    <div style="width: {pct:.0f}%; height: 100%; background: {colors[i]}; border-radius: 999px;"></div>
                </div>
                <div style="color: var(--text-muted); font-size: 0.75rem; margin-top: 6px;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Pie Chart ──
    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='none')
    wedge_colors = ['#10b981', '#14b8a6', '#3b82f6', '#8b5cf6']
    wedges, texts, autotexts = ax.pie(
        breakdown.values(), labels=breakdown.keys(),
        autopct='%1.1f%%', explode=[0.03]*4,
        colors=wedge_colors, shadow=False,
        textprops={'fontsize': 13, 'color': '#e2e8f0', 'fontweight': 'bold'},
        pctdistance=0.82, labeldistance=1.08
    )
    for at in autotexts:
        at.set_color('#94a3b8')
        at.set_fontsize(11)
    centre = plt.Circle((0,0), 0.55, fc='#0a0f1a')
    ax.add_patch(centre)
    ax.text(0, 0.06, f'{prediction:,}', ha='center', va='center', fontsize=28,
            fontweight='bold', color='#10b981', fontfamily='sans-serif')
    ax.text(0, -0.1, 'kgCO₂e', ha='center', va='center', fontsize=12,
            color='#64748b', fontfamily='sans-serif')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)

    _, chart_col, _ = st.columns([1, 2, 1])
    with chart_col:
        st.image(buf, use_container_width=True)

    # ── Tips ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💡 Reduction Tips</div>', unsafe_allow_html=True)

    tips_data = {
        "🚲 Switch to cycling or public transport": "Can cut travel emissions by up to 50%",
        "🥦 Adopt a plant-based diet": "Vegan diets produce up to 73% less CO₂",
        "♻️ Recycle plastic, paper, metal & glass": "Reduces waste-related emissions significantly",
        "💡 Use energy-efficient appliances": "Cuts household energy consumption by 25-30%",
        "🛍️ Buy fewer, higher-quality clothes": "Fast fashion is a major emissions contributor"
    }

    for tip, desc in tips_data.items():
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 16px; padding: 14px 20px;
                    background: var(--bg-glass); border: 1px solid var(--border-glass);
                    border-radius: 14px; margin-bottom: 10px; transition: all 0.3s ease;">
            <div style="flex: 1;">
                <div style="color: var(--text-primary); font-weight: 600; font-size: 0.95rem;">{tip}</div>
                <div style="color: var(--text-muted); font-size: 0.82rem; margin-top: 4px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-bar">
    <div class="footer-text">
        🌍 Carbon Footprint Calculator · Built with 💚 for the planet · © 2024
    </div>
</div>
""", unsafe_allow_html=True)

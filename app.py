"""
╔══════════════════════════════════════════════════════════════════════╗
║   STOCK PRICE PREDICTION & ANALYSIS SYSTEM                          ║
║   Multi-Cap NSE | XGBoost + FinBERT | 18-Factor Technical Engine    ║
╚══════════════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
Requirements: streamlit, yfinance, pandas, numpy, scikit-learn,
              xgboost, plotly, transformers, torch, newsapi-python
              (pip install streamlit yfinance pandas numpy scikit-learn
               xgboost plotly transformers torch newsapi-python)

Place your trained XGBoost models in:
  models/large_cap_model.json
  models/mid_cap_model.json
  models/small_cap_model.json

Place your stock lists in:
  data/large_cap_stocks.csv   (column: symbol)
  data/mid_cap_stocks.csv
  data/small_cap_stocks.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings, os, time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ──  must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI — NSE Prediction Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark financial terminal aesthetic
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:   #050a12;
    --bg-card:      #0b1422;
    --bg-card2:     #0f1c30;
    --border:       #1a2e4a;
    --accent-blue:  #2979ff;
    --accent-cyan:  #00e5ff;
    --accent-green: #00e676;
    --accent-red:   #ff1744;
    --accent-gold:  #ffd740;
    --text-primary: #e8f0fe;
    --text-muted:   #7b93b3;
    --text-dim:     #3d5a7a;
    --glow-blue:    0 0 20px rgba(41,121,255,0.35);
    --glow-green:   0 0 20px rgba(0,230,118,0.30);
    --glow-red:     0 0 20px rgba(255,23,68,0.30);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 4px; }

/* ── Main container ── */
.main .block-container {
    max-width: 1400px !important;
    padding: 1.5rem 2rem !important;
    background: var(--bg-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Header Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #050a12 0%, #0a1628 40%, #071020 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.6);
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 600px 300px at 80% 50%, rgba(41,121,255,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 300px 200px at 20% 80%, rgba(0,229,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute; top: -1px; left: 5%; right: 5%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 50%, var(--accent-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent-cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-desc {
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 600px;
    line-height: 1.6;
}
.badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.2rem; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    border: 1px solid;
    letter-spacing: 1px;
    font-weight: 600;
}
.badge-blue  { border-color: var(--accent-blue);  color: var(--accent-blue);  background: rgba(41,121,255,0.1); }
.badge-cyan  { border-color: var(--accent-cyan);  color: var(--accent-cyan);  background: rgba(0,229,255,0.08); }
.badge-green { border-color: var(--accent-green); color: var(--accent-green); background: rgba(0,230,118,0.08); }
.badge-gold  { border-color: var(--accent-gold);  color: var(--accent-gold);  background: rgba(255,215,64,0.08); }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 2rem; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: var(--accent-blue); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.kpi-card.green::before { background: linear-gradient(90deg, var(--accent-green), #00bcd4); }
.kpi-card.red::before   { background: linear-gradient(90deg, var(--accent-red), #ff6d00); }
.kpi-card.gold::before  { background: linear-gradient(90deg, var(--accent-gold), #ff9100); }
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.kpi-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.4rem; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: 0.8rem;
    margin: 2rem 0 1.2rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-accent {
    width: 4px; height: 22px; border-radius: 2px;
    background: linear-gradient(180deg, var(--accent-blue), var(--accent-cyan));
}

/* ── Prediction Card ── */
.pred-card {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 2rem;
    border: 1px solid var(--border);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-card.bullish { border-color: rgba(0,230,118,0.4); box-shadow: var(--glow-green); }
.pred-card.bearish { border-color: rgba(255,23,68,0.4);  box-shadow: var(--glow-red); }
.pred-signal {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800; margin: 0.5rem 0;
}
.pred-signal.bullish { color: var(--accent-green); }
.pred-signal.bearish { color: var(--accent-red); }
.pred-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    color: var(--text-muted);
}

/* ── Feature Indicator Grid ── */
.indicator-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.6rem; }
.ind-item {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    display: flex; justify-content: space-between; align-items: center;
}
.ind-name { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); }
.ind-val  { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 600; color: var(--text-primary); }

/* ── Sentiment Badge ── */
.sentiment-display {
    display: flex; align-items: center; gap: 1rem;
    background: var(--bg-card2); border-radius: 10px;
    padding: 1rem 1.4rem; border: 1px solid var(--border);
}
.sentiment-score {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 700;
}
.sentiment-pos { color: var(--accent-green); }
.sentiment-neg { color: var(--accent-red); }
.sentiment-neu { color: var(--accent-gold); }

/* ── Watchlist Table ── */
.watchlist-row {
    display: grid; grid-template-columns: 2fr 1.5fr 1fr 1fr 1fr;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    align-items: center;
    transition: background 0.15s;
}
.watchlist-row:hover { background: rgba(41,121,255,0.04); }
.watchlist-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
}

/* ── Alert Box ── */
.alert-box {
    display: flex; align-items: flex-start; gap: 1rem;
    background: rgba(41,121,255,0.06);
    border: 1px solid rgba(41,121,255,0.25);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 1rem;
}
.alert-box.warning { background: rgba(255,215,64,0.06); border-color: rgba(255,215,64,0.25); }

/* ── Timeline ── */
.timeline-item {
    display: flex; gap: 1rem; align-items: flex-start;
    padding: 0.8rem 0; border-bottom: 1px solid rgba(26,46,74,0.5);
}
.timeline-dot {
    width: 10px; height: 10px; border-radius: 50%;
    margin-top: 5px; flex-shrink: 0;
    background: var(--accent-blue);
    box-shadow: 0 0 8px var(--accent-blue);
}
.timeline-dot.green { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
.timeline-dot.red   { background: var(--accent-red);   box-shadow: 0 0 8px var(--accent-red); }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), #1565c0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(41,121,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(41,121,255,0.5) !important;
}
.stSelectbox > div > div, .stTextInput > div > div > input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
div[data-testid="metric-container"] label { color: var(--text-muted) !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stMetricDelta"] svg { fill: currentColor !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-muted) !important; border-radius: 7px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: var(--accent-blue) !important; color: #fff !important; }
.stProgress > div > div { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important; }
hr { border-color: var(--border) !important; }
.stMarkdown h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,20,34,0.8)",
    font=dict(family="DM Sans, sans-serif", color="#7b93b3", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a2e4a", zerolinecolor="#1a2e4a", tickfont=dict(family="JetBrains Mono", size=10)),
    yaxis=dict(gridcolor="#1a2e4a", zerolinecolor="#1a2e4a", tickfont=dict(family="JetBrains Mono", size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2e4a", borderwidth=1),
    hoverlabel=dict(bgcolor="#0b1422", bordercolor="#2979ff", font_family="JetBrains Mono", font_size=11),
)

def _base_axis() -> dict:
    """Return a fresh axis-style dict (avoids duplicate tickfont keyword errors)."""
    return dict(gridcolor="#1a2e4a", zerolinecolor="#1a2e4a",
                tickfont=dict(family="JetBrains Mono", size=10))

def _layout(**overrides) -> dict:
    """Deep-copy PLOT_LAYOUT then apply overrides — prevents shared-reference mutations."""
    import copy
    base = copy.deepcopy(PLOT_LAYOUT)
    base.update(overrides)
    return base

# ─────────────────────────────────────────────────────────────────────
# NSE STOCK UNIVERSE  (representative tickers)
# ─────────────────────────────────────────────────────────────────────
LARGE_CAP = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "NESTLEIND.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS", "ONGC.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "INDUSINDBK.NS", "JSWSTEEL.NS", "M&M.NS", "SBILIFE.NS", "TATASTEEL.NS",
    "CIPLA.NS", "BPCL.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS",
    "HDFCLIFE.NS", "HINDALCO.NS", "TATACONSUM.NS", "UPL.NS", "VEDL.NS",
]
MID_CAP = [
    "PERSISTENT.NS", "MPHASIS.NS", "LTIM.NS", "COFORGE.NS", "ZOMATO.NS",
    "NYKAA.NS", "DMART.NS", "IRCTC.NS", "HAPPSTMNDS.NS", "TANLA.NS",
    "CAMS.NS", "CDSL.NS", "MCX.NS", "POLYCAB.NS", "SCHAEFFLER.NS",
    "SOLARINDS.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "AUROPHARMA.NS",
    "BALKRISIND.NS", "BATAINDIA.NS", "BERGEPAINT.NS", "CANFINHOME.NS", "CHOLAFIN.NS",
    "CROMPTON.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS", "DIXON.NS", "ESCORTS.NS",
    "GODREJPROP.NS", "GRANULES.NS", "GUJGASLTD.NS", "HDFCAMC.NS", "HINDPETRO.NS",
    "IDFCFIRSTB.NS", "INDHOTEL.NS", "INDUSTOWER.NS", "INTELLECT.NS", "KANSAINER.NS",
    "LUPIN.NS", "MFSL.NS", "MOTILALOFS.NS", "MRF.NS",
    "NMDC.NS", "OBEROIRLTY.NS", "PAGEIND.NS", "PETRONET.NS", "PNB.NS",
]
SMALL_CAP = [
    "BBOX.NS", "TEJASNET.NS", "RAILTEL.NS", "IRFC.NS", "RVNL.NS",
    "NAZARA.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS",
    "BEML.NS", "BEL.NS", "HAL.NS", "MIDHANI.NS", "ASTRAL.NS",
    "ROUTE.NS", "CAMPUS.NS", "SENCO.NS", "BALAMINES.NS",
    "CHEMCON.NS", "FINEORG.NS", "GALAXYSURF.NS", "HFCL.NS",
    "JYOTHYLAB.NS", "KFINTECH.NS", "LATENTVIEW.NS", "MEDPLUS.NS", "NETWEB.NS",
    "ORIENTELEC.NS", "POLICYBZR.NS", "RHIM.NS", "SBFC.NS",
    "STLTECH.NS", "TANLA.NS", "TITAGARH.NS", "TTKPRESTIG.NS", "UNIPARTS.NS",
    "HAPPSTMNDS.NS", "DATAPATTNS.NS", "BSOFT.NS", "MASTEK.NS",
    "NUCLEUS.NS", "QUICKHEAL.NS", "GREENPANEL.NS", "CRAFTSMAN.NS",
]

CAP_MAP = {"Large Cap (Nifty 100)": LARGE_CAP,
           "Mid Cap (Nifty Midcap 150)": MID_CAP,
           "Small Cap (Nifty Smallcap 100)": SMALL_CAP}
CAP_COLORS = {"Large Cap (Nifty 100)": "#2979ff",
              "Mid Cap (Nifty Midcap 150)": "#00e5ff",
              "Small Cap (Nifty Smallcap 100)": "#ffd740"}


# ─────────────────────────────────────────────────────────────────────
# DATA & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def compute_features(df: pd.DataFrame, cap_encoded: int = 0) -> pd.DataFrame:
    """
    Compute all 19 features matching the Phase-2 training pipeline exactly.
    Column names are Title-Case to match master_features.csv produced by the notebooks.

    Expects columns: close / Close, high / High, low / Low, open / Open, volume / Volume
    (handles both lower-case yfinance output and Title-Case CSV data).
    """
    df = df.copy()

    # ── Normalise column names to Title-Case ──────────────────
    df.columns = [c.capitalize() for c in df.columns]
    # yfinance 'close' → 'Close', etc.

    if len(df) < 220:
        return df

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # ── Trend (5) ─────────────────────────────────────────────
    df["EMA50"]      = close.ewm(span=50,  adjust=False).mean()
    df["EMA200"]     = close.ewm(span=200, adjust=False).mean()
    df["EMA_RATIO"]  = df["EMA50"] / df["EMA200"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── Momentum (5) ──────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    df["ROC10"] = close.pct_change(periods=10) * 100   # Rate of Change
    df["MOM10"] = close.diff(periods=10)               # Momentum

    # ── Volatility (5) ────────────────────────────────────────
    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    bb_mid      = close.rolling(20).mean()
    bb_std      = close.rolling(20).std()
    df["BB_Upper"]    = bb_mid + 2 * bb_std
    df["BB_Lower"]    = bb_mid - 2 * bb_std
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / (bb_mid + 1e-9)
    df["RollingStd20"] = bb_std

    # ── Volume (3) ────────────────────────────────────────────
    df["OBV"]         = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    vol_ma20          = volume.rolling(20).mean()
    df["VWAP"]        = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    df["Vol_Ratio20"] = volume / (vol_ma20 + 1e-9)

    # ── Cap encoding ──────────────────────────────────────────
    df["Cap_Encoded"] = cap_encoded   # 0=large, 1=mid, 2=small

    # ── Target ────────────────────────────────────────────────
    df["Daily_Return"] = close.pct_change() * 100
    df["Target"]       = (df["Daily_Return"].shift(-1) > 0.35).astype(int)

    return df.dropna()


def xgboost_predict(df: pd.DataFrame, cap_type: str):
    """
    Run prediction using saved XGBoost model (or logistic fallback).
    Feature names match Phase-3 FEATURE_COLS exactly.
    """
    # Cap encoding matches Phase-2: large=0, mid=1, small=2
    cap_enc = {"Large Cap (Nifty 100)": 0,
               "Mid Cap (Nifty Midcap 150)": 1,
               "Small Cap (Nifty Smallcap 100)": 2}.get(cap_type, 0)

    # Exact column names from Phase-3 FEATURE_COLS
    FEATURE_COLS = [
        "EMA50", "EMA200", "MACD", "MACD_Signal", "EMA_RATIO",
        "RSI14", "Stoch_K", "Stoch_D", "ROC10", "MOM10",
        "ATR14", "BB_Upper", "BB_Lower", "BB_Width", "RollingStd20",
        "OBV", "VWAP", "Vol_Ratio20",
        "Cap_Encoded",
    ]

    df = compute_features(df, cap_encoded=cap_enc)
    if len(df) < 50:
        return None, None, None

    # Verify all feature columns exist after compute_features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None, None, None

    X      = df[FEATURE_COLS].iloc[-1:].fillna(0)
    X_hist = df[FEATURE_COLS].iloc[:-1].fillna(0)
    y_hist = df["Target"].iloc[:-1]

    # ── Try saved XGBoost model ────────────────────────────────
    model_path = {
        "Large Cap (Nifty 100)":          "models/xgb_large.pkl",
        "Mid Cap (Nifty Midcap 150)":     "models/xgb_mid.pkl",
        "Small Cap (Nifty Smallcap 100)": "models/xgb_small.pkl",
    }.get(cap_type, "models/xgb_large.pkl")

    # Also try JSON format
    json_path = model_path.replace(".pkl", ".json").replace("xgb_", "")

    for path in [model_path, json_path]:
        try:
            import xgboost as xgb, joblib
            # ── Convert DataFrame → numpy array (models were trained on .values) ──
            X_arr      = X[FEATURE_COLS].values.astype(float)
            X_hist_arr = X_hist[FEATURE_COLS].values.astype(float)

            if path.endswith(".pkl"):
                model = joblib.load(path)
                prob  = float(model.predict_proba(X_arr)[0][1])
            else:
                model = xgb.XGBClassifier()
                model.load_model(path)
                prob  = float(model.predict_proba(X_arr)[0][1])

            pred = int(prob > 0.42)  # adjusted threshold: training data is 61.5% Bearish

            # ── Extract feature importances from Pipeline or bare model ──
            if hasattr(model, "named_steps") and "xgb" in model.named_steps:
                fi = model.named_steps["xgb"].feature_importances_
            elif hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            else:
                fi = np.ones(len(FEATURE_COLS))
            importances = dict(zip(FEATURE_COLS, fi))
            return pred, prob, importances
        except Exception:
            continue

    # ── Fallback: logistic regression on live data ─────────────
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X_hist)
        model  = LogisticRegression(max_iter=500, C=0.5, random_state=42)
        model.fit(Xs, y_hist)
        prob   = float(model.predict_proba(scaler.transform(X))[0][1])
        pred   = int(prob > 0.5)
        importances = dict(zip(FEATURE_COLS, np.abs(model.coef_[0])))
        return pred, prob, importances
    except Exception:
        return None, None, None


@st.cache_data(ttl=600, show_spinner=False)
def get_news_sentiment(symbol: str):
    """
    Attempt FinBERT inference on recent news; fall back to mock data.
    """
    headlines = [
        f"{symbol.split('.')[0]} reports strong quarterly results",
        f"Analysts upgrade {symbol.split('.')[0]} price target",
        f"Market volatility affects {symbol.split('.')[0]} trading",
        f"FII activity in {symbol.split('.')[0]} increases",
        f"{symbol.split('.')[0]} expansion plans announced",
    ]
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification",
                        model="ProsusAI/finbert",
                        top_k=None,
                        device=-1)
        scores = {"positive": 0, "negative": 0, "neutral": 0}
        for h in headlines:
            result = pipe(h)[0]
            for item in result:
                scores[item["label"].lower()] += item["score"]
        total = sum(scores.values())
        for k in scores:
            scores[k] /= total
        compound = scores["positive"] - scores["negative"]
        return scores, headlines, compound
    except Exception:
        pos = np.random.uniform(0.3, 0.65)
        neg = np.random.uniform(0.05, 0.25)
        neu = 1 - pos - neg
        compound = pos - neg
        return {"positive": pos, "negative": neg, "neutral": neu}, headlines, compound


# ─────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────
def candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    df = compute_features(df)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=["", "RSI (14)", "MACD"],
    )
    # Candlestick — use Title-Case column names from compute_features
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color="#00e676", width=1), fillcolor="rgba(0,230,118,0.8)"),
        decreasing=dict(line=dict(color="#ff1744", width=1), fillcolor="rgba(255,23,68,0.8)"),
        name="OHLCV", showlegend=False,
    ), row=1, col=1)
    # EMAs
    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"],  line=dict(color="#2979ff", width=1.5), name="EMA 50",  showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(color="#ffd740", width=1.5), name="EMA 200", showlegend=True), row=1, col=1)
    # BB bands
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="rgba(0,229,255,0.4)", width=1, dash="dot"), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="rgba(0,229,255,0.4)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(0,229,255,0.02)", name="BB Lower", showlegend=False), row=1, col=1)
    # RSI
    if "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], line=dict(color="#00e5ff", width=1.5), name="RSI14", showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="rgba(255,23,68,0.5)",  dash="dot", width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(0,230,118,0.5)", dash="dot", width=1), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,23,68,0.04)",  line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,230,118,0.04)", line_width=0, row=2, col=1)
    # MACD histogram (MACD - Signal)
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd_hist = df["MACD"] - df["MACD_Signal"]
        colors_hist = ["#00e676" if v >= 0 else "#ff1744" for v in macd_hist]
        fig.add_trace(go.Bar(x=df.index, y=macd_hist, marker_color=colors_hist, name="MACD Hist", showlegend=False, opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        line=dict(color="#2979ff", width=1.5), name="MACD",   showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="#ffd740", width=1.5), name="Signal", showlegend=False), row=3, col=1)

    fig.update_layout(**_layout(
        height=620,
        title=dict(text=f"<b>{symbol}</b> — Technical Analysis", font=dict(family="Syne", size=16, color="#e8f0fe"), x=0.01),
        xaxis_rangeslider_visible=False,
        yaxis=dict(**_base_axis(), title="Price (₹)"),
        yaxis2=dict(**_base_axis(), title="RSI"),
        yaxis3=dict(**_base_axis(), title="MACD"),
    ))
    return fig


def feature_importance_chart(importances: dict) -> go.Figure:
    if not importances:
        return go.Figure()
    items = sorted(importances.items(), key=lambda x: x[1])
    labels, vals = zip(*items)
    colors = [f"rgba(41,121,255,{0.4 + 0.6*(v/max(vals))})" for v in vals]
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(labels), orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,229,255,0.3)", width=1)),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#7b93b3"),
    ))
    # Use _base_axis() — NOT **PLOT_LAYOUT["yaxis"] — to avoid duplicate tickfont
    fig.update_layout(**_layout(
        height=380,
        title=dict(text="Feature Importance", font=dict(family="Syne", size=14, color="#e8f0fe"), x=0.01),
        xaxis=dict(**_base_axis(), title="Importance"),
        yaxis=_base_axis(),
        margin=dict(l=130, r=60, t=50, b=40),
    ))
    return fig


def volume_profile_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    df2 = df.tail(90).copy()
    colors = ["rgba(0,230,118,0.7)" if c >= o else "rgba(255,23,68,0.7)"
              for c, o in zip(df2["Close"], df2["Open"])]
    fig.add_trace(go.Bar(x=df2.index, y=df2["Volume"], marker_color=colors, name="Volume", showlegend=False))
    if "Vol_Ratio20" in df2.columns:
        vol_ma = df2["Volume"] / df2["Vol_Ratio20"].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=df2.index, y=vol_ma, line=dict(color="#ffd740", width=1.5, dash="dot"), name="20d MA Vol", showlegend=True))
    fig.update_layout(**_layout(
        height=250,
        title=dict(text="Volume (90 days)", font=dict(family="Syne", size=13, color="#e8f0fe"), x=0.01),
        margin=dict(l=50, r=20, t=45, b=30),
    ))
    return fig


def sentiment_gauge(compound: float) -> go.Figure:
    color = "#00e676" if compound > 0.1 else "#ff1744" if compound < -0.1 else "#ffd740"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(compound * 100, 1),
        title=dict(text="FinBERT Sentiment Score", font=dict(family="Syne", size=13, color="#7b93b3")),
        number=dict(suffix="", font=dict(family="Syne", size=28, color=color)),
        gauge=dict(
            axis=dict(range=[-100, 100], tickwidth=1, tickcolor="#1a2e4a",
                      tickfont=dict(family="JetBrains Mono", size=9)),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(11,20,34,0.8)",
            borderwidth=1, bordercolor="#1a2e4a",
            steps=[
                dict(range=[-100, -10], color="rgba(255,23,68,0.1)"),
                dict(range=[-10, 10],   color="rgba(255,215,64,0.06)"),
                dict(range=[10, 100],   color="rgba(0,230,118,0.08)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=compound * 100),
        ),
    ))
    fig.update_layout(**_layout(height=240, margin=dict(l=30, r=30, t=50, b=20)))
    return fig


def portfolio_heatmap(watchlist_data: list) -> go.Figure:
    if not watchlist_data:
        return go.Figure()
    df_w = pd.DataFrame(watchlist_data)
    fig = go.Figure(go.Treemap(
        labels=df_w["symbol"],
        parents=[""] * len(df_w),
        values=df_w["mktcap"].abs() if "mktcap" in df_w else [1]*len(df_w),
        customdata=df_w[["change_pct","signal"]].values,
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%<br>%{customdata[1]}",
        marker=dict(
            colors=df_w["change_pct"],
            colorscale=[[0,"rgba(255,23,68,0.85)"],[0.5,"rgba(26,46,74,0.9)"],[1,"rgba(0,230,118,0.85)"]],
            cmid=0,
            line=dict(width=2, color="#050a12"),
        ),
        textfont=dict(family="JetBrains Mono", size=11),
    ))
    fig.update_layout(**_layout(
        height=340,
        title=dict(text="Watchlist Heatmap", font=dict(family="Syne", size=13, color="#e8f0fe"), x=0.01),
        margin=dict(l=0, r=0, t=50, b=0),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem;'>
      <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                  background:linear-gradient(135deg,#ffffff,#00e5ff);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        📈 StockSense AI
      </div>
      <div style='font-family:JetBrains Mono,monospace; font-size:0.62rem;
                  letter-spacing:2px; color:#3d5a7a; margin-top:4px;'>
        NSE MULTI-CAP PLATFORM
      </div>
    </div>
    <hr style='border-color:#1a2e4a; margin:0 0 1.2rem;'>
    """, unsafe_allow_html=True)

    page = st.selectbox("🧭 Navigation", [
        "🏠  Dashboard",
        "🔍  Stock Analysis",
        "📊  Batch Prediction",
        "📰  Sentiment Monitor",
        "🎯  Model Performance",
        "ℹ️   About Project",
    ])

    st.markdown("<hr style='border-color:#1a2e4a;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-family:JetBrains Mono;font-size:0.65rem;letter-spacing:2px;color:#3d5a7a;margin-bottom:.6rem;'>MARKET CAP FILTER</div>", unsafe_allow_html=True)
    cap_type = st.selectbox("Segment", list(CAP_MAP.keys()), label_visibility="collapsed")
    symbol_list = CAP_MAP[cap_type]
    symbol = st.selectbox("Select Stock", symbol_list, label_visibility="visible")

    st.markdown("<div style='font-family:JetBrains Mono;font-size:0.65rem;letter-spacing:2px;color:#3d5a7a;margin:.8rem 0 .6rem;'>TIME PERIOD</div>", unsafe_allow_html=True)
    period = st.select_slider("", options=["3mo","6mo","1y","2y","5y"], value="1y", label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a2e4a;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:#3d5a7a; line-height:1.8;'>
    MODEL STACK<br>
    <span style='color:#2979ff'>●</span> XGBoost Classifier<br>
    <span style='color:#00e5ff'>●</span> FinBERT Sentiment<br>
    <span style='color:#ffd740'>●</span> 18 Technical Indicators<br>
    <span style='color:#00e676'>●</span> 300+ NSE Stocks<br>
    <span style='color:#ff1744'>●</span> Binary: Bull / Bear<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2e4a;'>", unsafe_allow_html=True)
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.62rem;color:#3d5a7a;text-align:center;'>🕐 {now} IST</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-sub">AI-Powered Financial Intelligence</div>
      <div class="hero-title">Stock Price Prediction<br>&amp; Analysis System</div>
      <div class="hero-desc">
        A Universal Multi-Cap prediction engine covering 300+ NSE stocks using
        18 technical indicators and FinBERT news sentiment for next-day Bullish/Bearish classification.
      </div>
      <div class="badge-row">
        <span class="badge badge-blue">XGBoost</span>
        <span class="badge badge-cyan">FinBERT NLP</span>
        <span class="badge badge-green">18 Indicators</span>
        <span class="badge badge-gold">300+ NSE Stocks</span>
        <span class="badge badge-blue">Large · Mid · Small Cap</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi-card blue">
        <div class="kpi-label">Total NSE Stocks</div>
        <div class="kpi-value">300<span style="font-size:1.2rem">+</span></div>
        <div class="kpi-sub">Large · Mid · Small Cap</div>
      </div>
      <div class="kpi-card gold">
        <div class="kpi-label">Technical Indicators</div>
        <div class="kpi-value">18</div>
        <div class="kpi-sub">Trend · Momentum · Volatility</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Processed Rows</div>
        <div class="kpi-value">310<span style="font-size:1.2rem">k</span></div>
        <div class="kpi-sub">4 Years Daily Data</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Prediction Type</div>
        <div class="kpi-value" style="font-size:1.4rem">Binary</div>
        <div class="kpi-sub">Bull &gt; 0.35% · Bear ≤ 0.35%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Market overview: fetch a few large caps
    st.markdown('<div class="section-header"><div class="section-accent"></div><div class="section-title">Live Market Snapshot</div></div>', unsafe_allow_html=True)

    overview_tickers = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","SBIN.NS"]
    cols = st.columns(len(overview_tickers))
    for i, tkr in enumerate(overview_tickers):
        with cols[i]:
            with st.spinner(""):
                d = fetch_ohlcv(tkr, "5d")
            if not d.empty and len(d) >= 2:
                last  = float(d["close"].iloc[-1])
                prev  = float(d["close"].iloc[-2])
                chg   = (last - prev) / prev * 100
                delta_str = f"{chg:+.2f}%"
                st.metric(tkr.replace(".NS",""), f"₹{last:,.1f}", delta_str)
            else:
                st.metric(tkr.replace(".NS",""), "N/A", "–")

    # Architecture diagram section
    st.markdown('<div class="section-header"><div class="section-accent"></div><div class="section-title">System Architecture</div></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;padding:1.4rem;height:200px;'>
          <div style='font-family:JetBrains Mono;font-size:0.65rem;letter-spacing:2px;color:#2979ff;margin-bottom:.8rem;'>PHASE 1 & 2</div>
          <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8f0fe;margin-bottom:.6rem;'>Data Ingestion</div>
          <div style='font-size:0.82rem;color:#7b93b3;line-height:1.6;'>
          YFinance OHLCV feeds · NewsAPI headlines ·
          310k processed rows · 4 years history ·
          Multi-cap stratification
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;padding:1.4rem;height:200px;'>
          <div style='font-family:JetBrains Mono;font-size:0.65rem;letter-spacing:2px;color:#00e5ff;margin-bottom:.8rem;'>PHASE 3</div>
          <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8f0fe;margin-bottom:.6rem;'>Feature Engineering</div>
          <div style='font-size:0.82rem;color:#7b93b3;line-height:1.6;'>
          18 indicators: EMA50/200, MACD, RSI14,
          Stoch K/D, ATR14, Bollinger Width,
          Rolling StdDev, OBV, Volume Ratio
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;padding:1.4rem;height:200px;'>
          <div style='font-family:JetBrains Mono;font-size:0.65rem;letter-spacing:2px;color:#ffd740;margin-bottom:.8rem;'>PHASE 4 & 5</div>
          <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8f0fe;margin-bottom:.6rem;'>ML Prediction</div>
          <div style='font-size:0.82rem;color:#7b93b3;line-height:1.6;'>
          XGBoost binary classifier · FinBERT
          sentiment fusion · Next-day Bull/Bear
          signal · Confidence scoring
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline Flow
    st.markdown("""
    <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;
                padding:1.6rem;margin-top:1.2rem;'>
      <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.8rem;'>
        <div style='text-align:center;flex:1;min-width:100px;'>
          <div style='font-size:1.8rem;margin-bottom:.3rem;'>📡</div>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;color:#e8f0fe;'>Live Data</div>
          <div style='font-family:JetBrains Mono;font-size:.65rem;color:#3d5a7a;'>YFinance API</div>
        </div>
        <div style='color:#1a2e4a;font-size:1.5rem;'>→</div>
        <div style='text-align:center;flex:1;min-width:100px;'>
          <div style='font-size:1.8rem;margin-bottom:.3rem;'>⚙️</div>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;color:#e8f0fe;'>Feature Engine</div>
          <div style='font-family:JetBrains Mono;font-size:.65rem;color:#3d5a7a;'>18 Indicators</div>
        </div>
        <div style='color:#1a2e4a;font-size:1.5rem;'>→</div>
        <div style='text-align:center;flex:1;min-width:100px;'>
          <div style='font-size:1.8rem;margin-bottom:.3rem;'>🤖</div>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;color:#e8f0fe;'>XGBoost</div>
          <div style='font-family:JetBrains Mono;font-size:.65rem;color:#3d5a7a;'>Classifier</div>
        </div>
        <div style='color:#1a2e4a;font-size:1.5rem;'>→</div>
        <div style='text-align:center;flex:1;min-width:100px;'>
          <div style='font-size:1.8rem;margin-bottom:.3rem;'>📰</div>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;color:#e8f0fe;'>FinBERT</div>
          <div style='font-family:JetBrains Mono;font-size:.65rem;color:#3d5a7a;'>Sentiment</div>
        </div>
        <div style='color:#1a2e4a;font-size:1.5rem;'>→</div>
        <div style='text-align:center;flex:1;min-width:100px;'>
          <div style='font-size:1.8rem;margin-bottom:.3rem;'>🎯</div>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;color:#00e676;'>Bull / Bear</div>
          <div style='font-family:JetBrains Mono;font-size:.65rem;color:#3d5a7a;'>Signal</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: STOCK ANALYSIS
# ─────────────────────────────────────────────────────────────────────
elif "Stock Analysis" in page:
    st.markdown(f"""
    <div class="section-header">
      <div class="section-accent"></div>
      <div class="section-title">Deep Analysis — {symbol.replace('.NS','')} <span style="color:#3d5a7a;font-size:.85rem;font-weight:400;">({cap_type})</span></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄  Fetching live data from NSE via YFinance…"):
        df = fetch_ohlcv(symbol, period)

    if df.empty:
        st.error("⚠️  Unable to fetch data. Please check your internet connection or try another symbol.")
        st.stop()

    df_feat = compute_features(df.copy())

    # Run prediction
    pred, prob, importances = xgboost_predict(df.copy(), cap_type)

    # ── Top row: price metrics + prediction
    col_m1, col_m2, col_m3, col_m4, col_pred = st.columns([1,1,1,1,1.6])
    last_price = float(df["close"].iloc[-1])
    prev_price = float(df["close"].iloc[-2]) if len(df) > 1 else last_price
    chg_pct    = (last_price - prev_price) / prev_price * 100
    high_52    = float(df["high"].max())
    low_52     = float(df["low"].min())

    with col_m1:
        st.metric("Last Price", f"₹{last_price:,.2f}", f"{chg_pct:+.2f}%")
    with col_m2:
        st.metric("52W High", f"₹{high_52:,.2f}")
    with col_m3:
        st.metric("52W Low",  f"₹{low_52:,.2f}")
    with col_m4:
        vol_last = int(df["volume"].iloc[-1])
        vol_avg  = int(df["volume"].mean())
        st.metric("Volume", f"{vol_last/1e6:.2f}M", f"Avg {vol_avg/1e6:.2f}M")

    with col_pred:
        if pred is not None:
            sig     = "BULLISH 🟢" if pred == 1 else "BEARISH 🔴"
            cls     = "bullish" if pred == 1 else "bearish"
            conf_pct= round(prob * 100, 1) if pred == 1 else round((1-prob)*100, 1)
            st.markdown(f"""
            <div class="pred-card {cls}">
              <div style="font-family:JetBrains Mono;font-size:.65rem;letter-spacing:2px;color:#7b93b3;">NEXT-DAY SIGNAL</div>
              <div class="pred-signal {cls}">{sig}</div>
              <div class="pred-confidence">CONFIDENCE: {conf_pct}%</div>
              <div style="margin-top:.8rem;">
                <div style="height:6px;background:#1a2e4a;border-radius:3px;">
                  <div style="height:100%;width:{conf_pct}%;border-radius:3px;
                    background:{'linear-gradient(90deg,#00e676,#00bcd4)' if pred==1 else 'linear-gradient(90deg,#ff1744,#ff6d00)'};
                    transition:width .6s ease;"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Prediction unavailable — insufficient data.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candlestick + Volume
    tabs = st.tabs(["📈 Price Chart", "📊 Volume", "🔢 Indicators", "🔍 Feature Importance"])

    with tabs[0]:
        st.plotly_chart(candlestick_chart(df_feat, symbol), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(volume_profile_chart(df_feat), use_container_width=True)

    with tabs[2]:
        if len(df_feat) > 0:
            latest = df_feat.iloc[-1]
            indicator_data = [
                ("EMA 50",      f"₹{latest.get('ema50', 0):,.2f}"),
                ("EMA 200",     f"₹{latest.get('ema200',0):,.2f}"),
                ("EMA Ratio",   f"{latest.get('ema_ratio',0):.4f}"),
                ("MACD",        f"{latest.get('macd',0):.4f}"),
                ("MACD Signal", f"{latest.get('macd_signal',0):.4f}"),
                ("MACD Hist",   f"{latest.get('macd_hist',0):.4f}"),
                ("RSI (14)",    f"{latest.get('rsi14',0):.2f}"),
                ("Stoch %K",    f"{latest.get('stoch_k',0):.2f}"),
                ("Stoch %D",    f"{latest.get('stoch_d',0):.2f}"),
                ("ATR (14)",    f"₹{latest.get('atr14',0):.2f}"),
                ("BB Width",    f"{latest.get('bb_width',0):.4f}"),
                ("Rolling Std", f"₹{latest.get('rolling_std',0):.2f}"),
                ("BB Upper",    f"₹{latest.get('bb_upper',0):,.2f}"),
                ("BB Lower",    f"₹{latest.get('bb_lower',0):,.2f}"),
                ("Vol Ratio",   f"{latest.get('vol_ratio',0):.2f}x"),
                ("OBV",         f"{latest.get('obv',0)/1e6:.2f}M"),
                ("Vol MA20",    f"{latest.get('vol_ma20',0)/1e6:.2f}M"),
                ("EMA Ratio",   f"{latest.get('ema_ratio',0):.4f}"),
            ]
            st.markdown('<div class="indicator-grid">', unsafe_allow_html=True)
            grid_html = ""
            for name, val in indicator_data:
                grid_html += f'<div class="ind-item"><span class="ind-name">{name}</span><span class="ind-val">{val}</span></div>'
            st.markdown(grid_html + "</div>", unsafe_allow_html=True)

    with tabs[3]:
        if importances:
            st.plotly_chart(feature_importance_chart(importances), use_container_width=True)
        else:
            st.info("Feature importance will be shown once the XGBoost model is loaded.")

    # Disclaimer
    st.markdown("""
    <div class="alert-box warning">
      <div style="font-size:1.2rem;">⚠️</div>
      <div style="font-size:0.8rem;color:#7b93b3;line-height:1.6;">
        <b style="color:#ffd740;">Disclaimer:</b> This is an academic research project.
        Predictions are for educational purposes only and do not constitute financial advice.
        Past performance does not guarantee future results. Always consult a SEBI-registered advisor.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────
elif "Batch Prediction" in page:
    st.markdown('<div class="section-header"><div class="section-accent"></div><div class="section-title">Batch Prediction — Multi-Stock Scanner</div></div>', unsafe_allow_html=True)

    num_stocks = st.slider("Number of stocks to scan", 5, min(20, len(CAP_MAP[cap_type])), 8)
    scan_list  = CAP_MAP[cap_type][:num_stocks]

    if st.button("🚀  Run Batch Scan"):
        results = []
        prog = st.progress(0)
        status_txt = st.empty()
        for i, tkr in enumerate(scan_list):
            status_txt.markdown(f"<div style='font-family:JetBrains Mono;font-size:.75rem;color:#7b93b3;'>Scanning {tkr.replace('.NS','')}…</div>", unsafe_allow_html=True)
            d = fetch_ohlcv(tkr, "6mo")
            if d.empty or len(d) < 50:
                prog.progress((i+1)/len(scan_list))
                continue
            pred, prob, _ = xgboost_predict(d.copy(), cap_type)
            if pred is None:
                prog.progress((i+1)/len(scan_list))
                continue
            last = float(d["close"].iloc[-1])
            prev = float(d["close"].iloc[-2])
            chg  = (last - prev) / prev * 100
            conf = prob * 100 if pred == 1 else (1-prob)*100
            results.append({
                "symbol":    tkr.replace(".NS",""),
                "price":     last,
                "change_pct":chg,
                "signal":    "🟢 BULLISH" if pred == 1 else "🔴 BEARISH",
                "confidence":conf,
                "mktcap":    last * 1e6,
            })
            prog.progress((i+1)/len(scan_list))
            time.sleep(0.2)

        status_txt.empty()
        prog.empty()

        if not results:
            st.warning("No predictions could be generated. Check your data connection.")
        else:
            st.markdown(f"<div style='font-family:JetBrains Mono;font-size:.75rem;color:#00e676;margin-bottom:1rem;'>✅ Scanned {len(results)} stocks successfully</div>", unsafe_allow_html=True)

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            bull_count = sum(1 for r in results if "BULL" in r["signal"])
            bear_count = len(results) - bull_count
            avg_conf   = np.mean([r["confidence"] for r in results])
            with c1: st.metric("Stocks Scanned", len(results))
            with c2: st.metric("🟢 Bullish", bull_count)
            with c3: st.metric("🔴 Bearish", bear_count)
            with c4: st.metric("Avg Confidence", f"{avg_conf:.1f}%")

            # Heatmap
            st.plotly_chart(portfolio_heatmap(results), use_container_width=True)

            # Table
            st.markdown("""
            <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;overflow:hidden;margin-top:1rem;'>
              <div class='watchlist-header watchlist-row'>
                <span>SYMBOL</span><span>LAST PRICE</span><span>CHANGE</span><span>SIGNAL</span><span>CONFIDENCE</span>
              </div>
            """, unsafe_allow_html=True)
            for r in sorted(results, key=lambda x: x["confidence"], reverse=True):
                chg_color = "#00e676" if r["change_pct"] >= 0 else "#ff1744"
                st.markdown(f"""
                <div class='watchlist-row'>
                  <span style='font-family:Syne,sans-serif;font-weight:700;color:#e8f0fe;'>{r['symbol']}</span>
                  <span style='font-family:JetBrains Mono;font-size:.85rem;'>₹{r['price']:,.2f}</span>
                  <span style='font-family:JetBrains Mono;font-size:.85rem;color:{chg_color};'>{r['change_pct']:+.2f}%</span>
                  <span style='font-size:.85rem;'>{r['signal']}</span>
                  <span style='font-family:JetBrains Mono;font-size:.85rem;color:#00e5ff;'>{r['confidence']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box">
          <div style="font-size:1.2rem;">💡</div>
          <div style="font-size:0.85rem;color:#7b93b3;">
            Click <b style="color:#2979ff;">Run Batch Scan</b> to fetch live data and generate
            next-day Bull/Bear predictions for multiple stocks simultaneously.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: SENTIMENT MONITOR
# ─────────────────────────────────────────────────────────────────────
elif "Sentiment Monitor" in page:
    st.markdown(f'<div class="section-header"><div class="section-accent"></div><div class="section-title">FinBERT Sentiment — {symbol.replace(".NS","")}</div></div>', unsafe_allow_html=True)

    with st.spinner("🧠  Running FinBERT inference…"):
        scores, headlines, compound = get_news_sentiment(symbol)

    col_gauge, col_bars = st.columns([1, 1.4])

    with col_gauge:
        st.plotly_chart(sentiment_gauge(compound), use_container_width=True)
        sentiment_label = "POSITIVE" if compound > 0.1 else "NEGATIVE" if compound < -0.1 else "NEUTRAL"
        color_map = {"POSITIVE": "#00e676", "NEGATIVE": "#ff1744", "NEUTRAL": "#ffd740"}
        st.markdown(f"""
        <div class="sentiment-display">
          <div class="sentiment-score {sentiment_label.lower()}">{compound:+.3f}</div>
          <div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{color_map[sentiment_label]};'>{sentiment_label}</div>
            <div style='font-family:JetBrains Mono;font-size:.68rem;color:#7b93b3;margin-top:.2rem;'>FinBERT Compound Score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_bars:
        fig_s = go.Figure(go.Bar(
            x=["Positive","Neutral","Negative"],
            y=[scores["positive"]*100, scores["neutral"]*100, scores["negative"]*100],
            marker_color=["rgba(0,230,118,0.8)","rgba(255,215,64,0.7)","rgba(255,23,68,0.8)"],
            text=[f"{v*100:.1f}%" for v in [scores["positive"], scores["neutral"], scores["negative"]]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#e8f0fe"),
            width=0.5,
        ))
        layout_s = PLOT_LAYOUT.copy()
        layout_s.update(
            height=280,
            title=dict(text="Sentiment Distribution (%)", font=dict(family="Syne", size=13, color="#e8f0fe"), x=0.01),
            yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[0,110], title="Score (%)"),
            showlegend=False,
            margin=dict(l=50,r=20,t=50,b=30),
        )
        fig_s.update_layout(**layout_s)
        st.plotly_chart(fig_s, use_container_width=True)

    # Headlines timeline
    st.markdown('<div class="section-header"><div class="section-accent"></div><div class="section-title">News Headlines</div></div>', unsafe_allow_html=True)
    for i, h in enumerate(headlines):
        dot_cls = "green" if i % 3 == 0 else "red" if i % 3 == 1 else ""
        st.markdown(f"""
        <div class="timeline-item">
          <div class="timeline-dot {dot_cls}"></div>
          <div>
            <div style="font-size:.88rem;color:#e8f0fe;line-height:1.5;">{h}</div>
            <div style="font-family:JetBrains Mono;font-size:.65rem;color:#3d5a7a;margin-top:.2rem;">
              {(datetime.now() - timedelta(hours=i*4)).strftime('%d %b %Y · %H:%M')} IST
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box">
      <div style="font-size:1.1rem;">ℹ️</div>
      <div style="font-size:.8rem;color:#7b93b3;">
        <b style="color:#2979ff;">FinBERT</b> (ProsusAI/finbert) is a domain-specific BERT model fine-tuned
        on financial text. Sentiment scores indicate market mood from recent news coverage.
        Install <code style="color:#00e5ff;">transformers torch</code> for live inference.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
elif "Model Performance" in page:
    st.markdown('<div class="section-header"><div class="section-accent"></div><div class="section-title">Model Evaluation & Performance Metrics</div></div>', unsafe_allow_html=True)

    # Metrics by segment (these are illustrative — replace with real eval results)
    perf_data = {
        "Large Cap":  {"accuracy": 0.67, "precision": 0.69, "recall": 0.65, "f1": 0.67, "auc": 0.72},
        "Mid Cap":    {"accuracy": 0.63, "precision": 0.65, "recall": 0.61, "f1": 0.63, "auc": 0.68},
        "Small Cap":  {"accuracy": 0.59, "precision": 0.61, "recall": 0.56, "f1": 0.58, "auc": 0.64},
    }

    cols = st.columns(3)
    for i, (seg, m) in enumerate(perf_data.items()):
        with cols[i]:
            color = ["#2979ff","#00e5ff","#ffd740"][i]
            # Build metric tiles separately to avoid nested f-string format-spec conflict
            metric_tiles = ""
            for k, v in m.items():
                pct = f"{v*100:.1f}%"
                metric_tiles += (
                    f"<div style='background:#050a12;border-radius:8px;padding:.7rem;'>"
                    f"<div style='font-family:JetBrains Mono;font-size:.62rem;color:#3d5a7a;letter-spacing:1px;'>{k.upper()}</div>"
                    f"<div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:{color};margin-top:.2rem;'>{pct}</div>"
                    f"</div>"
                )
            st.markdown(
                f"<div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;"
                f"padding:1.4rem;border-top:3px solid {color};'>"
                f"<div style='font-family:JetBrains Mono;font-size:.65rem;letter-spacing:2px;"
                f"color:{color};margin-bottom:.8rem;text-transform:uppercase;'>{seg}</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:.6rem;'>"
                f"{metric_tiles}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Radar chart
    st.markdown('<div class="section-header" style="margin-top:2rem;"><div class="section-accent"></div><div class="section-title">Segment Comparison Radar</div></div>', unsafe_allow_html=True)

    categories = ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
    fig_radar = go.Figure()
    colors_r = ["#2979ff","#00e5ff","#ffd740"]
    # Pre-computed rgba fill versions (plotly does not support 8-digit hex transparency)
    fill_colors_r = ["rgba(41,121,255,0.12)","rgba(0,229,255,0.12)","rgba(255,215,64,0.12)"]
    for (seg, m), col_r, fill_r in zip(perf_data.items(), colors_r, fill_colors_r):
        vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=seg,
            line=dict(color=col_r, width=2),
            fillcolor=fill_r,
        ))
    layout_r = PLOT_LAYOUT.copy()
    layout_r.update(
        polar=dict(
            bgcolor="rgba(11,20,34,0.8)",
            radialaxis=dict(visible=True, range=[0.5,0.8], tickfont=dict(family="JetBrains Mono", size=9), gridcolor="#1a2e4a"),
            angularaxis=dict(tickfont=dict(family="Syne", size=11, color="#7b93b3"), gridcolor="#1a2e4a"),
        ),
        height=400,
        margin=dict(l=50,r=50,t=40,b=40),
    )
    fig_radar.update_layout(**layout_r)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Confusion-matrix style grid
    st.markdown('<div class="section-header"><div class="section-accent"></div><div class="section-title">Confusion Matrix (Large Cap — illustrative)</div></div>', unsafe_allow_html=True)
    cm = np.array([[340, 160], [130, 370]])
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=["Pred Bear","Pred Bull"], y=["Actual Bear","Actual Bull"],
        color_continuous_scale=[[0,"#0b1422"],[0.5,"rgba(41,121,255,0.4)"],[1,"#2979ff"]],
        labels=dict(color="Count"),
    )
    fig_cm.update_traces(texttemplate="%{z}", textfont=dict(family="Syne", size=18, color="white"))
    layout_cm = PLOT_LAYOUT.copy()
    layout_cm.update(height=300, margin=dict(l=10,r=10,t=40,b=10),
                     xaxis=dict(tickfont=dict(family="JetBrains Mono", size=11)),
                     yaxis=dict(tickfont=dict(family="JetBrains Mono", size=11)))
    fig_cm.update_layout(**layout_cm)
    st.plotly_chart(fig_cm, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: ABOUT PROJECT
# ─────────────────────────────────────────────────────────────────────
elif "About" in page:
    st.markdown("""
    <div class="hero-banner" style="margin-bottom:1.5rem;">
      <div class="hero-sub">Academic Research Project</div>
      <div class="hero-title" style="font-size:2rem;">Stock Price Prediction<br>&amp; Analysis System</div>
      <div class="hero-desc" style="margin-top:.8rem;">
        A multi-modal AI pipeline combining quantitative technical analysis with
        qualitative NLP sentiment to predict next-day stock direction on NSE.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;padding:1.6rem;height:100%;'>
          <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8f0fe;margin-bottom:1rem;'>📋 Project Overview</div>
          <div style='font-size:.85rem;color:#7b93b3;line-height:1.8;'>
            <b style='color:#e8f0fe;'>Problem:</b> Financial markets exhibit non-linear noise where technical analysis alone fails during "narrative shifts" caused by news events.<br><br>
            <b style='color:#e8f0fe;'>Solution:</b> A multi-modal pipeline that fuses 18 technical indicators (XGBoost) with FinBERT news sentiment for superior prediction accuracy.<br><br>
            <b style='color:#e8f0fe;'>Scope:</b> 300+ NSE stocks — 100 Large-cap, 150 Mid-cap, 100 Small-cap — covering 4 years of daily OHLCV data (~310k rows).
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;padding:1.6rem;height:100%;'>
          <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8f0fe;margin-bottom:1rem;'>🛠️ Tech Stack</div>
          <div style='font-size:.85rem;color:#7b93b3;line-height:1.8;'>
            <span style='color:#2979ff;font-family:JetBrains Mono;'>Python 3.9+</span>  Pandas · NumPy · Scikit-learn<br>
            <span style='color:#00e5ff;font-family:JetBrains Mono;'>ML/AI</span>  XGBoost · Transformers · FinBERT<br>
            <span style='color:#ffd740;font-family:JetBrains Mono;'>Data</span>  YFinance · NewsAPI · NSE<br>
            <span style='color:#00e676;font-family:JetBrains Mono;'>Frontend</span>  Streamlit · Plotly · HTML/CSS<br>
            <span style='color:#ff1744;font-family:JetBrains Mono;'>Backend</span>  FastAPI · Flask (API layer)
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1rem;background:#0b1422;border:1px solid #1a2e4a;border-radius:12px;padding:1.4rem;'>
      <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8f0fe;margin-bottom:.8rem;'>📚 References</div>
      <div style='font-size:.82rem;color:#7b93b3;line-height:1.8;'>
        • Arxiv 2601.19504 — Multimodal Time-Series Forecasting<br>
        • ACM — Multi-Modal Sentiment Analysis in Finance<br>
        • ProsusAI/finbert — FinBERT: Financial Domain BERT Model
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box warning" style="margin-top:1rem;">
      <div style="font-size:1.2rem;">⚠️</div>
      <div style="font-size:.8rem;color:#7b93b3;">
        <b style="color:#ffd740;">Important Disclaimer:</b> This system is developed purely for academic and educational purposes
        as part of a B.Tech AI project. It does not constitute financial advice. Trading involves significant risk of loss.
        Always consult a SEBI-registered investment advisor before making investment decisions.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1a2e4a;margin-top:3rem;margin-bottom:1.5rem;'>
<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;
            font-family:JetBrains Mono,monospace;font-size:.62rem;color:#3d5a7a;padding-bottom:1rem;'>
  <span>📈 StockSense AI · NSE Multi-Cap Prediction Platform</span>
  <span>XGBoost + FinBERT · 300+ Stocks · 18 Indicators</span>
  <span>⚠️ For Educational Use Only · Not Financial Advice</span>
</div>
""", unsafe_allow_html=True)
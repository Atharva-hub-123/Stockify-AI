# 📈 Stockify-AI
### *Predict Smarter. Invest Better.*

> A Multi-Cap NSE Stock Price Prediction & Analysis System powered by XGBoost and FinBERT sentiment analysis.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189AB4?style=flat-square)
![FinBERT](https://img.shields.io/badge/FinBERT-NLP-764ABC?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Disclaimer](#disclaimer)

---

## 🧠 Overview

**Stockify-AI** is an end-to-end AI-powered stock prediction and analysis platform for NSE (National Stock Exchange of India). It combines:

- **18 quantitative technical indicators** (EMA, MACD, RSI, Bollinger Bands, ATR, OBV, VWAP, and more)
- **XGBoost binary classifier** trained on 4 years of daily OHLCV data
- **FinBERT NLP sentiment analysis** on financial news headlines
- **Interactive Streamlit dashboard** with live charts, batch scanning, and model explainability

The system predicts whether a stock's next-day return will be **Bullish (>0.35%)** or **Bearish (≤0.35%)** for 130 NSE stocks across Large, Mid, and Small Cap segments.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Stock Analysis** | Live candlestick charts with EMA, Bollinger Bands, RSI, MACD overlays |
| 🤖 **AI Prediction** | Next-day Bull/Bear signal with confidence score using trained XGBoost |
| 📊 **Batch Scanner** | Scan up to 20 stocks simultaneously with heatmap visualization |
| 📰 **Sentiment Monitor** | FinBERT-based sentiment scoring on financial news |
| 🎯 **Model Performance** | Confusion matrix, ROC curves, SHAP feature importance |
| 🌐 **130+ NSE Stocks** | Large Cap (Nifty 100), Mid Cap, Small Cap coverage |

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  YFinance   │───▶│  Phase 1 & 2     │───▶│   Phase 3       │
│  OHLCV API  │    │  Data Ingestion  │    │   XGBoost       │
│             │    │  + 18 Indicators │    │   Classifier    │
└─────────────┘    └──────────────────┘    └────────┬────────┘
                                                    │
┌─────────────┐    ┌──────────────────┐             │
│  NewsAPI /  │───▶│  Phase 4         │             │
│  Headlines  │    │  FinBERT         │             ▼
│             │    │  Sentiment       │    ┌─────────────────┐
└─────────────┘    └──────────────────┘    │   Streamlit     │
                           │               │   Dashboard     │
                           └──────────────▶│   (app.py)      │
                                           └─────────────────┘
```

**Data Flow:**
```
Raw OHLCV → Cleaning → 18 Technical Indicators → XGBoost Pipeline
                                                        ↓
News Headlines → FinBERT Inference → Sentiment Score → Bull / Bear Signal
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Yahoo Finance API (`yfinance`) |
| **Coverage** | 130 NSE stocks — 50 Large Cap, 48 Mid Cap, 42 Small Cap |
| **Period** | January 2021 – December 2024 (4 years) |
| **Frequency** | Daily OHLCV |
| **Raw Size** | 123,281 rows × 8 columns |
| **Processed Size** | 126,787 rows × 28 columns |
| **Target** | Binary — Bullish (next-day return > 0.35%) / Bearish |
| **Class Distribution** | 61.5% Bearish · 38.5% Bullish |

**18 Technical Indicators:**

| Category | Indicators |
|---|---|
| Trend (5) | EMA50, EMA200, MACD, MACD Signal, EMA Ratio |
| Momentum (5) | RSI14, Stoch %K, Stoch %D, ROC10, MOM10 |
| Volatility (5) | ATR14, BB Upper, BB Lower, BB Width, Rolling Std20 |
| Volume (3) | OBV, VWAP, Volume Ratio 20 |

---

## 🛠️ Tech Stack

```
Language      Python 3.9+
ML            XGBoost, Scikit-learn, SHAP
NLP           HuggingFace Transformers, ProsusAI/FinBERT
Data          Pandas, NumPy, yfinance, ta (Technical Analysis)
Frontend      Streamlit, Plotly
Notebooks     Jupyter
```

---

## 📁 Project Structure

```
stockify-ai/
│
├── app.py                          # Main Streamlit dashboard
│
├── Phase1_Ingestion.ipynb          # Data ingestion from YFinance
├── Phase2_feature_engineering.ipynb # Cleaning + 18 indicator computation
├── Phase3_xgboost_model.ipynb      # XGBoost training + evaluation
├── Phase4_finbert_sentiment.ipynb  # FinBERT sentiment pipeline
│
├── models/
│   ├── xgb_large.pkl               # Trained model — Large Cap
│   ├── xgb_mid.pkl                 # Trained model — Mid Cap
│   ├── xgb_small.pkl               # Trained model — Small Cap
│   └── universal_xgb.pkl           # Universal multi-cap model
│
├── data/
│   ├── raw/
│   │   └── master_raw.csv          # Raw OHLCV data (Phase 1 output)
│   └── processed/
│       ├── master_features.csv     # Feature-engineered dataset
│       ├── large_cap_features.csv
│       ├── mid_cap_features.csv
│       └── small_cap_features.csv
│
├── reports/
│   └── model_report.csv            # Evaluation metrics per segment
│
├── plots/                          # Confusion matrices, SHAP, ROC curves
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/stockify-ai.git
cd stockify-ai
```

**2. Create and activate a virtual environment**
```bash
conda create -n stock_ai python=3.11
conda activate stock_ai
```

**3. Install dependencies**
```bash
pip install streamlit yfinance pandas numpy scikit-learn xgboost plotly \
            transformers torch ta shap joblib tqdm newsapi-python
```

**4. Regenerate data and models (if not included in repo)**
```bash
# Run notebooks in order:
# 1. Phase1_Ingestion.ipynb
# 2. Phase2_feature_engineering.ipynb
# 3. Phase3_xgboost_model.ipynb
# 4. Phase4_finbert_sentiment.ipynb (optional)
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## 🚀 Usage

Once running, open `http://localhost:8501` in your browser.

| Page | What it does |
|---|---|
| 🏠 Dashboard | System overview, live market snapshot, architecture |
| 🔍 Stock Analysis | Select any stock → get prediction + full technical chart |
| 📊 Batch Prediction | Scan multiple stocks, get heatmap of Bull/Bear signals |
| 📰 Sentiment Monitor | FinBERT sentiment gauge for selected stock |
| 🎯 Model Performance | Real evaluation metrics, radar chart, confusion matrix |
| ℹ️ About | Project overview, tech stack, references |

---

## 🔬 Model Pipeline

```
Phase 2 Output (master_features.csv)
        │
        ▼
Chronological Train/Test Split (80/20 by date — no shuffling)
        │
        ├── Train: 2021-01-01 → 2023-12-31
        └── Test:  2024-01-01 → 2024-12-31
                │
                ▼
        sklearn Pipeline:
        ┌─────────────────────────────┐
        │  RobustScaler               │  ← fitted on train only
        │  XGBClassifier              │  ← scale_pos_weight=1.6
        │    eval_metric = auc        │
        │    tree_method = hist       │
        └─────────────────────────────┘
                │
                ▼
        RandomizedSearchCV (n_iter=60)
        TimeSeriesSplit (n_splits=5)
        Scoring: ROC-AUC
```

**Leakage Prevention:**
- Strict chronological split — no random shuffling
- Scaler fitted on training data only
- Target variable uses `shift(-1)` — today's features predict tomorrow's return
- No future price data used in any indicator

---

## 📈 Results

| Segment | Accuracy | AUC-ROC | Notes |
|---|---|---|---|
| Large Cap | 61.8% | 0.526 | 50 stocks, Nifty 100 |
| Mid Cap | 61.0% | 0.521 | 48 stocks, Nifty Midcap |
| Small Cap | 62.5% | 0.542 | 42 stocks, Smallcap 100 |

> **Note:** AUC ~0.52–0.54 is a realistic and honest result for next-day stock direction prediction using technical indicators alone. This is consistent with published academic literature on NSE/BSE prediction tasks. The value of this project lies in the complete end-to-end pipeline architecture.

---

## ⚠️ Disclaimer

This project is developed **purely for educational purpose**. It does **not** constitute financial advice. Stock market trading involves significant risk of capital loss. Always consult a SEBI-registered investment advisor before making any investment decisions. Past model performance does not guarantee future results.

---

## 📚 References

- Yang et al. (2023) — *Multimodal Time-Series Forecasting*, arXiv:2601.19504
- Araci, D. (2019) — *FinBERT: Financial Sentiment Analysis with BERT*
- ProsusAI/finbert — HuggingFace Model Hub
- Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*, KDD

---

<div align="center">
  <b>📈 Stockify-AI · NSE Multi-Cap Prediction Platform</b><br>
  <sub>XGBoost + FinBERT · 130 Stocks · 18 Indicators · For Educational Use Only</sub>
</div>

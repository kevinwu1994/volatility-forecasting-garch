# Volatility Forecasting with GARCH Models

This project implements and compares volatility forecasting models from the GARCH family, with applications to Value-at-Risk (VaR) and risk management.

---

## Project Overview

The goal of this project is to model and forecast financial market volatility using time series techniques and evaluate their performance in risk management applications.

We focus on:

- Modeling conditional volatility
- Out-of-sample forecast evaluation
- Risk measurement using VaR
- Backtesting using statistical tests

---

## Data

- Assets: SPY, QQQ, DIA (U.S. equity ETFs)
- Frequency: Daily returns
- Period: 2015 – 2025
- Source: Yahoo Finance

---

## Models

We implement and compare the following models:

- GARCH(1,1)
- EGARCH
- GJR-GARCH

These models capture:

- Volatility clustering
- Persistence in variance
- Asymmetric effects (leverage effect)

---

## ⚙️ Methodology

### 1. Stylized Facts Analysis
- Log returns and squared returns
- Autocorrelation function (ACF)
- ARCH effects testing

### 2. Volatility Modeling
- Estimation of GARCH-family models
- Parameter interpretation

### 3. Forecast Evaluation
- Expanding window forecasting
- Loss functions:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - QLIKE

### 4. Risk Management Application
- VaR estimation (95% and 99%)
- Backtesting:
  - Violation rate
  - Kupiec test

---

## 📊 Key Results

- GARCH(1,1) performs best in out-of-sample forecasting
- Asymmetric models (EGARCH, GJR-GARCH) capture leverage effects
- VaR forecasts are statistically consistent based on Kupiec tests
- Models effectively capture tail risk during volatile periods

---

## Project Structure

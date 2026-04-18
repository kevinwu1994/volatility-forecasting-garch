# Volatility Forecasting with GARCH Models

This project studies and compares volatility forecasting models from the GARCH family, with applications to Value-at-Risk (VaR) and risk management.

---

## Project Overview

The objective of this project is to model and forecast financial market volatility and evaluate model performance in risk management applications.

We focus on:

- Modeling conditional volatility
- Out-of-sample forecast evaluation
- Risk measurement using Value-at-Risk (VaR)
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

These models are designed to capture:

- Volatility clustering  
- Persistence in conditional variance  
- Asymmetric responses to shocks (leverage effects)  

---

## Methodology

### 1. Stylized Facts Analysis
- Log returns and squared returns  
- Autocorrelation function (ACF)  
- ARCH effect testing  

### 2. Volatility Modeling
- Estimation of GARCH-family models  
- Maximum likelihood estimation (MLE)  
- Parameter interpretation  

### 3. Forecast Evaluation
- Rolling (or expanding) window out-of-sample forecasting  
- Loss functions:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - QLIKE loss  

### 4. Risk Management Application
- Value-at-Risk (VaR) estimation at 95% and 99% confidence levels  
- Backtesting:
  - Violation rate (hit rate)
  - Kupiec test  

---

## Key Results

- GARCH(1,1) consistently performs best in out-of-sample forecasting  
- Asymmetric models (EGARCH, GJR-GARCH) capture leverage effects but do not improve predictive accuracy  
- VaR forecasts are well-calibrated, with violation rates close to expected levels  
- Kupiec test results fail to reject the null hypothesis across assets  

---

## Limitations

- Conditional normality assumption may underestimate tail risk  
- VaR does not capture the magnitude of extreme losses (tail severity)  
- GARCH models assume stable volatility dynamics and may fail under structural breaks  
- Realized volatility proxy (rolling variance) is only an approximation  

---

## Conclusion

This project shows that a parsimonious GARCH(1,1) model provides robust volatility forecasts and produces statistically valid VaR estimates.

The results highlight that, in practice, model simplicity and out-of-sample robustness are often more important than model complexity in financial risk management.

---

## Project Structure


volatility-forecasting-garch/
│
├── src/
├── results/
├── report/
├── slides/
├── README.md
└── LICENSE
---

## How to Use

1. Review the report for full methodology and results:
   - `report/volatility-forecasting-garch-report.pdf`

2. Check the presentation slides for a concise summary:
   - `slides/volatility-forecasting-garch-slides.pptx`

3. (Optional) Run the model code in `src/` to reproduce results

---

## Key Takeaway

A simple GARCH(1,1) model provides robust volatility forecasts and produces reliable Value-at-Risk estimates when validated through proper backtesting.

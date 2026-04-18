# Project Report

This folder contains the final project report for:

**Volatility Forecasting and Risk Management Using GARCH Models**

---

## File

- [Final Project Report](./volatility-forecasting-garch-report.pdf)

---

## Report Structure

The report is organized as follows:

- **Introduction & Research Objectives**  
  Motivation for volatility modeling and key research questions.

- **Data & Stylized Facts**  
  Empirical characteristics of ETF returns, including volatility clustering, skewness, and excess kurtosis.

- **Volatility Modeling**  
  Estimation and comparison of GARCH-family models: GARCH(1,1), EGARCH, and GJR-GARCH.

- **Out-of-Sample Forecast Evaluation**  
  Rolling-window forecasting framework evaluated using MSE, MAE, and QLIKE loss functions.

- **Value-at-Risk (VaR) Construction & Backtesting**  
  Transformation of volatility forecasts into VaR estimates and validation using the Kupiec test.

- **Risk Management Implications**  
  Interpretation of results in the context of practical risk management applications.

- **Limitations**  
  Model assumptions, distributional constraints, and potential sources of estimation bias.

- **Conclusion & Future Work**  
  Summary of findings and directions for model improvement.

---

## Key Result

The empirical results show that the standard GARCH(1,1) model consistently outperforms more complex specifications in out-of-sample volatility forecasting across all assets.

---

## Main Takeaway

A parsimonious GARCH(1,1) model provides more robust and generalizable volatility forecasts than more complex alternatives, and these forecasts can be effectively translated into well-calibrated and statistically validated VaR-based risk measures.

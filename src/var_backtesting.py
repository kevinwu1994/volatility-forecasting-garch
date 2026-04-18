import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, chi2


def kupiec_test(violations: pd.Series, alpha: float) -> tuple[float, float]:
    """
    Kupiec unconditional coverage test.
    """
    T = len(violations)
    x = violations.sum()

    if x == 0 or x == T:
        return np.nan, np.nan

    p_hat = x / T

    lr_uc = -2 * (
        (T - x) * np.log((1 - alpha) / (1 - p_hat))
        + x * np.log(alpha / p_hat)
    )

    p_value = 1 - chi2.cdf(lr_uc, df=1)
    return lr_uc, p_value


def expanding_var_forecast(
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.Series:
    """
    Generate 1-step-ahead variance forecasts using GARCH(1,1).
    """
    var_preds = []

    for i in range(len(y_test)):
        y_hist = pd.concat([y_train, y_test.iloc[:i]])

        model = arch_model(
            y_hist,
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            mean="constant",
        )

        res = model.fit(disp="off")
        forecast = res.forecast(horizon=1)
        var_pred = forecast.variance.iloc[-1, 0]
        var_preds.append(var_pred)

    return pd.Series(var_preds, index=y_test.index, name="Forecast_Variance")


def var_forecast_backtest(
    asset: str,
    log_returns_rescaled: pd.DataFrame,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
):
    """
    Construct VaR and run Kupiec backtesting for one asset.
    """
    train_returns = log_returns_rescaled.loc[:train_end]
    test_returns = log_returns_rescaled.loc[test_start:]

    y_train = train_returns[asset].dropna()
    y_test = test_returns[asset].dropna()

    var_forecast = expanding_var_forecast(y_train, y_test)
    vol_forecast = np.sqrt(var_forecast)

    mu_hat = y_train.mean()

    z_95 = norm.ppf(0.05)
    z_99 = norm.ppf(0.01)

    var_95 = mu_hat + z_95 * vol_forecast
    var_99 = mu_hat + z_99 * vol_forecast

    actual_returns = y_test.loc[var_forecast.index]

    violations_95 = actual_returns < var_95
    violations_99 = actual_returns < var_99

    n_test = len(actual_returns)
    n_viol_95 = int(violations_95.sum())
    n_viol_99 = int(violations_99.sum())

    viol_rate_95 = n_viol_95 / n_test
    viol_rate_99 = n_viol_99 / n_test

    var_summary_df = pd.DataFrame(
        {
            "Asset": [asset, asset],
            "VaR Level": ["95%", "99%"],
            "Expected Violation Rate": [0.05, 0.01],
            "Actual Violation Rate": [viol_rate_95, viol_rate_99],
            "Number of Violations": [n_viol_95, n_viol_99],
            "Total Observations": [n_test, n_test],
        }
    )

    lr95, p95 = kupiec_test(violations_95, 0.05)
    lr99, p99 = kupiec_test(violations_99, 0.01)

    kupiec_df = pd.DataFrame(
        {
            "Asset": [asset, asset],
            "VaR Level": ["95%", "99%"],
            "Kupiec LR Statistic": [lr95, lr99],
            "Kupiec p-value": [p95, p99],
        }
    )

    plot_df = pd.DataFrame(
        {
            "Actual Return": actual_returns,
            "VaR 95%": var_95,
            "VaR 99%": var_99,
        }
    )

    return var_summary_df, kupiec_df, plot_df


def run_var_backtesting_for_all_assets(
    log_returns_rescaled: pd.DataFrame,
    assets: list[str] | None = None,
):
    """
    Run VaR forecasting and backtesting for all assets.
    """
    if assets is None:
        assets = list(log_returns_rescaled.columns)

    all_var_summaries = []
    all_kupiec_results = {}
    all_plot_data = {}

    for asset in assets:
        var_summary_df, kupiec_df, plot_df = var_forecast_backtest(
            asset=asset,
            log_returns_rescaled=log_returns_rescaled,
        )

        all_var_summaries.append(var_summary_df)
        all_kupiec_results[asset] = kupiec_df
        all_plot_data[asset] = plot_df

    final_var_summary = pd.concat(all_var_summaries, ignore_index=True)

    return final_var_summary, all_kupiec_results, all_plot_data
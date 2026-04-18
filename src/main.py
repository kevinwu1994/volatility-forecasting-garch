# ==============================
# Main Pipeline for GARCH Project
# ==============================

import os
import matplotlib.pyplot as plt

from data_loader import (
    load_price_data,
    compute_log_returns,
    rescale_returns,
    compute_realized_variance,
)

from garch_models import (
    run_arch_test,
    estimate_all_models,
)

from forecast_evaluation import evaluate_all_assets
from var_backtesting import run_var_backtesting_for_all_assets


RESULTS_DIR = "results"


def ensure_results_dir() -> None:
    """
    Create results directory if it does not exist.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_var_forecast(plot_df, asset_name: str) -> None:
    """
    Save VaR backtesting plot for one asset.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df["Actual Return"], label="Actual Returns")
    plt.plot(plot_df.index, plot_df["VaR 95%"], label="95% VaR")
    plt.plot(plot_df.index, plot_df["VaR 99%"], label="99% VaR")
    plt.title(f"{asset_name}: Actual Returns and VaR Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Return / VaR (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/var_backtest_{asset_name.lower()}.png", dpi=300)
    plt.close()


def main() -> None:
    """
    Run the full volatility forecasting and VaR backtesting pipeline.
    """
    ensure_results_dir()

    print("Step 1: Loading price data...")
    price_data = load_price_data()

    print("Step 2: Computing log returns...")
    log_returns = compute_log_returns(price_data)

    print("Step 3: Rescaling returns...")
    log_returns_rescaled = rescale_returns(log_returns)

    print("Step 4: Computing realized variance proxy...")
    realized_var = compute_realized_variance(log_returns_rescaled)

    print("\n==============================")
    print("ARCH-LM Test Results")
    print("==============================")
    arch_results = run_arch_test(log_returns)
    print(arch_results.round(6))

    print("\n==============================")
    print("Estimating GARCH-family models")
    print("==============================")
    model_results = estimate_all_models(log_returns_rescaled)
    print(model_results)

    print("\n==============================")
    print("Running Forecast Evaluation")
    print("==============================")
    forecast_results = evaluate_all_assets(
        log_returns_rescaled=log_returns_rescaled,
        realized_var=realized_var,
    )
    print(forecast_results.round(6))

    print("\n==============================")
    print("Running VaR Backtesting")
    print("==============================")
    var_summary, kupiec_results, plot_data = run_var_backtesting_for_all_assets(
        log_returns_rescaled=log_returns_rescaled
    )

    print("\nVaR Backtesting Summary:")
    print(var_summary.round(6))

    print("\nKupiec Test Results:")
    for asset, df in kupiec_results.items():
        print(f"\n--- {asset} ---")
        print(df.round(6))

    print("\n==============================")
    print("Saving Tables to results/")
    print("==============================")
    arch_results.to_csv(f"{RESULTS_DIR}/arch_test_results.csv", index=False)
    model_results.to_csv(f"{RESULTS_DIR}/garch_model_estimation.csv", index=False)
    forecast_results.to_csv(f"{RESULTS_DIR}/forecast_evaluation_results.csv", index=False)
    var_summary.to_csv(f"{RESULTS_DIR}/var_backtesting_summary.csv", index=False)

    kupiec_combined = []
    for asset, df in kupiec_results.items():
        kupiec_combined.append(df)

    if kupiec_combined:
        kupiec_combined_df = kupiec_combined[0]
        if len(kupiec_combined) > 1:
            import pandas as pd
            kupiec_combined_df = pd.concat(kupiec_combined, ignore_index=True)

        kupiec_combined_df.to_csv(f"{RESULTS_DIR}/kupiec_test_results.csv", index=False)

    print("\n==============================")
    print("Saving VaR Plots to results/")
    print("==============================")
    for asset, df in plot_data.items():
        plot_var_forecast(df, asset)
        print(f"Saved: {RESULTS_DIR}/var_backtest_{asset.lower()}.png")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

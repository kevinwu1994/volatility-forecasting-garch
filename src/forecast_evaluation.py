import numpy as np
import pandas as pd
from arch import arch_model
from garch_models import MODEL_SPECS


def mse_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mae_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(np.abs(y_true - y_pred))


def qlike_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def expanding_window_forecast(
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> pd.Series:
    """
    Generate 1-step-ahead variance forecasts using an expanding window.
    """
    spec = MODEL_SPECS[model_name]
    preds = []

    for i in range(len(y_test)):
        y_hist = pd.concat([y_train, y_test.iloc[:i]])

        model = arch_model(
            y_hist,
            vol=spec["vol"],
            p=spec["p"],
            o=spec["o"],
            q=spec["q"],
            mean="constant",
        )

        res = model.fit(disp="off")
        forecast = res.forecast(horizon=1)
        var_pred = forecast.variance.iloc[-1, 0]
        preds.append(var_pred)

    return pd.Series(preds, index=y_test.index, name=model_name)


def evaluate_asset_forecast(
    asset: str,
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    test_variance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate all models for a single asset.
    """
    y_train = train_returns[asset].dropna()
    y_test = test_returns[asset].dropna()
    rv_test = test_variance[asset].dropna()

    rv_aligned = rv_test.loc[y_test.index]
    results_table = []

    for model_name in MODEL_SPECS.keys():
        pred = expanding_window_forecast(y_train, y_test, model_name)

        results_table.append(
            {
                "Asset": asset,
                "Model": model_name,
                "MSE": mse_loss(rv_aligned, pred),
                "MAE": mae_loss(rv_aligned, pred),
                "QLIKE": qlike_loss(rv_aligned, pred),
            }
        )

    return pd.DataFrame(results_table)


def evaluate_all_assets(
    log_returns_rescaled: pd.DataFrame,
    realized_var: pd.DataFrame,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Evaluate all assets using expanding-window out-of-sample forecasting.
    """
    train_returns = log_returns_rescaled.loc[:train_end]
    test_returns = log_returns_rescaled.loc[test_start:]

    test_variance = realized_var.loc[test_start:]

    all_results = []

    for asset in log_returns_rescaled.columns:
        asset_result = evaluate_asset_forecast(
            asset=asset,
            train_returns=train_returns,
            test_returns=test_returns,
            test_variance=test_variance,
        )
        all_results.append(asset_result)

    return pd.concat(all_results, ignore_index=True)
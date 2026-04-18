import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch


MODEL_SPECS = {
    "GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
    "GJR-GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
}


def run_arch_test(log_returns: pd.DataFrame, nlags: int = 10) -> pd.DataFrame:
    """
    Run ARCH-LM test for each asset.
    """
    rows = []

    for asset in log_returns.columns:
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(log_returns[asset].dropna(), nlags=nlags)

        rows.append(
            {
                "Asset": asset,
                "LM Statistic": lm_stat,
                "LM p-value": lm_pvalue,
                "F Statistic": f_stat,
                "F p-value": f_pvalue,
            }
        )

    return pd.DataFrame(rows)


def fit_single_model(
    series: pd.Series,
    model_name: str,
    model_specs: dict = MODEL_SPECS,
):
    """
    Fit one GARCH-family model to one return series.
    """
    spec = model_specs[model_name]

    model = arch_model(
        series.dropna(),
        vol=spec["vol"],
        p=spec["p"],
        o=spec["o"],
        q=spec["q"],
        mean="constant",
    )

    res = model.fit(disp="off")
    return res


def estimate_all_models(
    log_returns_rescaled: pd.DataFrame,
    model_specs: dict = MODEL_SPECS,
) -> pd.DataFrame:
    """
    Estimate all GARCH-family models for all assets and summarize parameters.
    """
    rows = []

    for asset in log_returns_rescaled.columns:
        for model_name in model_specs.keys():
            res = fit_single_model(log_returns_rescaled[asset], model_name, model_specs)
            params = res.params

            alpha = params.get("alpha[1]", np.nan)
            beta = params.get("beta[1]", np.nan)
            gamma = params.get("gamma[1]", np.nan)

            rows.append(
                {
                    "Asset": asset,
                    "Model": model_name,
                    "mu": round(params.get("mu", np.nan), 4),
                    "omega": round(params.get("omega", np.nan), 4),
                    "alpha": round(alpha, 4) if pd.notna(alpha) else np.nan,
                    "beta": round(beta, 4) if pd.notna(beta) else np.nan,
                    "gamma": round(gamma, 4) if pd.notna(gamma) else np.nan,
                    "alpha+beta": round(alpha + beta, 4)
                    if model_name == "GARCH(1,1)" and pd.notna(alpha) and pd.notna(beta)
                    else np.nan,
                    "AIC": round(res.aic, 2),
                }
            )

    summary_df = pd.DataFrame(rows)
    return summary_df
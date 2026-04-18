import yfinance as yf
import numpy as np
import pandas as pd


TICKERS = ["SPY", "QQQ", "DIA"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


def load_price_data(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.
    """
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data.dropna()


def compute_log_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns.
    """
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    return log_returns


def compute_squared_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute squared log returns for stylized facts analysis.
    """
    return log_returns**2


def rescale_returns(log_returns: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """
    Rescale returns for numerical stability in GARCH estimation.
    """
    return log_returns * scale


def compute_realized_variance(
    log_returns_rescaled: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Use rolling variance as a proxy for realized volatility.
    """
    realized_var = log_returns_rescaled.rolling(window=window).var().dropna()
    return realized_var
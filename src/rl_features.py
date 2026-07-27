"""
Shared feature construction for the RL Quant Model.
This file is imported by both the training environment (rl_env.py)
and the production inference wrapper (rl_agent.py).
It strictly avoids importing gymnasium/torch to remain Vercel-friendly.
"""
from typing import Dict
import numpy as np
import pandas as pd

OBS_DIM = 10

def compute_obs_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the mean/std used for z-score features from a training DataFrame.

    Call this on the training slice and pass the returned dict to both
    ``TradingEnv`` (at construction) and ``RLTraderAgent.predict`` (at
    inference), so the two never drift out of sync.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns produced by DataHandler._add_technical_indicators.

    Returns
    -------
    dict with keys: macd_mean, macd_std, macd_hist_mean, macd_hist_std
    """
    return {
        "macd_mean":      float(df["macd"].mean()),
        "macd_std":       float(df["macd"].std()) or 1.0,
        "macd_hist_mean": float(df["macd_hist"].mean()),
        "macd_hist_std":  float(df["macd_hist"].std()) or 1.0,
    }

def build_obs_vector(row: pd.Series, stats: Dict[str, float]) -> np.ndarray:
    """
    Build the normalised 10-feature observation vector for a single bar.

    This function is the **single source of truth** for feature normalisation.
    It is called by both ``TradingEnv.step`` (training) and
    ``RLTraderAgent.predict`` (inference) to guarantee they use identical logic.

    Parameters
    ----------
    row   : pd.Series or dict-like with the indicator columns.
    stats : dict returned by ``compute_obs_stats`` on the training slice.

    Returns
    -------
    np.ndarray of shape (OBS_DIM,), dtype float32
    """
    def _safe(key, default=0.0):
        v = row.get(key) if hasattr(row, "get") else getattr(row, key, default)
        return float(v) if v is not None and not np.isnan(v) else default

    # 0 — RSI / 100
    f0 = np.clip(_safe("rsi", 50.0) / 100.0, 0.0, 1.0)

    # 1 — MACD histogram z-score, clamped [-3, +3]
    mh_mean = stats.get("macd_hist_mean", 0.0)
    mh_std  = stats.get("macd_hist_std",  1.0)
    f1 = np.clip((_safe("macd_hist") - mh_mean) / mh_std, -3.0, 3.0)

    # 2 — Bollinger band position [0, 1]
    f2 = np.clip(_safe("bb_position", 0.5), 0.0, 1.0)

    # 3 — SMA-20 / SMA-50 ratio − 1, so 0 means cross
    sma20 = _safe("sma_20", 1.0)
    sma50 = _safe("sma_50", 1.0)
    f3 = np.clip((sma20 / sma50 if sma50 != 0 else 1.0) - 1.0, -0.5, 0.5)

    # 4 — Volume ratio capped at 5× then scaled to [0, 1]
    f4 = np.clip(_safe("volume_ratio", 1.0), 0.0, 5.0) / 5.0

    # 5 — Momentum (10d % change) / 100, clamped [-0.5, +0.5]
    f5 = np.clip(_safe("momentum", 0.0) / 100.0, -0.5, 0.5)

    # 6 — HV-14 (annualised vol %) / 100
    f6 = np.clip(_safe("hv_14", 20.0) / 100.0, 0.0, 1.0)

    # 7 — ATR-14 as fraction of close price, clamped [0, 0.2]
    close = _safe("close", 1.0)
    atr   = _safe("atr_14", 0.0)
    f7 = np.clip(atr / close if close > 0 else 0.0, 0.0, 0.2)

    # 8 — Drawdown / −50 → [0, 1]; 0 = at peak, 1 = −50% from peak
    dd = _safe("drawdown", 0.0)          # negative percentage, e.g. −15.3
    f8 = np.clip(dd / -50.0, 0.0, 1.0)

    # 9 — MACD line z-score, clamped [-3, +3]
    m_mean = stats.get("macd_mean", 0.0)
    m_std  = stats.get("macd_std",  1.0)
    f9 = np.clip((_safe("macd") - m_mean) / m_std, -3.0, 3.0)

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)

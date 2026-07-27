"""
scripts/window_backtest.py — Run corrected mark-to-market backtest on any date window
======================================================================================
Usage:
  python scripts/window_backtest.py --start 2023-01-03 --end 2023-10-31
  python scripts/window_backtest.py --start 2025-10-31 --end 2026-07-24   # original test window

Loads the saved model + obs_stats (must have run train_rl_agent.py first).
Slices each ticker's full history to [start, end] and runs the backtest.
Clearly labels whether each window is IN-SAMPLE (inside training split) or
OUT-OF-SAMPLE (inside val/test split).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

from data_handler import DataHandler
from rl_env import TradingEnv, compute_obs_stats, TRANSACTION_COST, ACTION_HOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("window_backtest")

TICKERS    = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
YEARS      = 5
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
MODEL_DIR  = ROOT / "models"
MODEL_PATH = MODEL_DIR / "rl_policy_ppo.zip"
STATS_PATH = MODEL_DIR / "rl_obs_stats.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_full_df(ticker: str) -> pd.DataFrame:
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=YEARS * 365 + 10)).strftime("%Y-%m-%d")
    dh = DataHandler(tickers=[ticker], start_date=start_date, end_date=end_date)
    df = dh._fetch_ohlcv(ticker)
    df = dh._add_technical_indicators(df)
    return df.dropna().reset_index(drop=True)


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows whose date falls within [start, end] inclusive."""
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def split_boundary_dates(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (train_end_date, val_end_date) for reference."""
    n       = len(df)
    i_val   = int(n * TRAIN_FRAC)
    i_test  = int(n * (TRAIN_FRAC + VAL_FRAC))
    return str(df.iloc[i_val - 1]["date"])[:10], str(df.iloc[i_test - 1]["date"])[:10]


# ---------------------------------------------------------------------------
# Mark-to-market backtest (identical to train_rl_agent.py — shared logic)
# ---------------------------------------------------------------------------

def run_backtest(
    model,
    df_window: pd.DataFrame,
    obs_stats: Dict[str, float],
    holding_period: int,
    ticker: str = "",
    in_sample: bool = False,
) -> Dict[str, float]:
    _POS = {2: 1.0, 0: -1.0, 1: 0.0}

    start_row   = df_window.iloc[0]
    end_row     = df_window.iloc[-1]
    start_date  = str(start_row["date"])[:10]
    end_date    = str(end_row["date"])[:10]
    start_price = float(start_row["close"])
    end_price   = float(end_row["close"])
    actual_chg  = (end_price - start_price) / start_price * 100
    sample_tag  = "⚠ IN-SAMPLE" if in_sample else "✓ OUT-OF-SAMPLE"
    logger.info(
        "  %-5s [%s]  $%.2f (%s) → $%.2f (%s)  actual_Δ=%+.1f%%",
        ticker, sample_tag, start_price, start_date, end_price, end_date, actual_chg,
    )

    env = TradingEnv(df_window, obs_stats=obs_stats,
                     holding_period=holding_period, random_start=False)
    obs, _ = env.reset()

    daily_rets_policy: List[float] = []
    daily_rets_bah:    List[float] = []
    prev_action = ACTION_HOLD
    terminated  = False

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(int(action))

        t = info["t"]
        if t + 1 < len(df_window):
            c0 = float(df_window.iloc[t]["close"])
            c1 = float(df_window.iloc[t + 1]["close"])
            day_ret = (c1 - c0) / c0 if c0 > 0 else 0.0
        else:
            day_ret = 0.0

        position    = _POS[int(action)]
        tc          = TRANSACTION_COST if int(action) != prev_action else 0.0
        prev_action = int(action)

        daily_rets_policy.append(position * day_ret - tc)
        daily_rets_bah.append(day_ret)

    policy_arr = np.array(daily_rets_policy)
    bah_arr    = np.array(daily_rets_bah)

    def _cumret(a): return float(np.prod(1.0 + a) - 1.0) * 100.0
    def _sharpe(a, p=252):
        return 0.0 if len(a) < 2 or a.std() == 0 else float(a.mean() / a.std() * np.sqrt(p))
    def _maxdd(a):
        if not len(a): return 0.0
        eq = np.cumprod(1.0 + a); peak = np.maximum.accumulate(eq)
        return float(((eq - peak) / peak).min()) * 100.0

    return {
        "cumret_policy":    _cumret(policy_arr),
        "cumret_bah":       _cumret(bah_arr),
        "actual_price_chg": actual_chg,
        "sharpe_policy":    _sharpe(policy_arr),
        "sharpe_bah":       _sharpe(bah_arr),
        "maxdd_policy":     _maxdd(policy_arr),
        "maxdd_bah":        _maxdd(bah_arr),
        "n_steps":          len(daily_rets_policy),
        "start_date":       start_date,
        "end_date":         end_date,
        "start_price":      start_price,
        "end_price":        end_price,
        "in_sample":        in_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start: str, end: str, holding_period: int):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 not installed — run: pip install -r requirements-training.txt")
        sys.exit(1)

    for p in [MODEL_PATH, STATS_PATH]:
        if not p.exists():
            logger.error("Missing %s — run train_rl_agent.py first.", p)
            sys.exit(1)

    logger.info("Loading model: %s", MODEL_PATH)
    model = PPO.load(str(MODEL_PATH))

    with open(STATS_PATH) as f:
        obs_stats = json.load(f)

    logger.info("Fetching full history for all tickers …")
    rows = []
    for ticker in TICKERS:
        full_df = fetch_full_df(ticker)

        # Determine split boundaries for this ticker to label in/out of sample
        train_end, val_end = split_boundary_dates(full_df)
        # The requested window is in-sample if its end date is before or at train_end
        window_end = pd.Timestamp(end)
        window_is_in_sample = window_end <= pd.Timestamp(train_end)

        df_window = slice_window(full_df, start, end)
        if len(df_window) <= holding_period + 1:
            logger.warning("  %s: window too short (%d rows), skipping", ticker, len(df_window))
            continue

        metrics = run_backtest(
            model, df_window, obs_stats, holding_period,
            ticker=ticker, in_sample=window_is_in_sample,
        )

        logger.info(
            "  %-5s  policy=%+.1f%%  bah=%+.1f%%  actual_Δ=%+.1f%%  "
            "sharpe_pol=%+.2f  sharpe_bah=%+.2f  "
            "maxdd_pol=%+.1f%%  maxdd_bah=%+.1f%%",
            ticker,
            metrics["cumret_policy"], metrics["cumret_bah"],
            metrics["actual_price_chg"],
            metrics["sharpe_policy"], metrics["sharpe_bah"],
            metrics["maxdd_policy"],  metrics["maxdd_bah"],
        )

        rows.append({
            "Ticker":        ticker,
            "Sample":        "IN" if metrics["in_sample"] else "OUT",
            "Steps":         metrics["n_steps"],
            "Policy cum%":   f"{metrics['cumret_policy']:+.1f}%",
            "B&H cum%":      f"{metrics['cumret_bah']:+.1f}%",
            "Actual Δ%":     f"{metrics['actual_price_chg']:+.1f}%",
            "Policy Sharpe": f"{metrics['sharpe_policy']:+.2f}",
            "B&H Sharpe":    f"{metrics['sharpe_bah']:+.2f}",
            "Policy MaxDD%": f"{metrics['maxdd_policy']:+.1f}%",
            "B&H MaxDD%":    f"{metrics['maxdd_bah']:+.1f}%",
        })

    if rows:
        def _avg(key):
            vals = [float(r[key].replace("%","").replace("+","")) for r in rows]
            return sum(vals) / len(vals)
        logger.info("-" * 70)
        logger.info(
            "  %-5s  policy=%+.1f%%  bah=%+.1f%%  actual_Δ=%+.1f%%  "
            "sharpe_pol=%+.2f  sharpe_bah=%+.2f  "
            "maxdd_pol=%+.1f%%  maxdd_bah=%+.1f%%",
            "AVG",
            _avg("Policy cum%"), _avg("B&H cum%"), _avg("Actual Δ%"),
            _avg("Policy Sharpe"), _avg("B&H Sharpe"),
            _avg("Policy MaxDD%"), _avg("B&H MaxDD%"),
        )
        print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mark-to-market backtest on a specific date window")
    parser.add_argument("--start", required=True, help="Window start date, e.g. 2023-01-03")
    parser.add_argument("--end",   required=True, help="Window end date,   e.g. 2023-10-31")
    parser.add_argument("--holding-period", type=int, default=5)
    args = parser.parse_args()
    main(start=args.start, end=args.end, holding_period=args.holding_period)

"""
scripts/train_rl_agent.py — Train a shared PPO policy for NEXUS RL Trader
=========================================================================
Usage
-----
  cd <repo_root>
  python scripts/train_rl_agent.py [--timesteps 200000] [--holding-period 5]

Design choices
--------------
ONE shared policy across all 5 tickers (AAPL, GOOGL, MSFT, TSLA, NVDA):
  • Pro: ~5× more data → better generalisation, single .zip for inference.
  • Pro: Normalisation (ATR-%-of-price, z-score MACD) already removes
    cross-ticker scale differences.
  • Pro: At inference we call predict() once regardless of ticker.
  • Con: Tickers have different beta/volatility profiles.
  → If the shared policy underperforms B&H on every ticker, a
    one-policy-per-ticker variant is a ≤1-hour change.

Split (per ticker, chronological — NEVER shuffled)
  train : earliest 70%
  val   : next 15%   (reserved for early-stopping in future iterations)
  test  : final 15%  (held out; used ONLY for the backtest below)

Backtest metrics reported (per ticker + average)
  • Cumulative return (%) — policy vs. buy-and-hold
  • Annualised Sharpe ratio (daily returns, 252 bars)
  • Max drawdown (%)
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow imports from project root and src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

from data_handler import DataHandler
from rl_env import TradingEnv, compute_obs_stats, TRANSACTION_COST, ACTION_HOLD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("train_rl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKERS        = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
YEARS_HISTORY  = 5           # ~5 years of daily bars per ticker
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
# TEST_FRAC    = 0.15  (implicit: remainder)
MODEL_DIR      = ROOT / "models"
MODEL_PATH     = MODEL_DIR / "rl_policy_ppo.zip"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_ticker_df(ticker: str, years: int = YEARS_HISTORY) -> pd.DataFrame:
    """Fetch OHLCV + indicators for a single ticker spanning `years` of history."""
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365 + 10)).strftime("%Y-%m-%d")

    dh = DataHandler(tickers=[ticker], start_date=start_date, end_date=end_date)
    df = dh._fetch_ohlcv(ticker)
    df = dh._add_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df chronologically into (train, val, test). NO shuffle."""
    n     = len(df)
    i_val  = int(n * TRAIN_FRAC)
    i_test = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:i_val].copy(), df.iloc[i_val:i_test].copy(), df.iloc[i_test:].copy()


def print_splits(ticker: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    def _rng(d):
        return f"{str(d['date'].iloc[0])[:10]} → {str(d['date'].iloc[-1])[:10]}  ({len(d)} bars)"
    logger.info(f"  {ticker}  train: {_rng(train)}")
    logger.info(f"  {ticker}  val:   {_rng(val)}")
    logger.info(f"  {ticker}  test:  {_rng(test)}")


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

def run_backtest(
    model,
    df_test: pd.DataFrame,
    obs_stats: Dict[str, float],
    holding_period: int = 5,
    ticker: str = "",
) -> Dict[str, float]:
    """
    Mark-to-market daily backtest — NO overlapping forward windows.

    At each bar t the policy picks an action, which sets a position for that
    day.  The realized P&L for bar t is simply:

        position_t × (close[t+1] - close[t]) / close[t]  -  TC_if_changed

    where position_t: BUY(2)=+1, SELL(0)=-1, HOLD(1)=0.

    This replaces the previous approach of compounding the holding-period
    forward return at every step, which counted every price move
    `holding_period` times (once per overlapping step window).

    B&H baseline uses position=+1 every day, so its cumulative return
    should equal (close[-1] - close[0]) / close[0] — a sanity-checkable
    number against real price history.
    """
    _POS = {2: 1.0, 0: -1.0, 1: 0.0}   # action → position

    # ------------------------------------------------------------------
    # Print start/end dates and actual price move for sanity checking
    # ------------------------------------------------------------------
    start_row   = df_test.iloc[0]
    end_row     = df_test.iloc[-1]
    start_date  = str(start_row["date"])[:10]
    end_date    = str(end_row["date"])[:10]
    start_price = float(start_row["close"])
    end_price   = float(end_row["close"])
    actual_chg  = (end_price - start_price) / start_price * 100
    logger.info(
        "  %-5s  price: $%.2f (%s) → $%.2f (%s)  actual_chg=%+.1f%%",
        ticker, start_price, start_date, end_price, end_date, actual_chg,
    )

    env = TradingEnv(
        df_test,
        obs_stats=obs_stats,
        holding_period=holding_period,
        random_start=False,
    )
    obs, _ = env.reset()

    daily_rets_policy: List[float] = []
    daily_rets_bah:    List[float] = []

    prev_action = ACTION_HOLD
    terminated  = False

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(int(action))

        t = info["t"]   # bar index where the action was taken

        # 1-day forward return: close[t+1] / close[t] - 1
        # t+1 is always valid because max_start = len(df)-holding_period-1 ≤ len(df)-2
        if t + 1 < len(df_test):
            c0 = float(df_test.iloc[t]["close"])
            c1 = float(df_test.iloc[t + 1]["close"])
            day_ret = (c1 - c0) / c0 if c0 > 0 else 0.0
        else:
            day_ret = 0.0

        position = _POS[int(action)]
        tc       = TRANSACTION_COST if int(action) != prev_action else 0.0
        prev_action = int(action)

        daily_rets_policy.append(position * day_ret - tc)
        daily_rets_bah.append(day_ret)    # B&H: always long, no TC

    policy_arr = np.array(daily_rets_policy)
    bah_arr    = np.array(daily_rets_bah)

    def _cumret(arr: np.ndarray) -> float:
        return float(np.prod(1.0 + arr) - 1.0) * 100.0

    def _sharpe(arr: np.ndarray, periods: int = 252) -> float:
        if len(arr) < 2 or arr.std() == 0:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(periods))

    def _maxdd(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.0
        eq   = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / peak
        return float(dd.min()) * 100.0

    return {
        "cumret_policy":    _cumret(policy_arr),
        "cumret_bah":       _cumret(bah_arr),
        "actual_price_chg": actual_chg,          # sanity check: bah ≈ this
        "sharpe_policy":    _sharpe(policy_arr),
        "sharpe_bah":       _sharpe(bah_arr),
        "maxdd_policy":     _maxdd(policy_arr),
        "maxdd_bah":        _maxdd(bah_arr),
        "n_steps":          len(daily_rets_policy),
        "start_date":       start_date,
        "end_date":         end_date,
        "start_price":      start_price,
        "end_price":        end_price,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(timesteps: int = 200_000, holding_period: int = 5):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        logger.error(
            "stable-baselines3 not installed. "
            "Run: pip install -r requirements-training.txt"
        )
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fetch data + split
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Fetching ~5 years of OHLCV for %s", TICKERS)
    logger.info("=" * 60)

    ticker_data = {}
    for ticker in TICKERS:
        logger.info("Downloading %s …", ticker)
        df = fetch_ticker_df(ticker)
        logger.info("  %s: %d bars after dropna", ticker, len(df))
        ticker_data[ticker] = df

    logger.info("\nChronological splits (NO shuffle):")
    splits: Dict[str, Tuple] = {}
    for ticker, df in ticker_data.items():
        train_df, val_df, test_df = chronological_split(df)
        splits[ticker] = (train_df, val_df, test_df)
        print_splits(ticker, train_df, val_df, test_df)

    # ------------------------------------------------------------------
    # 2. Compute normalisation stats from training data only
    #    Use the concatenated training slice of all tickers so the shared
    #    policy sees consistent z-score scaling across tickers.
    # ------------------------------------------------------------------
    all_train = pd.concat([splits[t][0] for t in TICKERS], ignore_index=True)
    obs_stats = compute_obs_stats(all_train)
    logger.info("\nGlobal obs stats (from all training bars):\n  %s", obs_stats)

    # Save stats alongside the model so rl_agent.py can load them
    import json
    stats_path = MODEL_DIR / "rl_obs_stats.json"
    with open(stats_path, "w") as f:
        json.dump(obs_stats, f, indent=2)
    logger.info("Obs stats saved → %s", stats_path)

    # ------------------------------------------------------------------
    # 3. Build vectorised training environment (one env per ticker)
    # ------------------------------------------------------------------
    def _make_env_fn(ticker: str):
        def _fn():
            train_df = splits[ticker][0]
            return TradingEnv(
                train_df,
                obs_stats=obs_stats,
                holding_period=holding_period,
                random_start=True,      # ← randomised start for training diversity
            )
        return _fn

    logger.info("\nBuilding %d parallel training envs (one per ticker) …", len(TICKERS))
    env_fns = [_make_env_fn(t) for t in TICKERS]

    # Try SubprocVecEnv (parallel); fall back to DummyVecEnv on Windows/notebooks
    try:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    except Exception as exc:
        logger.warning("SubprocVecEnv failed (%s) — falling back to DummyVecEnv", exc)
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)

    # ------------------------------------------------------------------
    # 4. Build validation env (unused during training; reserved for future
    #    EvalCallback early-stopping). Use DummyVecEnv — no multiprocessing needed.
    # ------------------------------------------------------------------
    from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv
    val_df_aapl = splits["AAPL"][1]
    eval_env = VecMonitor(
        _DummyVecEnv([lambda: TradingEnv(
            val_df_aapl, obs_stats=obs_stats,
            holding_period=holding_period, random_start=False
        )])
    )

    # ------------------------------------------------------------------
    # 5. PPO model
    # ------------------------------------------------------------------
    logger.info("\nInitialising PPO …")
    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = 3e-4,
        n_steps       = 2048,       # steps per env before each update
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        verbose       = 1,
        tensorboard_log=str(MODEL_DIR / "tb_logs"),
        seed          = 42,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("\nTraining PPO for %d timesteps …", timesteps)
    logger.info("(Each timestep = one env.step() across all %d envs in parallel)", len(TICKERS))

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save(str(MODEL_PATH))
    logger.info("\n✓ Model saved → %s", MODEL_PATH)
    vec_env.close()

    # ------------------------------------------------------------------
    # 7. Backtest on held-out test slices (sequential, random_start=False)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST — held-out test slices (15%% per ticker)")
    logger.info("=" * 60)

    rows = []
    for ticker in TICKERS:
        _, _, test_df = splits[ticker]
        if len(test_df) <= holding_period + 1:
            logger.warning("%s test slice too short (%d rows), skipping", ticker, len(test_df))
            continue

        metrics = run_backtest(model, test_df, obs_stats, holding_period, ticker=ticker)
        rows.append({
            "Ticker":           ticker,
            "Steps":            metrics["n_steps"],
            "Policy cum%":      f"{metrics['cumret_policy']:+.1f}%",
            "B&H cum%":         f"{metrics['cumret_bah']:+.1f}%",
            "Actual Δ%":        f"{metrics['actual_price_chg']:+.1f}%",
            "Policy Sharpe":    f"{metrics['sharpe_policy']:+.2f}",
            "B&H Sharpe":       f"{metrics['sharpe_bah']:+.2f}",
            "Policy MaxDD%":    f"{metrics['maxdd_policy']:+.1f}%",
            "B&H MaxDD%":       f"{metrics['maxdd_bah']:+.1f}%",
        })

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

    # Average across tickers
    if rows:
        def _avg(key):
            vals = [float(r[key].replace("%","").replace("+","")) for r in rows]
            return sum(vals) / len(vals)

        logger.info("-" * 60)
        logger.info(
            "  %-6s  policy=%+.1f%%  bah=%+.1f%%  "
            "sharpe_pol=%+.2f  sharpe_bah=%+.2f  "
            "maxdd_pol=%+.1f%%  maxdd_bah=%+.1f%%",
            "AVG",
            _avg("Policy cum%"), _avg("B&H cum%"),
            _avg("Policy Sharpe"), _avg("B&H Sharpe"),
            _avg("Policy MaxDD%"), _avg("B&H MaxDD%"),
        )

    logger.info("\n" + "=" * 60)
    logger.info("STOP — awaiting approval of backtest results before Phase 3.")
    logger.info("=" * 60)

    # Pretty print as a table
    try:
        summary = pd.DataFrame(rows)
        print("\n" + summary.to_string(index=False))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO RL trading policy")
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total PPO training timesteps (default: 200,000)"
    )
    parser.add_argument(
        "--holding-period", type=int, default=5,
        help="Forward-return horizon in trading days (default: 5)"
    )
    parser.add_argument(
        "--backtest-only", action="store_true",
        help="Skip training; load saved model and rerun backtest only"
    )
    args = parser.parse_args()

    if args.backtest_only:
        # ------------------------------------------------------------------
        # Backtest-only mode: load saved model + obs_stats, skip training
        # ------------------------------------------------------------------
        import json
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable-baselines3 not installed — run: pip install -r requirements-training.txt")
            sys.exit(1)

        if not MODEL_PATH.exists():
            logger.error("Model not found at %s — train first.", MODEL_PATH)
            sys.exit(1)

        stats_path = MODEL_DIR / "rl_obs_stats.json"
        if not stats_path.exists():
            logger.error("Obs stats not found at %s — train first.", stats_path)
            sys.exit(1)

        logger.info("Loading model from %s", MODEL_PATH)
        model = PPO.load(str(MODEL_PATH))

        with open(stats_path) as f:
            obs_stats = json.load(f)
        logger.info("Loaded obs stats: %s", obs_stats)

        # Fetch fresh data and re-split identically to training run
        logger.info("Fetching data for backtest …")
        splits: Dict[str, Tuple] = {}
        for ticker in TICKERS:
            df = fetch_ticker_df(ticker)
            train_df, val_df, test_df = chronological_split(df)
            splits[ticker] = (train_df, val_df, test_df)

        # Run backtest
        logger.info("\n" + "=" * 60)
        logger.info("CORRECTED BACKTEST (mark-to-market 1-day returns)")
        logger.info("B&H cum%% should closely match Actual Δ%% — sanity check")
        logger.info("=" * 60)

        rows = []
        for ticker in TICKERS:
            _, _, test_df = splits[ticker]
            if len(test_df) <= args.holding_period + 1:
                continue
            metrics = run_backtest(model, test_df, obs_stats, args.holding_period, ticker=ticker)
            rows.append({
                "Ticker":        ticker,
                "Steps":         metrics["n_steps"],
                "Policy cum%":   f"{metrics['cumret_policy']:+.1f}%",
                "B&H cum%":      f"{metrics['cumret_bah']:+.1f}%",
                "Actual Δ%":     f"{metrics['actual_price_chg']:+.1f}%",
                "Policy Sharpe": f"{metrics['sharpe_policy']:+.2f}",
                "B&H Sharpe":    f"{metrics['sharpe_bah']:+.2f}",
                "Policy MaxDD%": f"{metrics['maxdd_policy']:+.1f}%",
                "B&H MaxDD%":    f"{metrics['maxdd_bah']:+.1f}%",
            })
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

        if rows:
            def _avg(key):
                vals = [float(r[key].replace("%","").replace("+","")) for r in rows]
                return sum(vals) / len(vals)
            logger.info("-" * 60)
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
    else:
        main(timesteps=args.timesteps, holding_period=args.holding_period)

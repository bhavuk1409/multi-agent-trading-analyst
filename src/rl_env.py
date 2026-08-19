"""
TradingEnv — Gymnasium environment for RL-based trading decisions
=================================================================
A single-asset, daily-bar environment that rewards an agent for correctly
predicting the direction of the *next holding_period trading days* of returns.

Observation space (10 normalized features)
------------------------------------------
All features are derived exclusively from columns produced by
``DataHandler._add_technical_indicators``.  Normalization is computed
from statistics of the DataFrame passed at construction time (never
from future bars), so there is **no lookahead bias**.

Feature layout (fixed order, used identically in ``rl_agent.py``):

  idx  raw column        normalization
  ---  ---------------   ---------------------------------------------------
  0    rsi               / 100                     → [0, 1]
  1    macd_hist         z-score (df stats), clamp → [-3, +3]
  2    bb_position       as-is (already [0, 1])   → [0, 1]
  3    sma_20/sma_50     ratio − 1.0               → centred at 0; ~[-0.2, +0.2]
  4    volume_ratio      min(x, 5) / 5             → [0, 1], outliers capped
  5    momentum          / 100, clamp [-0.5, +0.5] → scale-free pct-change
  6    hv_14             / 100                     → ~[0, 1] (annualised vol %)
  7    atr_14 / close    clamp [0, 0.2]            → price-agnostic ATR
  8    drawdown          / −50, clamp [0, 1]       → 0=no drawdown, 1=−50% dd
  9    macd              z-score (df stats), clamp → [-3, +3]

Action space: Discrete(3)
  0 = SELL  |  1 = HOLD  |  2 = BUY

Reward (step t)
  r = direction(action) × forward_return(t, t+holding_period)
        − transaction_cost × I(action ≠ prev_action)

  where forward_return = (close[t+H] − close[t]) / close[t]
        direction: BUY=+1, SELL=−1, HOLD=0
        transaction_cost = 0.0005  (5 bps)

No-lookahead guarantee
  The observation at step t uses only df.iloc[t] (rolling/EWM indicators
  that look *backward* over past bars).  The reward at step t uses
  close[t + holding_period], but that price is **not part of the
  observation** — it is the target the agent is trying to predict.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRANSACTION_COST = 0.0005  # 5 bps, applied on position change

from rl_features import build_obs_vector, compute_obs_stats, OBS_DIM

# Action encoding
ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY  = 2
_DIRECTION  = {ACTION_BUY: 1.0, ACTION_SELL: -1.0, ACTION_HOLD: 0.0}




# ---------------------------------------------------------------------------
# TradingEnv
# ---------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """
    Single-asset discrete-action trading environment.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicator DataFrame for ONE ticker, sorted ascending by date.
        Must contain columns: close, rsi, macd, macd_hist, macd_signal,
        bb_position, sma_20, sma_50, volume_ratio, momentum, hv_14, atr_14,
        drawdown.  Produced by DataHandler._add_technical_indicators.
    obs_stats : dict
        Pre-computed mean/std for z-score features from ``compute_obs_stats``.
        Pass stats computed on the training slice so evaluation/inference are
        consistent.
    holding_period : int
        Number of trading days to look forward for the reward.  Default 5.
    random_start : bool
        If True (training mode), each reset() picks a random valid start index.
        If False (eval/backtest mode), always starts at index 0.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        obs_stats: Dict[str, float],
        holding_period: int = 5,
        random_start: bool = True,
    ):
        super().__init__()

        # ---- data -----------------------------------------------------------
        self.df = df.reset_index(drop=True)
        self.obs_stats = obs_stats
        self.holding_period = holding_period
        self.random_start = random_start

        # Last valid start index so that t + holding_period is still in bounds
        self._max_start = len(self.df) - holding_period - 1
        if self._max_start < 0:
            raise ValueError(
                f"DataFrame too short ({len(self.df)} rows) for "
                f"holding_period={holding_period}"
            )

        # ---- spaces ---------------------------------------------------------
        self.observation_space = spaces.Box(
            low  = np.array([-3, -3,  0, -0.5, 0, -0.5, 0,  0, 0, -3], dtype=np.float32),
            high = np.array([ 1,  3,  1,  0.5, 1,  0.5, 1, 0.2, 1,  3], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=sell, 1=hold, 2=buy

        # ---- episode state --------------------------------------------------
        self._step_idx: int  = 0
        self._prev_action: int = ACTION_HOLD

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self.random_start:
            self._step_idx = int(self.np_random.integers(0, self._max_start + 1))
        else:
            self._step_idx = 0

        self._prev_action = ACTION_HOLD
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance one trading day.

        Parameters
        ----------
        action : int  — 0=SELL, 1=HOLD, 2=BUY

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        t = self._step_idx
        row = self.df.iloc[t]

        # ---- reward ---------------------------------------------------------
        close_t  = float(row["close"])
        close_t_H = float(self.df.iloc[t + self.holding_period]["close"])

        forward_return = (close_t_H - close_t) / close_t if close_t > 0 else 0.0
        direction      = _DIRECTION[int(action)]
        signed_return  = direction * forward_return

        position_changed = int(action) != int(self._prev_action)
        tc = TRANSACTION_COST if position_changed else 0.0

        reward = float(signed_return - tc)

        # ---- advance --------------------------------------------------------
        self._prev_action = int(action)
        self._step_idx += 1

        terminated = self._step_idx > self._max_start
        truncated  = False

        obs  = self._get_obs() if not terminated else np.zeros(OBS_DIM, dtype=np.float32)
        info = {
            "t":              t,
            "close_t":        close_t,
            "close_t_H":      close_t_H,
            "forward_return": forward_return,
            "action":         int(action),
            "reward":         reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # headless environment

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build observation for the current step index (no future data)."""
        row = self.df.iloc[self._step_idx]
        return build_obs_vector(row, self.obs_stats)

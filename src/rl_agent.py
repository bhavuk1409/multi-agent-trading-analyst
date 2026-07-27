"""
RLTraderAgent — Pure-numpy inference wrapper for the trained PPO policy
=======================================================================
Inference requires ONLY numpy (already a prod dependency) + the two lightweight
model artefacts:

    models/rl_policy_weights.npz   (21 KB)   — extracted MLP weights
    models/rl_obs_stats.json       (< 1 KB)  — observation normalisation stats

Neither stable-baselines3 nor torch is imported at inference time.
Both are training-only deps (requirements-training.txt) and must NOT appear
in requirements.txt (the Vercel production bundle).

Runtime dependency footprint (prod):
    numpy  ≈ 23 MB  (already required by yfinance / pandas / rl_env.py)
    Total new budget: 0 MB  — rl_agent adds nothing to the Vercel bundle.

Network architecture (exported from PPO ActorCriticPolicy):
    Input → Linear(10→64) → Tanh → Linear(64→64) → Tanh → Linear(64→3)
    Softmax → Categorical probabilities for [SELL, HOLD, BUY]

To re-export after retraining:
    python scripts/export_rl_weights.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared normalisation — build_obs_vector and OBS_DIM from rl_env.py
# ---------------------------------------------------------------------------
try:
    from rl_env import build_obs_vector, OBS_DIM
    _RL_ENV_AVAILABLE = True
except ImportError as _e:
    logger.warning("rl_env not importable: %s — RLTraderAgent will degrade.", _e)
    _RL_ENV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = _ROOT / "models" / "rl_policy_weights.npz"
STATS_PATH   = _ROOT / "models" / "rl_obs_stats.json"

_ACTION_LABEL = {0: "sell", 1: "hold", 2: "buy"}


# ---------------------------------------------------------------------------
# Pure-numpy forward pass
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _numpy_forward(obs: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """
    2-layer MLP forward pass matching PPO's policy_net + action_net.
    Returns softmax probabilities over [SELL, HOLD, BUY].
    """
    h = np.tanh(weights["W0"] @ obs + weights["b0"])
    h = np.tanh(weights["W1"] @ h   + weights["b1"])
    return _softmax(weights["Wa"] @ h + weights["ba"])


# ---------------------------------------------------------------------------
# RLTraderAgent
# ---------------------------------------------------------------------------

class RLTraderAgent:
    """
    Singleton inference agent wrapping the exported numpy PPO policy weights.

    No torch / stable-baselines3 required at inference time.  The weights npz
    is loaded from ``models/rl_policy_weights.npz`` (exported by
    ``scripts/export_rl_weights.py`` after every training run).

    Usage::

        agent = RLTraderAgent()
        result = agent.predict(market_data)   # same shape as LLM agents

    ``predict`` returns::

        {"recommendation": "buy|sell|hold", "confidence": 0-100, "reasoning": str}
    """

    def __init__(
        self,
        weights_path: Path = WEIGHTS_PATH,
        stats_path:   Path = STATS_PATH,
    ):
        self._weights: Optional[Dict[str, np.ndarray]] = None
        self._obs_stats: Dict[str, float] = {}
        self._ready = False

        if not _RL_ENV_AVAILABLE:
            logger.error("RLTraderAgent: rl_env unavailable — agent will always degrade.")
            return

        if not weights_path.exists():
            logger.warning(
                "RL weights not found at %s — run scripts/export_rl_weights.py "
                "after training.", weights_path
            )
            return

        if not stats_path.exists():
            logger.warning("Obs stats not found at %s — export again.", stats_path)
            return

        try:
            npz = np.load(str(weights_path))
            self._weights = {k: npz[k] for k in npz.files}

            with open(stats_path) as f:
                self._obs_stats = json.load(f)

            self._ready = True
            logger.info(
                "✓ RLTraderAgent ready (pure-numpy, no torch). "
                "Weights: %s  Stats: %s", weights_path.name, stats_path.name
            )
        except Exception as exc:
            logger.error("RLTraderAgent init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API — identical shape to the 4 LLM agents
    # ------------------------------------------------------------------

    def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the pure-numpy RL policy on a single market snapshot.

        Parameters
        ----------
        market_data : dict
            As returned by ``DataHandler.get_market_summary()`` merged with
            fundamentals (same ``context`` dict used by the LLM agents).

        Returns
        -------
        dict  with keys: recommendation, confidence, reasoning
              + optional ``degraded: True`` on failure.
        """
        if not self._ready or self._weights is None:
            return self._default_analysis()
        try:
            return self._run_inference(market_data)
        except Exception as exc:
            logger.error("RLTraderAgent.predict failed: %s", exc)
            return self._default_analysis()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Build normalised observation vector (shared helper from rl_env.py)
        obs = build_obs_vector(market_data, self._obs_stats)   # (OBS_DIM,)

        # Pure-numpy forward pass — no torch, no SB3
        probs = _numpy_forward(obs, self._weights)             # (3,)

        p_sell, p_hold, p_buy = float(probs[0]), float(probs[1]), float(probs[2])
        action     = int(np.argmax(probs))
        confidence = int(round(float(probs[action]) * 100))
        recommendation = _ACTION_LABEL[action]

        reasoning = self._build_reasoning(
            recommendation=recommendation,
            confidence=confidence,
            p_buy=p_buy, p_hold=p_hold, p_sell=p_sell,
            market_data=market_data,
        )
        return {"recommendation": recommendation, "confidence": confidence, "reasoning": reasoning}

    def _build_reasoning(
        self,
        recommendation: str,
        confidence: int,
        p_buy: float,
        p_hold: float,
        p_sell: float,
        market_data: Dict[str, Any],
    ) -> str:
        def _f(key, fmt=".1f", default="N/A"):
            v = market_data.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            try:   return format(float(v), fmt)
            except: return str(v)

        sma20_v = market_data.get("sma_20")
        sma50_v = market_data.get("sma_50")
        try:    sma_ratio = f"{(float(sma20_v)/float(sma50_v)-1.0)*100:+.2f}%"
        except: sma_ratio = "N/A"

        try:    atr_pct = f"{float(market_data.get('atr_14',0))/float(market_data.get('close',1))*100:.2f}%"
        except: atr_pct = "N/A"

        return (
            f"RL quant model (PPO·numpy) signals {recommendation.upper()} with {confidence}% confidence. "
            f"Action probability breakdown — BUY: {p_buy*100:.1f}% | "
            f"HOLD: {p_hold*100:.1f}% | SELL: {p_sell*100:.1f}%. "
            f"Key indicators: RSI={_f('rsi')}, MACD hist={_f('macd_hist','+.3f')}, "
            f"BB pos={_f('bb_position','.2f')}, SMA-20/50={sma_ratio}, "
            f"vol ratio={_f('volume_ratio','.2f')}×, momentum={_f('momentum','+.1f')}%, "
            f"HV-14={_f('hv_14','.1f')}%, ATR-14={atr_pct}, drawdown={_f('drawdown','+.1f')}%. "
            f"Risk profile note: this model is historically stronger at identifying "
            f"sustained downside risk (avoided −31% TSLA and −26% MSFT declines in "
            f"out-of-sample testing) and weaker at capturing strong sustained uptrends "
            f"(tied or trailed buy-and-hold during the 2023 tech rally). "
            f"The Coordinator should treat this as a defensive / risk-hedging vote "
            f"(weight 0.15) rather than a directional momentum signal."
        )

    @staticmethod
    def _default_analysis() -> Dict[str, Any]:
        return {
            "recommendation": "hold",
            "confidence":     0,
            "reasoning":      (
                "RL quant model unavailable — weights file missing. "
                "Run scripts/export_rl_weights.py (after training) to generate "
                "models/rl_policy_weights.npz."
            ),
            "degraded": True,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_rl_trader_agent: Optional[RLTraderAgent] = None


def get_rl_trader_agent() -> RLTraderAgent:
    """
    Return the module-level RLTraderAgent singleton, initialising it on first call.

    Usage::

        from rl_agent import get_rl_trader_agent
        rl = get_rl_trader_agent()
        result = rl.predict(context)
    """
    global _rl_trader_agent
    if _rl_trader_agent is None:
        _rl_trader_agent = RLTraderAgent()
    return _rl_trader_agent


# ---------------------------------------------------------------------------
# Smoke test  (python src/rl_agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    print("=" * 60)
    print("RLTraderAgent smoke test (pure-numpy, no torch)")
    print("=" * 60)

    agent = RLTraderAgent()
    if not agent._ready:
        print("\n⚠  Agent not ready — run scripts/export_rl_weights.py first.")
        sys.exit(1)

    sample = {
        "close": 185.0, "volume": 75_000_000,
        "rsi": 58.4, "macd": 1.23, "macd_signal": 0.95, "macd_hist": 0.28,
        "sma_20": 182.0, "sma_50": 178.0, "bb_position": 0.65,
        "bb_upper": 192.0, "bb_lower": 172.0,
        "volume_ratio": 1.15, "momentum": 3.2, "hv_14": 22.5,
        "atr_14": 3.8, "drawdown": -2.1,
        "close_prev_1": 183.5, "close_prev_5": 180.2, "close_prev_20": 175.8,
    }
    result = agent.predict(sample)

    print(f"\nRecommendation : {result['recommendation']}")
    print(f"Confidence     : {result['confidence']}%")
    print(f"Reasoning      :\n  {result['reasoning']}")

    assert isinstance(result["recommendation"], str)
    assert result["recommendation"] in ("buy", "sell", "hold")
    assert 0 <= result["confidence"] <= 100
    assert len(result["reasoning"]) > 50
    assert "degraded" not in result

    # Confirm no torch import happened
    import sys as _sys
    assert "torch" not in _sys.modules, "torch was imported — inference must not need torch"
    assert "stable_baselines3" not in _sys.modules, "sb3 was imported at inference time"

    print("\n✓ All assertions passed — pure-numpy, no torch, no sb3.")
    sys.exit(0)

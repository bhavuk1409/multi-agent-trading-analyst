"""
tests/test_rl_env.py — pytest + standalone runner for TradingEnv
================================================================
Run as pytest:   pytest tests/test_rl_env.py -v
Run standalone:  python tests/test_rl_env.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from rl_env import (
    TradingEnv,
    build_obs_vector,
    compute_obs_stats,
    ACTION_BUY,
    ACTION_SELL,
    ACTION_HOLD,
    OBS_DIM,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV fixture + real indicators (via DataHandler)."""
    from data_handler import DataHandler

    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0005, 0.015, n)
    close   = 100.0 * np.exp(np.cumsum(log_ret))
    dates   = pd.date_range("2020-01-02", periods=n, freq="B")

    df = pd.DataFrame({
        "date":   dates,
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.000, 0.015, n)),
        "low":    close * (1 - rng.uniform(0.000, 0.015, n)),
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })

    dh = DataHandler.__new__(DataHandler)
    df = dh._add_technical_indicators(df)
    return df.dropna().reset_index(drop=True)


@pytest.fixture(scope="module")
def df():
    return _make_df(300)


@pytest.fixture(scope="module")
def stats(df):
    return compute_obs_stats(df)


@pytest.fixture(scope="module")
def env(df, stats):
    return TradingEnv(df, obs_stats=stats, holding_period=5, random_start=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObsStats:
    def test_keys_present(self, stats):
        for key in ("macd_mean", "macd_std", "macd_hist_mean", "macd_hist_std"):
            assert key in stats, f"Missing key: {key}"

    def test_std_positive(self, stats):
        assert stats["macd_std"] > 0
        assert stats["macd_hist_std"] > 0


class TestBuildObsVector:
    def test_shape(self, df, stats):
        row = df.iloc[60]
        obs = build_obs_vector(row, stats)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_dtype(self, df, stats):
        obs = build_obs_vector(df.iloc[60], stats)
        assert obs.dtype == np.float32

    def test_finite(self, df, stats):
        for i in range(0, len(df), 10):
            obs = build_obs_vector(df.iloc[i], stats)
            assert np.all(np.isfinite(obs)), f"Non-finite obs at row {i}: {obs}"

    def test_bounds(self, env, df, stats):
        lo = env.observation_space.low
        hi = env.observation_space.high
        for i in range(0, len(df), 10):
            obs = build_obs_vector(df.iloc[i], stats)
            assert np.all(obs >= lo - 1e-6), f"Below lower bound at row {i}: {obs}"
            assert np.all(obs <= hi + 1e-6), f"Above upper bound at row {i}: {obs}"


class TestTradingEnv:
    def test_spaces(self, env):
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.action_space.n == 3

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(info, dict)

    def test_10_random_steps(self, env):
        env.reset(seed=7)
        rng = np.random.default_rng(7)
        for i in range(10):
            action = int(rng.integers(0, 3))
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (OBS_DIM,), f"Bad obs shape at step {i}"
            assert np.isfinite(reward),      f"Non-finite reward at step {i}: {reward}"
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "forward_return" in info
            assert "close_t" in info
            if terminated:
                break

    def test_buy_positive_return_gives_positive_reward(self, df, stats):
        """BUY action on a bar with positive forward return → positive reward."""
        H = 5
        env = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        # Find a bar where forward return is clearly positive
        for t in range(len(df) - H - 1):
            fwd = (df.iloc[t + H]["close"] - df.iloc[t]["close"]) / df.iloc[t]["close"]
            if fwd > 0.01:          # >1% positive return
                env.reset()
                env._step_idx = t
                env._prev_action = ACTION_BUY   # same action → no TC
                _, reward, _, _, _ = env.step(ACTION_BUY)
                assert reward > 0, f"Expected positive reward, got {reward:.6f} (fwd={fwd:.4f})"
                return
        pytest.skip("No clearly positive forward return found in test fixture data")

    def test_sell_negative_return_gives_positive_reward(self, df, stats):
        """SELL action on a bar with negative forward return → positive reward."""
        H = 5
        env = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        for t in range(len(df) - H - 1):
            fwd = (df.iloc[t + H]["close"] - df.iloc[t]["close"]) / df.iloc[t]["close"]
            if fwd < -0.01:
                env.reset()
                env._step_idx = t
                env._prev_action = ACTION_SELL  # no TC
                _, reward, _, _, _ = env.step(ACTION_SELL)
                assert reward > 0, f"Expected positive reward, got {reward:.6f} (fwd={fwd:.4f})"
                return
        pytest.skip("No clearly negative forward return in test fixture data")

    def test_hold_reward_is_zero_regardless_of_return(self, df, stats):
        """HOLD always returns 0 (minus possible TC if switching from non-HOLD)."""
        H = 5
        env = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        env.reset()
        env._prev_action = ACTION_HOLD
        _, reward, _, _, _ = env.step(ACTION_HOLD)
        assert reward == 0.0, f"HOLD reward should be 0.0, got {reward}"

    def test_transaction_cost_on_switch(self, df, stats):
        """Switching from BUY to SELL incurs transaction cost."""
        H = 5
        env_tc = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        env_no = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        env_tc.reset(); env_tc._prev_action = ACTION_BUY  # previous was BUY
        env_no.reset(); env_no._prev_action = ACTION_SELL  # same action

        _, r_with_tc, _, _, _ = env_tc.step(ACTION_SELL)
        _, r_no_tc,   _, _, _ = env_no.step(ACTION_SELL)
        assert abs(r_no_tc - r_with_tc - 0.0005) < 1e-9, (
            f"TC mismatch: r_with_tc={r_with_tc:.8f}, r_no_tc={r_no_tc:.8f}"
        )

    def test_episode_terminates_correctly(self, df, stats):
        """Episode must end exactly when t > max_start (no early or late termination)."""
        H = 5
        env = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(ACTION_HOLD)
            steps += 1
        expected_steps = env._max_start + 1
        assert steps == expected_steps, (
            f"Episode ran {steps} steps, expected {expected_steps}"
        )

    def test_random_start_varies(self, df, stats):
        """random_start=True should produce different start indices across resets."""
        env = TradingEnv(df, obs_stats=stats, holding_period=5, random_start=True)
        starts = set()
        for seed in range(20):
            env.reset(seed=seed)
            starts.add(env._step_idx)
        assert len(starts) > 1, "random_start=True always returned the same start index"

    def test_no_lookahead_in_observation(self, df, stats):
        """
        Confirm that two identical bars at step t produce identical observations
        regardless of what comes AFTER them — the obs must be a pure function
        of df.iloc[t] only.
        """
        H = 5
        env = TradingEnv(df, obs_stats=stats, holding_period=H, random_start=False)
        env.reset(seed=0)
        env._step_idx = 10
        obs_a = env._get_obs()

        # Corrupt bars 11 onwards — obs must not change
        df_copy = df.copy()
        df_copy.loc[11:, "close"] *= 99
        env2 = TradingEnv(df_copy, obs_stats=stats, holding_period=H, random_start=False)
        env2.reset(seed=0)
        env2._step_idx = 10
        obs_b = env2._get_obs()

        np.testing.assert_array_almost_equal(
            obs_a, obs_b,
            decimal=6,
            err_msg="Observation at t=10 changed when future data was modified — lookahead detected!",
        )


# ---------------------------------------------------------------------------
# Standalone runner (prints output even when not using pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("tests/test_rl_env.py — standalone runner")
    print("=" * 60)

    df_ = _make_df(300)
    stats_ = compute_obs_stats(df_)
    env_  = TradingEnv(df_, obs_stats=stats_, holding_period=5, random_start=False)
    print(f"DataFrame: {len(df_)} rows | Stats: {stats_}")
    print(f"Obs space: {env_.observation_space}")
    print(f"Act space: {env_.action_space}")

    # Run 10 random steps and report
    obs_, _ = env_.reset(seed=0)
    rng_ = np.random.default_rng(0)
    print("\n10 random steps:")
    for i in range(10):
        a = int(rng_.integers(0, 3))
        obs_, r, done, _, info_ = env_.step(a)
        name = {0: "SELL", 1: "HOLD", 2: "BUY"}[a]
        print(
            f"  [{i}] {name:4s} | close_t={info_['close_t']:7.2f} "
            f"close_t+H={info_['close_t_H']:7.2f} "
            f"fwd={info_['forward_return']:+.4f} "
            f"r={r:+.6f} obs={obs_.shape} finite={np.all(np.isfinite(obs_))}"
        )
        if done:
            break

    # Run all pytest tests programmatically
    print("\nRunning pytest tests...")
    import pytest as _pytest
    exit_code = _pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)

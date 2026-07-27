# Changelog

All notable changes to NEXUS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-07-27

### Added
- **RL Quant Model** — 5th agent: a PPO reinforcement-learning policy
  trained on 4+ years of historical OHLCV data across all five tickers
  (`src/rl_env.py`, `src/rl_agent.py`)
- **Pure-numpy inference** — `rl_agent.py` loads a 21 KB `.npz` weight
  file and runs a numpy-only forward pass; no `torch` or `stable-baselines3`
  at runtime (zero Vercel bundle size increase)
- `scripts/train_rl_agent.py` — PPO training with two-window out-of-sample
  backtest (mark-to-market daily returns, Sharpe ratio, max drawdown)
- `scripts/export_rl_weights.py` — extracts PPO MLP weights to `.npz`,
  validates numpy vs torch to <1e-5 across 100 observations
- `scripts/window_backtest.py` — standalone policy vs buy-and-hold table
- `tests/test_rl_env.py` — 16 tests for the Gymnasium environment
- `requirements-training.txt` — training-only deps (sb3, torch, gymnasium)
  separated from the lean production `requirements.txt`
- `models/rl_policy_weights.npz` + `models/rl_obs_stats.json` — committed
  inference artefacts (21 KB total)
- `QUANT MODEL` badge on the 5th AgentCard in the frontend
- `IconRL` — CPU-chip SVG icon for the Quant Model card
- `Makefile` — unified dev command interface
- `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`
- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- Issue & PR templates (`.github/`)

### Changed
- Agent weights rebalanced: `technical` 0.20, `fundamental` 0.225,
  `sentiment` 0.225, `risk_manager` 0.20, `rl_trader` 0.15
- Coordinator prompt updated: 5-agent synthesis rules, RL framed as
  defensive/risk-hedging signal, position_size arithmetic-expression fix,
  3-attempt retry loop with exponential backoff
- `api/index.py` updated with RL singleton wiring (Vercel parity with
  `api_server.py`)
- `rl_analysis` added to `AnalysisResults` TypeScript interface
- `App.tsx` updated: 5 agents in `INITIAL_AGENTS`, animation sequence,
  `agentKeys`, and result rendering
- `DecisionPanel.tsx` fallback detection fixed — was incorrectly triggering
  on the word "unavailable" in a successful Coordinator response; now checks
  `confidence === 0` and exact fallback string prefixes

### Fixed
- `isDecisionFallback()` false-positive: Coordinator's success message
  *"No agents reported 'unavailable'"* contained the substring `unavailable`,
  causing the Final Decision panel to show "NO DECISION" on clean runs
- Coordinator JSON validation failure: Groq was generating arithmetic
  expressions (`0.117 * 0.25`) for `position_size` instead of decimal
  literals; fixed with explicit prompt instruction

---

## [1.0.0] — 2026-07 (initial release)

### Added
- Four parallel LLM agents: Technical, Fundamental, Sentiment, Risk
- Coordinator Agent (LLaMA 3.3 70B via Groq) synthesising a final decision
  with entry price, stop-loss, take-profit, and position size
- `src/data_handler.py` — yfinance OHLCV, technical indicators (RSI-14
  Wilder, MACD, SMA, Bollinger Bands, ATR-14, HV-14, drawdown), Exa news
- `src/multi_agent_system.py` — `asyncio.gather()` parallel agent execution,
  pre-computed weighted signal, hard Coordinator rules
- React 19 + TypeScript + Vite frontend with Framer Motion animations
- Real-time watchlist via Finnhub WebSocket (falls back to yfinance polling)
- `api_server.py` (local) + `api/index.py` (Vercel serverless) entry points
- Vercel deployment at `nexus-multi-agent-trading-analyst.vercel.app`
- 29 tests across `test_data_handler.py` and `test_multi_agent_system.py`

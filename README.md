# Nexus — Multi-Agent Trading Analyst

Five specialised models — **Technical**, **Fundamental**, **Sentiment**,
**Risk** (all LLM-powered), and a **Quant Model** (trained PPO
Reinforcement-Learning policy) — analyse real market data in parallel and
synthesise a final trading recommendation with an entry price, stop-loss,
take-profit, and position size.

![python](https://img.shields.io/badge/python-3.10+-3776ab)
![frontend](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61dafb)
![llm](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-F55036)
![rl](https://img.shields.io/badge/RL-PPO%20%28Stable--Baselines3%29-orange)
![data](https://img.shields.io/badge/market%20data-yfinance-000000)
![deploy](https://img.shields.io/badge/deploy-Vercel-000000)

### 🔴 Live demo

| | URL |
| --- | --- |
| **App** | **https://nexus-multi-agent-trading-analyst.vercel.app** |

---

## What's new (v2 — RL Trader)

The original four-agent LLM system has been extended with a **fifth agent:
the Quant Model**, a PPO reinforcement-learning policy trained on 4+ years
of historical OHLCV data across all five supported tickers.

| | v1 | v2 |
|---|---|---|
| Agents | 4 × LLM | 4 × LLM + 1 × RL policy |
| RL inference | — | Pure numpy (<1 ms, no torch at runtime) |
| Coordinator | Equal 0.25 weights | Tuned weights (see below) |
| Vercel bundle | ~15 MB | ~15 MB (RL adds 0 MB — numpy already required) |

**Key design decision:** the RL policy weights are exported to a 21 KB
`.npz` file (`models/rl_policy_weights.npz`). Inference uses a pure-numpy
forward pass — `torch` and `stable-baselines3` are **not imported at
runtime**, keeping the Vercel serverless bundle lean. The trained
`rl_policy_ppo.zip` is only needed to retrain or re-export, and is
excluded from git (see `.gitignore`).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     React Frontend                           │
│  (Vite · TypeScript · Framer Motion)   localhost:5174        │
└────────────────────────┬─────────────────────────────────────┘
                         │  HTTP / Vite proxy
┌────────────────────────▼─────────────────────────────────────┐
│           Python API Server  (api_server.py / api/index.py)  │
│   GET  /api/health        — liveness check                   │
│   GET  /api/watchlist     — live quotes (5 tickers)          │
│   POST /api/analyze       — full multi-agent run             │
└──────┬────────────────────────────────────┬──────────────────┘
       │                                    │
┌──────▼──────┐                   ┌─────────▼──────────────────┐
│  yfinance   │                   │  AdvancedMultiAgentSystem   │
│  Exa API    │                   │                             │
│  (OHLCV,    │                   │  ┌─ asyncio.gather() ─────┐ │
│  news,      │──────────────────►│  │ Technical  (LLM) 0.20 │ │
│  fundament.)│                   │  │ Fundamental(LLM) 0.225│ │
└─────────────┘                   │  │ Sentiment  (LLM) 0.225│ │
                                  │  │ Risk Mgr   (LLM) 0.20 │ │
                                  │  └───────────────────────┘ │
                                  │                             │
                                  │  ┌─ synchronous (pre-gather)┐│
                                  │  │ Quant Model (RL)  0.15 ││
                                  │  │ (numpy, <1 ms)         ││
                                  │  └────────────────────────┘│
                                  │                             │
                                  │  Coordinator LLM            │
                                  │  (synthesises final call)   │
                                  └─────────────────────────────┘
```

The RL agent runs **synchronously before** `asyncio.gather()` — its <1 ms
numpy forward pass completes before the event loop starts, so it adds zero
wall-clock latency to the total request time (~4 s end-to-end).

On Vercel, `api/index.py` is the serverless entry point instead of
`api_server.py` — identical agent wiring, just wrapped in a
`BaseHTTPRequestHandler` subclass Vercel's Python runtime can invoke.

---

## Agent weights & Coordinator rules

| Agent | Type | Weight | Notes |
|---|---|---|---|
| Technical Analyst | LLM (LLaMA 3.3 70B) | **0.20** | RSI, MACD, Bollinger, SMA crossovers |
| Fundamental Analyst | LLM (LLaMA 3.3 70B) | **0.225** | P/E, EPS, sector, analyst targets |
| Sentiment Analyst | LLM (LLaMA 3.3 70B) | **0.225** | News, social signals |
| Risk Manager | LLM (LLaMA 3.3 70B) | **0.20** | HV-14, ATR, drawdown, position sizing |
| **Quant Model** | **PPO RL (numpy)** | **0.15** | Defensive/risk-hedging — see below |

The RL agent's lower weight (0.15) reflects two-window backtesting findings:
it is **stronger at avoiding sustained downtrends** (avoided −31% TSLA and
−26% MSFT drawdowns in out-of-sample testing) and **weaker at capturing
strong sustained uptrends** (tied or trailed buy-and-hold during the 2023
tech rally). The Coordinator is instructed to reduce its influence when
fundamental and sentiment agents are strongly bullish.

---

## Technical indicators & fundamentals

| Indicator | Formula | Used by |
|---|---|---|
| **RSI-14** | Wilder's exponential smoothing (`ewm(alpha=1/14)`) | Technical, RL |
| **MACD** (line + signal + histogram) | 12/26 EMA diff, 9-period signal | Technical, RL |
| **SMA-20 / SMA-50** | Simple rolling means | Technical, RL |
| **Bollinger Bands** (20-period, ±2σ) | position 0–1 (lower→upper) | Technical, RL |
| **Volume ratio** | today vs 20-day mean | Technical, RL |
| **Momentum** | 10-day percent change | Technical, RL |
| **HV-14** | 14-day std of daily log returns × √252 | Technical, Risk, RL |
| **ATR-14** | Wilder-smoothed True Range | Risk, Coordinator (stops), RL |
| **Max drawdown** | % from running cummax | Risk, RL |
| **Fundamentals** | `marketCap`, `trailingPE`, `forwardPE`, `EPS`, `dividendYield`, `sector`, `beta`, 52w high/low, analyst targets | Fundamental |
| **Multi-day closes** | `close_prev_1` / `close_prev_5` / `close_prev_20` | Technical, RL |

---

## Data sources

| Data | Source | API key |
|---|---|---|
| Price history (OHLCV) | Yahoo Finance via `yfinance` (400 days) | No — free |
| Live watchlist quotes | Yahoo Finance `fast_info` | No — free |
| Fundamentals | Yahoo Finance `.info` | No — free |
| News (primary) | Exa neural search | Optional |
| News (fallback) | Yahoo Finance headlines | No — free |
| LLM reasoning | Groq (LLaMA 3.3 70B) | Yes — required |
| RL training data | Yahoo Finance OHLCV (same `DataHandler`) | No — free |

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A free [Groq API key](https://console.groq.com)
- (Optional) An [Exa API key](https://exa.ai) for richer news

### 2. Clone & configure

```bash
git clone https://github.com/bhavuk1409/multi-agent-trading-analyst.git
cd multi-agent-trading-analyst

cp .env.example .env
# open .env and set GROQ_API_KEY (and optionally EXA_API_KEY)
```

### 3. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### 5. Run

```bash
./start.sh
```

Then open [http://localhost:5174](http://localhost:5174).

**Manual start**, if you'd rather run each piece yourself:

```bash
# Terminal 1
.venv/bin/python api_server.py

# Terminal 2
cd frontend && npm run dev
```

> **Note:** The RL model weights (`models/rl_policy_weights.npz`) are
> committed to the repo and loaded automatically at startup. No extra step
> needed to use the Quant Model agent.

---

## RL agent — training & retraining

The Quant Model ships pre-trained. To retrain from scratch (e.g. after
extending the ticker list or tuning hyperparameters):

### 1. Install training-only dependencies

```bash
pip install -r requirements-training.txt
```

> Training deps (`stable-baselines3`, `torch`, `gymnasium`) are intentionally
> separated from `requirements.txt` — they are **not** installed in the
> production/Vercel environment.

### 2. Train

```bash
python scripts/train_rl_agent.py
# Trains a PPO policy on all 5 tickers (2020-01-01 → split date)
# Saves: models/rl_policy_ppo.zip   (torch format, ~150 KB)
```

### 3. Export weights for inference

```bash
python scripts/export_rl_weights.py
# Extracts the MLP weights to a 21 KB numpy file
# Saves: models/rl_policy_weights.npz  ← this is what the API loads
```

### 4. Validate (sanity check)

The export script runs 100 forward passes and confirms the numpy output
matches the torch output to <1e-5 absolute error. You can also run the
full two-window backtest to compare policy vs buy-and-hold:

```bash
python scripts/window_backtest.py
```

---

## Project structure

```
├── api_server.py              # Local Python HTTP API server
├── api/
│   └── index.py               # Vercel serverless entry point (same logic)
├── start.sh                   # One-command local startup
├── requirements.txt           # Prod Python deps (lean, Vercel-friendly, no torch)
├── requirements-training.txt  # Training-only deps (sb3, torch, gymnasium)
├── .env.example               # Environment variable template
│
├── config/
│   └── config.yaml            # Tickers, LLM model/params, agent weights
│
├── src/
│   ├── data_handler.py        # yfinance + Exa fetching, indicators, fundamentals
│   ├── multi_agent_system.py  # 5-agent pipeline + Coordinator (Groq via openai SDK)
│   ├── rl_env.py              # Gymnasium TradingEnv (training environment)
│   └── rl_agent.py            # Pure-numpy RL inference wrapper (no torch at runtime)
│
├── models/
│   ├── rl_policy_weights.npz  # 21 KB — numpy MLP weights (committed, used at runtime)
│   ├── rl_obs_stats.json      # Observation normalisation stats
│   └── rl_policy_ppo.zip      # Torch PPO checkpoint (gitignored, training only)
│
├── scripts/
│   ├── train_rl_agent.py      # PPO training + out-of-sample backtest
│   ├── export_rl_weights.py   # Extract torch weights → numpy .npz
│   └── window_backtest.py     # Two-window policy vs buy-and-hold comparison
│
├── tests/
│   ├── test_data_handler.py        # 24 tests — indicators, data shape, news sources
│   ├── test_multi_agent_system.py  # 5 integration tests (real Groq API)
│   └── test_rl_env.py              # 16 tests — Gym env, obs shape, reward, step logic
│
└── frontend/
    ├── src/
    │   ├── App.tsx              # Main application — 5-agent state, animation, results
    │   ├── types.ts             # TypeScript interfaces (incl. rl_analysis)
    │   └── components/
    │       ├── AgentCard.tsx    # Agent result cards (AI ANALYST / QUANT MODEL badge)
    │       ├── DecisionPanel.tsx
    │       ├── MarketReadout.tsx
    │       ├── NewsFeed.tsx
    │       ├── Icons.tsx        # SVG icons incl. IconRL (CPU chip)
    │       └── ...
    ├── package.json
    └── vite.config.ts
```

---

## Configuration

Edit `config/config.yaml` to change the tracked tickers, LLM model, or
agent weights. All five weights must sum to 1.0:

```yaml
llm:
  model: "llama-3.3-70b-versatile"   # or llama3-70b-8192, gemma2-9b-it
  temperature: 0.7

agents:
  technical_analyst:   { enabled: true, weight: 0.20  }
  fundamental_analyst: { enabled: true, weight: 0.225 }
  sentiment_analyst:   { enabled: true, weight: 0.225 }
  risk_manager:        { enabled: true, weight: 0.20  }
  rl_trader:           { enabled: true, weight: 0.15  }
```

---

## API reference

### `GET /api/health`
```json
{ "status": "ok", "tickers": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"] }
```

### `GET /api/watchlist`
Live quotes for all 5 tickers from Yahoo Finance.

### `POST /api/analyze`
**Body:** `{ "ticker": "AAPL" }`

**Response (trimmed):**
```json
{
  "ticker": "AAPL",
  "date": "2026-07-24",
  "market_data": { "close": 333.02, "rsi": 65.2, "macd_hist": 0.91, "..." },
  "price_history": [ { "date": "2025-09-02", "close": 229.07 }, "..." ],
  "news": [ { "title": "...", "source": "Reuters", "sentiment": "neutral" } ],

  "technical_analysis":   { "recommendation": "sell", "confidence": 70, "reasoning": "..." },
  "fundamental_analysis": { "recommendation": "sell", "confidence": 80, "reasoning": "..." },
  "sentiment_analysis":   { "recommendation": "buy",  "confidence": 80, "reasoning": "..." },
  "risk_analysis":        { "recommendation": "hold", "confidence": 60, "reasoning": "..." },
  "rl_analysis":          { "recommendation": "buy",  "confidence": 78, "reasoning": "RL quant model (PPO·numpy) signals BUY..." },

  "final_decision": {
    "action": "hold",
    "confidence": 40,
    "conviction": "low",
    "position_size": 0.06,
    "entry_price": 333.02,
    "stop_loss_price": 316.49,
    "take_profit_price": 354.13,
    "time_horizon": "short-term",
    "reasoning": "Given the mixed signals from the 5 agents..."
  }
}
```

The Coordinator receives a pre-computed weighted signal (range [−1, +1])
as a starting point and applies hard rules: stop = entry − max(2 × ATR-14,
5% of entry); take-profit = stop-distance × ≥ 1.5 risk:reward; position
size = Kelly-lite (confidence/100 × 0.25), clamped to [0.02, 0.30].

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

Three test files, 45 tests total:

| File | Tests | Covers |
|---|---|---|
| `test_data_handler.py` | 24 | yfinance fetching, indicator correctness (Wilder RSI, MACD, ATR, HV-14, drawdown), news authenticity |
| `test_multi_agent_system.py` | 5 | Real Groq integration — shape, parallelism (<15 s), weight normalisation |
| `test_rl_env.py` | 16 | Gym env — obs shape, action space, step/reset logic, reward sign, episode termination |

> `test_multi_agent_system.py` is skipped automatically if `GROQ_API_KEY`
> is not set. `test_rl_env.py` runs fully offline (no API key needed).

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + TypeScript + Vite + Framer Motion |
| Backend (local) | Python 3 stdlib `http.server` (no framework) |
| Backend (Vercel) | `api/index.py` — same handler, serverless |
| LLM | Groq API (LLaMA 3.3 70B) via `openai` SDK — no LangChain |
| RL training | Stable-Baselines3 PPO + Gymnasium (training only) |
| RL inference | Pure numpy (21 KB weights, no torch at runtime) |
| Market data | yfinance (Yahoo Finance) |
| News search | Exa API (optional, falls back to yfinance headlines) |

---

## Supported tickers

| Ticker | Company |
|---|---|
| AAPL | Apple Inc. |
| GOOGL | Alphabet Inc. |
| MSFT | Microsoft Corp. |
| TSLA | Tesla, Inc. |
| NVDA | NVIDIA Corp. |

---

## License

MIT

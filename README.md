# Nexus — Multi-Agent LLM Trading Analyst

Four specialised AI agents — **Technical**, **Fundamental**, **Sentiment**,
and **Risk** — analyse real market data in parallel and synthesise a final
trading recommendation with an entry price, stop loss, take-profit, and
position size.

![python](https://img.shields.io/badge/python-3.10-3776ab) ![frontend](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61dafb) ![backend](https://img.shields.io/badge/backend-stdlib%20http.server-3776ab) ![llm](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-F55036) ![data](https://img.shields.io/badge/market%20data-yfinance-000000) ![deploy](https://img.shields.io/badge/deploy-Vercel-000000)

### 🔴 Live demo

| | URL |
| --- | --- |
| **App** | **https://nexus-multi-agent-trading-analyst.vercel.app** |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  (Vite · TypeScript · Framer Motion)                │
│   localhost:5174                                     │
└──────────────────────┬──────────────────────────────┘
                       │  HTTP / Vite proxy
┌──────────────────────▼──────────────────────────────┐
│              Python API Server (api_server.py)       │
│   GET  /api/health       — liveness check            │
│   GET  /api/watchlist    — live quotes (5 tickers)  │
│   POST /api/analyze      — full multi-agent run      │
└──────┬──────────────────────────────┬───────────────┘
       │                              │
┌──────▼──────┐              ┌────────▼────────────┐
│  yfinance   │              │  AdvancedMultiAgent  │
│  (Yahoo     │              │  System              │
│  Finance)   │              │  (Groq, direct       │
│  OHLCV data │              │   OpenAI-SDK calls)  │
│  No key req.│              │                      │
└─────────────┘              │  • Technical Agent   │
                             │  • Fundamental Agent │
┌─────────────┐              │  • Sentiment Agent   │
│  Exa API    │              │  • Risk Manager      │
│  (optional) │──────────────►  (LLaMA 3.3 70B)    │
│  Rich news  │              └─────────────────────-┘
└─────────────┘
```

On Vercel, `api/index.py` is the serverless entry point instead of
`api_server.py` — same agent system and data handler underneath, just
wrapped in a `BaseHTTPRequestHandler` subclass Vercel's Python runtime can
invoke, with the agent system cached across warm invocations.

**Worth knowing:** the multi-agent system talks to Groq directly through the
lightweight `openai` SDK (pointed at Groq's OpenAI-compatible endpoint)
rather than through LangChain — the code comments explicitly frame this as
cutting package weight for the Vercel serverless bundle.

---

## Technical indicators & fundamentals

The full 400-day price history is fetched once per analyze call. The
`DataHandler` then computes the following in-place and ships them to the
agents:

| Indicator | Formula | Used by |
|---|---|---|
| **RSI-14** | Wilder's exponential smoothing (`ewm(alpha=1/14)`) — matches TradingView / Bloomberg | Technical |
| **MACD** (line + 9-period signal + histogram) | 12/26 EMA diff, 9-period EMA signal, `macd − signal` | Technical |
| **SMA-20 / SMA-50** | Simple rolling means | Technical |
| **Bollinger Bands** (20-period, ±2σ) | position 0–1 (lower→upper) | Technical |
| **Volume ratio** | today vs 20-day mean | Technical |
| **Momentum** | 10-day percent change | Technical |
| **HV-14** | 14-day std of daily log returns, × √252, in % | Technical + Risk |
| **ATR-14** | Wilder-smoothed True Range | Risk + Coordinator (stops) |
| **Max drawdown** | `% from running cummax` across full history | Risk |
| **Fundamentals** | `marketCap`, `trailingPE`, `forwardPE`, `EPS`, `dividendYield`, `sector`, `industry`, `beta`, `52w high/low`, `shortPercentOfFloat`, analyst targets | Fundamental |
| **Multi-day closes** | `close_prev_1` / `close_prev_5` / `close_prev_20` | Technical (crossovers) |

The historical price series is the full 400-day window — the frontend
slices it client-side based on the user's selected chart period (30D / 60D
/ 3M / 6M / 1Y).

---

## Data sources

| Data | Source | API key needed |
|---|---|---|
| Price history (OHLCV) | Yahoo Finance via `yfinance` (400 calendar days) | No — free |
| Live watchlist quotes | Yahoo Finance via `yfinance` (fast_info) | No — free |
| Fundamentals (P/E, EPS, sector, beta, 52w, target prices, …) | Yahoo Finance `.info` | No — free |
| News (primary) | Exa neural search | Optional |
| News (fallback) | Yahoo Finance headlines | No — free |
| LLM reasoning | Groq (LLaMA 3.3 70B) | Yes — required |

`data_handler.py` is explicit that it never fabricates data: if `yfinance`
returns nothing for a ticker, it raises a `ValueError` instead of silently
returning made-up numbers. Fundamentals lookups degrade gracefully — a missing
field is reported as `null`/`N/A` rather than invented.

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
pip install -r requirements.txt -c constraints.txt
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

---

## Project structure

```
├── api_server.py          # Local Python HTTP API server
├── api/index.py           # Vercel serverless entry point (same logic, packaged for prod)
├── start.sh               # One-command local startup script
├── requirements.txt       # Python dependencies (lean, Vercel-friendly)
├── constraints.txt        # pip resolver constraints
├── .env.example           # Environment variable template
├── config/
│   └── config.yaml        # Tickers, LLM model/params, agent weights
├── src/
│   ├── data_handler.py    # yfinance + Exa data fetching, technical indicators, fundamentals
│   └── multi_agent_system.py   # Direct Groq (openai SDK) agent calls + coordinator
├── tests/
│   ├── test_data_handler.py         # Indicator correctness + data-shape (24 tests)
│   └── test_multi_agent_system.py   # Integration tests against real Groq (5 tests)
├── vercel.json            # Vercel build + rewrites config
└── frontend/              # React + TypeScript UI
    ├── src/
    │   ├── App.tsx         # Main application
    │   ├── api.ts          # Backend API client
    │   ├── types.ts        # TypeScript interfaces
    │   └── components/
    │       ├── AgentCard.tsx
    │       ├── DecisionPanel.tsx
    │       ├── MarketReadout.tsx
    │       ├── NewsFeed.tsx
    │       ├── NeuralOrb.tsx / AgentNeuralNet.tsx
    │       ├── CompanySelect.tsx
    │       └── Header.tsx
    ├── package.json
    └── vite.config.ts
```

---

## Configuration

Edit `config/config.yaml` to change the tracked tickers, LLM model, or agent
weights (the four agent weights must sum to 1.0):

```yaml
llm:
  model: "llama-3.3-70b-versatile"   # or llama3-70b-8192, gemma2-9b-it
  temperature: 0.7

agents:
  technical_analyst:   { enabled: true, weight: 0.25 }
  fundamental_analyst: { enabled: true, weight: 0.25 }
  sentiment_analyst:   { enabled: true, weight: 0.25 }
  risk_manager:        { enabled: true, weight: 0.25 }
```

---

## API reference

### `GET /api/health`
```json
{ "status": "ok", "tickers": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"] }
```

### `GET /api/watchlist`
Live quotes for all 5 tickers from Yahoo Finance.
```json
{
  "quotes": [
    { "ticker": "AAPL", "price": 333.02, "change": 1.45, "change_pct": 0.44, "is_positive": true }
  ]
}
```

### `POST /api/analyze`
**Body:** `{ "ticker": "AAPL" }`

**Response (trimmed):**
```json
{
  "ticker": "AAPL",
  "date": "2026-07-26",
  "market_data": {
    "close": 333.02, "volume": 50_000_000,
    "rsi": 67.7, "macd": 2.1, "macd_signal": 1.8, "macd_hist": 0.3,
    "sma_20": 328.5, "sma_50": 317.4,
    "bb_position": 0.62, "volume_ratio": 1.15, "momentum": 4.20,
    "hv_14": 22.4, "atr_14": 8.27, "drawdown": -3.4,
    "close_prev_1": 331.6, "close_prev_5": 326.1, "close_prev_20": 314.8,
    "market_cap": 5_240_000_000_000, "pe_trailing": 28.4, "pe_forward": 25.1,
    "eps_trailing": 6.42, "eps_forward": 7.21, "dividend_yield": 0.45,
    "sector": "Technology", "industry": "Consumer Electronics",
    "52w_high": 339.4, "52w_low": 219.6, "beta": 1.24,
    "short_pct": 0.0078,
    "target_mean_price": 358.0, "target_high_price": 410.0, "target_low_price": 215.0
  },
  "price_history": [ { "date": "2025-09-02", "close": 229.07 }, ... ] /* ~225 trading days */,
  "news": [
    { "title": "...", "source": "Reuters", "sentiment": "neutral",
      "summary": "...", "published_date": "2026-07-25", "url": "..." }
  ],
  "technical_analysis":   { "recommendation": "buy",  "confidence": 78, "reasoning": "..." },
  "fundamental_analysis": { "recommendation": "hold", "confidence": 62, "reasoning": "..." },
  "sentiment_analysis":   { "recommendation": "buy",  "confidence": 80, "reasoning": "..." },
  "risk_analysis":        { "recommendation": "buy",  "confidence": 71, "reasoning": "..." },
  "final_decision": {
    "action": "buy",
    "confidence": 75,
    "conviction": "medium",
    "position_size": 0.18,
    "entry_price": 333.02,
    "stop_loss_price": 316.49,
    "take_profit_price": 357.11,
    "time_horizon": "medium-term",
    "reasoning": "..."
  }
}
```

The `stop_loss_price` and `take_profit_price` are anchored to the
coordinator's rule: **stop = entry − max(2 × ATR-14, 5 % of entry)**,
**take-profit = entry + risk-distance × ≥ 1.5** (so the risk:reward ratio
stays above 1.5). The LLM is sent a pre-computed weighted signal as a
starting point and is told to justify deviations from it. Falls back to a
neutral `hold` with `"degraded": true` if any agent fails.

The Vercel deployment (`api/index.py`) exposes the same three endpoints,
also reachable without the `/api` prefix (`/health`, `/watchlist`,
`/analyze`).

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

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

Two test files, 29 tests total:

- **`tests/test_data_handler.py`** — 24 tests covering yfinance data
  fetching, data-shape, news-source authenticity (no mock sources, no
  `example.com` URLs), and **8 fixture-based indicator correctness tests**
  (Wilder RSI ≠ simple-mean RSI, MACD signal + histogram, ATR-14 > 0,
  HV-14 annualised, drawdown negative on decline, market summary exposes
  all new fields).
- **`tests/test_multi_agent_system.py`** — 5 integration tests against the
  real Groq API (skipped automatically if `GROQ_API_KEY` is missing):
  `analyze()` stays sync, the result shape is unchanged, four agents run
  in parallel under 15 s, configured weights reach the coordinator prompt,
  and dirty weights get renormalised to sum to 1.0.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + TypeScript + Vite 8 |
| Animations | Framer Motion |
| Backend (local) | Python 3 stdlib `http.server` (no framework) |
| Backend (Vercel) | `api/index.py`, same handler pattern, serverless |
| LLM | Groq API (LLaMA 3.3 70B) via the `openai` SDK directly — no LangChain |
| Market data | yfinance (Yahoo Finance) |
| News search | Exa API (optional) |

---

## License

MIT

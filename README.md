# NEXUS — Multi-Agent LLM Trading Analyst

Four specialised AI agents — **Technical**, **Fundamental**, **Sentiment**,
and **Risk** — analyse real market data in parallel and synthesise a final
trading recommendation with an entry price, stop loss, take-profit, and
position size.

![python](https://img.shields.io/badge/python-3.10-3776ab) ![frontend](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61dafb) ![backend](https://img.shields.io/badge/backend-stdlib%20http.server-3776ab) ![llm](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-F55036) ![data](https://img.shields.io/badge/market%20data-yfinance-000000) ![deploy](https://img.shields.io/badge/deploy-Vercel-000000)

### 🔴 Live demo

| | URL |
| --- | --- |
| **App** | **https://multi-agent-trading-analyst.vercel.app** |

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

## Data sources

| Data | Source | API key needed |
|---|---|---|
| Price history (OHLCV) | Yahoo Finance via `yfinance` | No — free |
| Live watchlist quotes | Yahoo Finance via `yfinance` | No — free |
| News (primary) | Exa neural search | Optional |
| News (fallback) | Yahoo Finance headlines | No — free |
| LLM reasoning | Groq (LLaMA 3.3 70B) | Yes — required |

`data_handler.py` is explicit that it never fabricates data: if `yfinance`
returns nothing for a ticker, it raises a `ValueError` instead of silently
returning made-up numbers.

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
│   ├── data_handler.py    # yfinance + Exa data fetching, technical indicators
│   └── multi_agent_system.py   # Direct Groq (openai SDK) agent calls
├── tests/
│   └── test_data_handler.py
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
  "market_data": { "close": 333.02, "rsi": 67.7, "macd": 2.1 },
  "price_history": [{ "date": "2026-06-27", "close": 310.5 }],
  "news": [{ "title": "...", "source": "...", "sentiment": "neutral" }],
  "technical_analysis":   { "recommendation": "buy", "confidence": 78, "reasoning": "..." },
  "fundamental_analysis": { "recommendation": "hold", "confidence": 62, "reasoning": "..." },
  "sentiment_analysis":   { "recommendation": "buy", "confidence": 80, "reasoning": "..." },
  "risk_analysis":        { "recommendation": "buy", "confidence": 71, "reasoning": "..." },
  "final_decision": {
    "action": "buy",
    "confidence": 75,
    "conviction": "medium",
    "position_size": 0.18,
    "entry_price": 333.02,
    "stop_loss_price": 316.37,
    "take_profit_price": 383.47,
    "time_horizon": "medium-term",
    "reasoning": "..."
  }
}
```

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
pytest tests/
```

`tests/test_data_handler.py` covers the yfinance data-fetching and
technical-indicator logic in `src/data_handler.py`.

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

# NEXUS — Multi-Agent LLM Trading Analyst

> Four specialised AI agents — Technical, Fundamental, Sentiment, and Risk — analyse real market data in parallel and synthesise a final trading recommendation.

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
│  Finance)   │              │  (LangChain + Groq)  │
│  OHLCV data │              │                      │
│  No key req.│              │  • Technical Agent   │
└─────────────┘              │  • Fundamental Agent │
                             │  • Sentiment Agent   │
┌─────────────┐              │  • Risk Manager      │
│  Exa API    │              │  • Coordinator       │
│  (optional) │──────────────►  (LLaMA 3.3 70B)    │
│  Rich news  │              └─────────────────────-┘
└─────────────┘
```

## Data Sources

| Data | Source | API Key? |
|---|---|---|
| Price history (OHLCV) | Yahoo Finance via `yfinance` | ❌ Free |
| Live watchlist quotes | Yahoo Finance via `yfinance` | ❌ Free |
| News (primary) | Exa neural search | ✅ Optional |
| News (fallback) | Yahoo Finance headlines | ❌ Free |
| LLM reasoning | Groq (LLaMA 3.3 70B) | ✅ Required |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A free [Groq API key](https://console.groq.com)
- (Optional) An [Exa API key](https://exa.ai) for richer news

### 2. Clone & configure

```bash
git clone https://github.com/your-username/nexus-trading-analyst.git
cd nexus-trading-analyst

cp .env.example .env
# Open .env and set GROQ_API_KEY (and optionally EXA_API_KEY)
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

> **Manual start** (if you prefer):
> ```bash
> # Terminal 1
> .venv/bin/python api_server.py
>
> # Terminal 2
> cd frontend && npm run dev
> ```

---

## Project Structure

```
nexus-trading-analyst/
├── api_server.py          # Python HTTP API server
├── start.sh               # One-command startup script
├── requirements.txt       # Python dependencies
├── constraints.txt        # pip resolver constraints
├── .env.example           # Environment variable template
├── config/
│   └── config.yaml        # LLM model, agent weights
├── src/
│   ├── data_handler.py    # yfinance + Exa data fetching
│   └── multi_agent_system.py   # LangChain agent chains
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
    │       └── …
    ├── package.json
    └── vite.config.ts
```

---

## Configuration

Edit `config/config.yaml` to change the LLM model or agent weights:

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

## API Reference

### `GET /api/health`
```json
{ "status": "ok", "agent_system": "ready", "tickers": ["AAPL","GOOGL","MSFT","TSLA","NVDA"] }
```

### `GET /api/watchlist`
Returns live quotes for all 5 tickers from Yahoo Finance.
```json
{
  "quotes": [
    { "ticker": "AAPL", "price": 333.02, "change": 1.45, "change_pct": 0.44, "is_positive": true },
    …
  ]
}
```

### `POST /api/analyze`
**Body:** `{ "ticker": "AAPL" }`

**Response:**
```json
{
  "ticker": "AAPL",
  "date": "2026-07-26",
  "market_data": { "close": 333.02, "rsi": 67.7, "macd": 2.1, … },
  "price_history": [{ "date": "2026-06-27", "close": 310.5 }, …],
  "news": [{ "title": "…", "source": "…", "sentiment": "neutral" }, …],
  "technical_analysis":   { "recommendation": "buy", "confidence": 78, "reasoning": "…" },
  "fundamental_analysis": { "recommendation": "hold", "confidence": 62, "reasoning": "…" },
  "sentiment_analysis":   { "recommendation": "buy", "confidence": 80, "reasoning": "…" },
  "risk_analysis":        { "recommendation": "buy", "confidence": 71, "reasoning": "…" },
  "final_decision": {
    "action": "buy",
    "confidence": 75,
    "conviction": "medium",
    "position_size": 0.18,
    "entry_price": 333.02,
    "stop_loss_price": 316.37,
    "take_profit_price": 383.47,
    "time_horizon": "medium-term",
    "reasoning": "…"
  }
}
```

---

## Supported Tickers

| Ticker | Company |
|---|---|
| AAPL | Apple Inc. |
| GOOGL | Alphabet Inc. |
| MSFT | Microsoft Corp. |
| TSLA | Tesla, Inc. |
| NVDA | NVIDIA Corp. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript + Vite |
| Animations | Framer Motion |
| Backend | Python 3 stdlib `http.server` (no framework) |
| LLM | Groq API (LLaMA 3.3 70B via LangChain) |
| Market Data | yfinance (Yahoo Finance) |
| News Search | Exa API (optional) |

---

## License

MIT

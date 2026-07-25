# LLM-Enhanced Trading Analysis Platform

A multi-agent LLM trading analysis system: four specialized AI agents
(**Technical, Fundamental, Sentiment, Risk**) analyze a ticker in parallel
and synthesize a trading recommendation, served through a clean Streamlit
interface and powered by **Groq**'s fast LLM inference and **Exa**'s
real-time news search.

![python](https://img.shields.io/badge/python-3.8%2B-3776ab) ![ui](https://img.shields.io/badge/UI-Streamlit-ff4b4b) ![llm](https://img.shields.io/badge/LLM-Groq-f55036) ![orchestration](https://img.shields.io/badge/orchestration-LangChain-1c3c3c) ![news](https://img.shields.io/badge/news-Exa%20API-000000)

### 🔴 Live demo

**https://llm-trading-platform.streamlit.app** — pick a ticker, click **Run Analysis**, get instant multi-agent trading insights.

---

## What it does

- **Multi-agent analysis** — four specialized agents each score a different
  dimension of a trading decision, then a coordinator agent synthesizes
  them into a final call:
  - **Technical Analyst** — chart patterns, RSI/MACD/Bollinger Bands,
    moving averages, momentum, volume.
  - **Fundamental Analyst** — valuation, market conditions, financial
    metrics, industry trends.
  - **Sentiment Analyst** — real-time news processing, market sentiment,
    social signals, news impact.
  - **Risk Manager** — portfolio risk, position sizing, stop-loss /
    take-profit levels, risk/reward.
- **Fast inference** — Groq API powers rapid LLM responses across all four
  agents.
- **Real-time news** — Exa API feeds live news into the sentiment agent.
- **Structured outputs** — agents return Pydantic-validated structured data,
  not free text, for consistent downstream synthesis.
- **JSON export** — download the full analysis, timestamped, for later use.
- **Optional RL training** — a Gymnasium-compatible trading environment and
  `stable-baselines3` training script for reinforcement-learning experiments
  on top of (or instead of) the LLM agents.

---

## Architecture

| Component | File | Role |
| --- | --- | --- |
| Multi-agent system | `src/multi_agent_system.py` | Groq-powered LLM agents, structured Pydantic outputs, coordinator agent, configurable weights |
| Data handler | `src/data_handler.py` | Market data fetching/preprocessing, technical indicators, Exa news aggregation, synthetic data for testing |
| Trading environment | `src/trading_env.py` | Gymnasium-compatible env, continuous position sizing, transaction costs, portfolio tracking |
| Web app | `app.py` | Streamlit UI, modular render functions, JSON export |

---

## Repository layout

```
├── app.py                    # Main Streamlit application
├── config/
│   └── config.yaml           # System configuration
├── src/
│   ├── data_handler.py       # Data fetching and processing
│   ├── multi_agent_system.py # LLM multi-agent logic
│   └── trading_env.py        # RL trading environment
├── scripts/
│   └── train.py              # RL / LLM-enhanced training scripts
├── api/                      # API layer
├── requirements.txt          # Python dependencies
├── constraints.txt           # Version constraints
├── test_exa.py                # Exa API smoke test
└── .env.example               # API key template
```

---

## Quickstart

### 1. Clone & set up

```bash
git clone https://github.com/bhavuk1409/llm-trading.git
cd llm-trading

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt -c constraints.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
EXA_API_KEY=your_exa_api_key_here
```

- **Groq API** — free key at [console.groq.com](https://console.groq.com); fast inference, multiple models (LLaMA 3.3, Mixtral, etc.)
- **Exa API** — key at [exa.ai](https://exa.ai); real-time, AI-powered news search

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Configuration

Edit `config/config.yaml` to customize tickers, trading parameters, the LLM
model, and per-agent weights:

```yaml
data:
  tickers: ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
  start_date: "2022-01-01"
  end_date: "2024-01-01"

trading:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  max_position: 0.3

llm:
  model: "llama-3.3-70b-versatile"
  temperature: 0.7
  max_tokens: 2000

agents:
  technical_analyst:
    enabled: true
    weight: 0.25
  fundamental_analyst:
    enabled: true
    weight: 0.25
  sentiment_analyst:
    enabled: true
    weight: 0.25
  risk_manager:
    enabled: true
    weight: 0.25
```

---

## Usage

**Web interface:**

1. Open the app (live or local)
2. Select a ticker
3. Click **Run Analysis**
4. Review Market Data, Technical Indicators, Recent News, per-agent Analysis
   tabs, and the Final Trading Decision
5. Export results as JSON

**Command-line RL training** (optional, advanced):

```bash
# RL only
python scripts/train.py --mode rl

# LLM-enhanced RL
python scripts/train.py --mode llm --timesteps 50000
```

---

## Technical stack

| Category | Tools |
| --- | --- |
| Core | Python 3.8+, Streamlit, LangChain |
| LLM / search | Groq API, Exa API |
| Data | Pandas, NumPy, pandas-ta, PyYAML |
| RL (optional) | PyTorch, stable-baselines3, Gymnasium |

---

## Testing

```bash
# Syntax validation
python -m py_compile app.py

# Import check
python -c "from src.multi_agent_system import AdvancedMultiAgentSystem"
```

---

## Deployment

Deployed on Streamlit Cloud with automatic updates from `main` and
environment variables configured as Streamlit secrets. To deploy your own:

1. Fork this repository
2. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your fork
4. Add `GROQ_API_KEY` / `EXA_API_KEY` as secrets
5. Deploy

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and test thoroughly
4. Submit a pull request

---

## Acknowledgments

Groq (LLM inference), Exa (news search), LangChain (orchestration),
Streamlit (web framework), stable-baselines3 (RL algorithms).

---

## Support

- Issues: [GitHub Issues](https://github.com/bhavuk1409/llm-trading/issues)
- Live demo: https://llm-trading-platform.streamlit.app

---

## License

Provided as-is for educational and research purposes.

---

**Made with ❤️ using Groq, Exa, and Streamlit**

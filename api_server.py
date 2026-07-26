"""
NEXUS — Multi-Agent Trading Analyst: API Server
================================================
Lightweight HTTP server exposing:
  GET  /api/health      — liveness check
  GET  /api/watchlist   — live quotes for all supported tickers
  POST /api/analyze     — full multi-agent analysis for a single ticker

The AdvancedMultiAgentSystem is initialised once at startup (not per-request)
so LangChain/Groq sessions are reused efficiently.
"""

import http.server
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: logging + env + src path
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("api_server")

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import yaml
    from data_handler import DataHandler
    from multi_agent_system import AdvancedMultiAgentSystem
    try:
        from data_handler import iter_quote_stream
    except Exception as _exc:
        logger.warning(f"iter_quote_stream unavailable: {_exc}")
        iter_quote_stream = None
except ImportError as exc:
    logger.error(f"Import error: {exc}")
    logger.error("Activate the virtual environment and run: pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

# How many calendar days of history to fetch for technical indicator calculation.
# 400 calendar days ≈ 260 trading days — enough for the 1Y chart selector with
# a few days of slack for holidays.
HISTORY_DAYS = 400


def load_config(config_path: str = "config/config.yaml") -> dict:
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {
            "llm":    {"model": "llama-3.3-70b-versatile", "temperature": 0.7},
            "agents": {},
        }


# ---------------------------------------------------------------------------
# Singleton agent system — initialised once at server startup
# ---------------------------------------------------------------------------
_config = load_config()

try:
    _agent_system = AdvancedMultiAgentSystem(
        model=_config.get("llm", {}).get("model", "llama-3.3-70b-versatile"),
        temperature=_config.get("llm", {}).get("temperature", 0.7),
        agent_config=_config.get("agents", {}),
    )
    logger.info("✓ AdvancedMultiAgentSystem ready")
except Exception as exc:
    logger.error(f"Failed to initialise agent system: {exc}")
    _agent_system = None


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class APIRequestHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        logger.info("%s  %s", self.address_string(), fmt % args)

    # CORS helpers -----------------------------------------------------------

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    # JSON helpers -----------------------------------------------------------

    def _json_ok(self, data: dict):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json_error(self, status: int, message: str):
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # GET --------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/api/health":
            self._handle_health()
        elif self.path == "/api/watchlist":
            self._handle_watchlist()
        elif self.path == "/api/tickers":
            self._json_ok({"tickers": SUPPORTED_TICKERS})
        elif self.path == "/api/stream/watchlist":
            self._handle_sse_watchlist()
        else:
            self._json_error(404, "Not found")

    def _handle_health(self):
        self._json_ok({
            "status": "ok",
            "agent_system": "ready" if _agent_system else "unavailable",
            "tickers": SUPPORTED_TICKERS,
        })

    def _handle_watchlist(self):
        """Return live quotes for all supported tickers using yfinance fast_info."""
        try:
            # Use a short-window DataHandler purely for quote fetching
            today     = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
            handler   = DataHandler(
                tickers=SUPPORTED_TICKERS,
                start_date=yesterday,
                end_date=today,
            )
            quotes = [handler.fetch_live_quote(t) for t in SUPPORTED_TICKERS]
            self._json_ok({"quotes": quotes})
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before we finished sending — expected during
            # EventSource reconnect churn. Silenced.
            pass
        except Exception as exc:
            logger.exception("Watchlist fetch failed")
            try:
                self._json_error(500, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _handle_sse_watchlist(self):
        """Stream per-tick prices as Server-Sent Events from Finnhub.

        For local dev only — Vercel uses the same route in api/index.py.
        Runs until the client disconnects; no Vercel reap on the local server.
        """
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            self._json_error(503, "FINNHUB_API_KEY not configured")
            return
        if iter_quote_stream is None:
            self._json_error(503, "Quote stream unavailable (websockets import failed)")
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            for ticker, price in iter_quote_stream(SUPPORTED_TICKERS, api_key):
                payload = json.dumps({"ticker": ticker, "price": price})
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Browser closed the EventSource — normal during reconnect churn.
            pass
        except Exception as exc:
            logger.warning(f"SSE watchlist stream ended: {exc}")

    # POST -------------------------------------------------------------------

    def do_POST(self):
        if self.path == "/api/analyze":
            self._handle_analyze()
        else:
            self._json_error(404, "Not found")

    def _handle_analyze(self):
        # Parse body
        try:
            length   = int(self.headers.get("Content-Length", 0))
            raw      = self.rfile.read(length)
            req      = json.loads(raw.decode())
            ticker   = req.get("ticker", "AAPL").upper().strip()
        except Exception:
            self._json_error(400, "Invalid JSON body")
            return

        if ticker not in SUPPORTED_TICKERS:
            self._json_error(400, f"Unsupported ticker '{ticker}'. Supported: {SUPPORTED_TICKERS}")
            return

        if not _agent_system:
            self._json_error(503, "Agent system not initialised. Check GROQ_API_KEY.")
            return

        logger.info(f"Analysis request → {ticker}")

        try:
            today      = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

            handler = DataHandler(
                tickers=[ticker],
                start_date=start_date,
                end_date=today,
            )

            df = handler.fetch_and_process()
            if df.empty:
                raise ValueError(f"No market data returned for {ticker}")

            latest     = df[df["ticker"] == ticker].iloc[-1]
            date_str   = str(latest["date"])
            market_data = handler.get_market_summary(df, ticker, latest["date"])
            fundamentals = handler.fetch_fundamentals(ticker)
            news        = handler.fetch_news(ticker, days_back=7)

            # Merge fundamentals into the context dict so the Fundamental agent
            # actually receives PE / sector / market-cap instead of inventing them.
            context = {**market_data, **fundamentals}

            # Run all 4 agents + coordinator
            results = _agent_system.analyze(
                ticker=ticker,
                date=date_str,
                market_data=context,
                news=news,
            )

            # Attach context data the frontend needs for charts / news feed
            results["ticker"]       = ticker
            results["date"]         = date_str
            results["market_data"]  = market_data
            results["news"]         = news

            # Price history: full fetched window. The frontend slices
            # this array client-side based on the user's selected period
            # (30D / 60D / 3M / 6M / 1Y).
            ticker_df = df[df["ticker"] == ticker]  # full fetched window
            results["price_history"] = [
                {"date": str(row["date"]), "close": float(row["close"])}
                for _, row in ticker_df.iterrows()
            ]

            logger.info(f"✓ Analysis complete for {ticker}")
            self._json_ok(results)

        except Exception as exc:
            logger.exception(f"Analysis failed for {ticker}")
            try:
                self._json_error(500, str(exc))
            except Exception:
                # Response may already be partially written; bail silently.
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(port: int = 8000):
    addr = ("127.0.0.1", port)
    httpd = http.server.ThreadingHTTPServer(addr, APIRequestHandler)
    logger.info(f"NEXUS API running on http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down …")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)

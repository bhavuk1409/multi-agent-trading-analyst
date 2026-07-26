"""
Vercel Serverless Entry Point for NEXUS API
==========================================
Exposes Vercel-compatible serverless handler subclassing BaseHTTPRequestHandler.
Handles:
  GET  /api/health
  GET  /api/watchlist
  GET  /api/tickers
  POST /api/analyze
"""

import http.server
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root and src/ to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_handler import DataHandler
from multi_agent_system import AdvancedMultiAgentSystem

SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
HISTORY_DAYS = 90

# Re-used across warm serverless invocations
_agent_system = None


def get_agent_system():
    global _agent_system
    if _agent_system is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        _agent_system = AdvancedMultiAgentSystem(
            model=model,
            temperature=0.7,
        )
    return _agent_system


class handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _json_ok(self, data: dict):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str):
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/api/health", "/health"):
            self._json_ok({
                "status": "ok",
                "tickers": SUPPORTED_TICKERS,
            })
        elif path in ("/api/watchlist", "/watchlist"):
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
                dh = DataHandler(tickers=SUPPORTED_TICKERS, start_date=yesterday, end_date=today)
                quotes = [dh.fetch_live_quote(t) for t in SUPPORTED_TICKERS]
                self._json_ok({"quotes": quotes})
            except Exception as e:
                self._json_error(500, str(e))
        elif path in ("/api/tickers", "/tickers"):
            self._json_ok({"tickers": SUPPORTED_TICKERS})
        else:
            self._json_error(404, "Not found")

    def do_POST(self):
        path = self.path.split("?")[0]

        if path in ("/api/analyze", "/analyze"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode())
                ticker = req.get("ticker", "AAPL").upper().strip()

                if ticker not in SUPPORTED_TICKERS:
                    self._json_error(400, f"Unsupported ticker '{ticker}'")
                    return

                agent_sys = get_agent_system()
                today = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

                dh = DataHandler(tickers=[ticker], start_date=start_date, end_date=today)
                df = dh.fetch_and_process()
                if df.empty:
                    raise ValueError(f"No market data for {ticker}")

                latest = df[df["ticker"] == ticker].iloc[-1]
                date_str = str(latest["date"])
                market_data = dh.get_market_summary(df, ticker, latest["date"])
                news = dh.fetch_news(ticker, days_back=7)

                results = agent_sys.analyze(
                    ticker=ticker,
                    date=date_str,
                    market_data=market_data,
                    news=news,
                )

                results["ticker"] = ticker
                results["date"] = date_str
                results["market_data"] = market_data
                results["news"] = news

                ticker_df = df[df["ticker"] == ticker].tail(30)
                results["price_history"] = [
                    {"date": str(row["date"]), "close": float(row["close"])}
                    for _, row in ticker_df.iterrows()
                ]

                self._json_ok(results)
            except Exception as e:
                self._json_error(500, str(e))
        else:
            self._json_error(404, "Not found")

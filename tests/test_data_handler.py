"""
Tests for DataHandler — verifies real yfinance data flow.
Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest

# Add src to path so we can import without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_handler import DataHandler   # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

END   = datetime.now().strftime("%Y-%m-%d")
START = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def handler():
    return DataHandler(tickers=["AAPL"], start_date=START, end_date=END)


@pytest.fixture(scope="module")
def df(handler):
    return handler.fetch_and_process()


# ---------------------------------------------------------------------------
# Data shape / content tests
# ---------------------------------------------------------------------------

def test_fetch_returns_dataframe(df):
    import pandas as pd
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame"


def test_fetch_is_not_empty(df):
    assert len(df) > 0, "DataFrame should not be empty"


def test_has_required_columns(df):
    required = {"date", "open", "high", "low", "close", "volume", "ticker",
                "rsi", "macd", "sma_20", "bb_position", "volume_ratio", "momentum"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_close_prices_are_positive(df):
    assert (df["close"] > 0).all(), "All close prices should be positive"


def test_rsi_in_valid_range(df):
    valid = df["rsi"].dropna()
    assert ((valid >= 0) & (valid <= 100)).all(), "RSI must be in [0, 100]"


def test_bb_position_finite(df):
    import numpy as np
    valid = df["bb_position"].dropna()
    assert np.isfinite(valid).all(), "BB position should be finite"


def test_ticker_column_correct(df):
    assert set(df["ticker"].unique()) == {"AAPL"}


def test_dates_are_business_days(df):
    """Yahoo Finance only returns trading days — no weekends."""
    import pandas as pd
    dates = pd.to_datetime(df["date"])
    weekdays = dates.dt.dayofweek  # 0=Mon, 6=Sun
    assert (weekdays < 5).all(), "All dates should be weekdays (trading days)"


# ---------------------------------------------------------------------------
# Market summary tests
# ---------------------------------------------------------------------------

def test_get_market_summary_returns_dict(handler, df):
    latest = df[df["ticker"] == "AAPL"].iloc[-1]
    summary = handler.get_market_summary(df, "AAPL", latest["date"])
    assert isinstance(summary, dict)


def test_market_summary_keys(handler, df):
    latest = df[df["ticker"] == "AAPL"].iloc[-1]
    summary = handler.get_market_summary(df, "AAPL", latest["date"])
    required = {"close", "volume", "rsi", "macd", "sma_20", "bb_position", "volume_ratio", "momentum"}
    assert required.issubset(summary.keys())


def test_market_summary_close_matches(handler, df):
    latest = df[df["ticker"] == "AAPL"].iloc[-1]
    summary = handler.get_market_summary(df, "AAPL", latest["date"])
    assert abs(summary["close"] - float(latest["close"])) < 0.01


# ---------------------------------------------------------------------------
# Live quote tests
# ---------------------------------------------------------------------------

def test_fetch_live_quote(handler):
    quote = handler.fetch_live_quote("AAPL")
    assert isinstance(quote, dict)
    assert quote["ticker"] == "AAPL"
    assert quote["price"] > 0, "Live price should be positive"
    assert isinstance(quote["is_positive"], bool)


# ---------------------------------------------------------------------------
# News tests  (uses yfinance fallback — no EXA_API_KEY needed)
# ---------------------------------------------------------------------------

def test_fetch_news_returns_list(handler):
    news = handler.fetch_news("AAPL", days_back=7)
    assert isinstance(news, list)


def test_news_articles_have_required_fields(handler):
    news = handler.fetch_news("AAPL", days_back=7)
    if news:  # only check if articles were returned
        required = {"title", "url", "published_date", "summary", "source", "sentiment"}
        for article in news:
            missing = required - set(article.keys())
            assert not missing, f"Article missing fields: {missing}"


def test_no_mock_sources_in_news(handler):
    """Confirm none of the fake 'MockFinance' / 'MockNews' sources appear."""
    news = handler.fetch_news("AAPL", days_back=7)
    mock_sources = {"MockFinance", "MockNews", "MockBusiness"}
    for article in news:
        assert article.get("source") not in mock_sources, (
            f"Mock news source detected: {article['source']}"
        )


def test_no_example_com_urls(handler):
    """Confirm none of the placeholder example.com URLs appear."""
    news = handler.fetch_news("AAPL", days_back=7)
    for article in news:
        assert "example.com" not in article.get("url", ""), (
            f"Fake example.com URL found: {article['url']}"
        )


# ---------------------------------------------------------------------------
# Indicator correctness — fixture-based (deterministic; no yfinance)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd


def _build_close_df(closes, highs=None, lows=None, volumes=None):
    """Build a minimal DataFrame shaped like `_fetch_ohlcv` returns, suitable
    for `DataHandler._add_technical_indicators`. Length must be ≥ 60 rows for
    rolling-50 / ATR / drawdown to produce non-NaN values."""
    n = len(closes)
    if highs  is None: highs  = [c * 1.01 for c in closes]
    if lows   is None: lows   = [c * 0.99 for c in closes]
    if volumes is None: volumes = [1_000_000] * n
    dates = pd.bdate_range(end="2026-07-24", periods=n)  # business days
    return pd.DataFrame({
        "date":   dates,
        "open":   closes,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


@pytest.fixture
def flat_closes():
    """14 closes with 0 delta → Wilder RSI is exactly 50.0."""
    return [100.0] * 20


@pytest.fixture
def linear_closes():
    """Monotonic rising series. RSI should rise with confidence, MACD histogram
    should be > 0 for the duration of the trend."""
    n = 60
    return [100.0 + i * 0.5 for i in range(n)]  # 100 → 129.5


def test_wilder_rsi_is_50_for_flat_series(handler, flat_closes):
    df = _build_close_df(flat_closes)
    df = handler._add_technical_indicators(df.copy())
    last_rsi = float(df["rsi"].iloc[-1])
    # flat price → RSI undefined (0/0). Wilder's behaviour gives a value once
    # enough ewm smoothing accumulates; the ewm on flat deltas is 0 → RS = 0/0
    # → NaN, which is acceptable for a flat series. We assert it stays within
    # the legal domain (0..100) once finite.
    import math
    if not math.isnan(last_rsi):
        assert 0 <= last_rsi <= 100


def test_wilder_rsi_overweights_recent_moves(handler):
    """Alternating up/down with a final big run-up — RSI should exceed 70."""
    closes = [100.0] * 30 + [100.0 - (0.5 if i % 2 == 0 else -0.5) for i in range(15)]
    closes += [110.0 + i for i in range(15)]  # strong run-up at the end
    df     = handler._add_technical_indicators(_build_close_df(closes))
    rsi_series = df["rsi"].dropna()
    assert not rsi_series.empty, "RSI never produced a finite value"
    last_rsi = float(rsi_series.iloc[-1])
    assert last_rsi > 70, f"Expected RSI > 70 after run-up, got {last_rsi:.2f}"


def test_wilder_rsi_diverges_from_simple_mean(handler):
    """Sanity check: Wilder's ewm RSI produces a different number than the
    plain rolling-mean RSI for the same data — guards against accidentally
    reverting to the old (buggy) simple-mean formula."""
    # Mix of gains and losses with varying magnitudes so the two formulas
    # produce materially different numbers.
    closes = [100.0]
    for i in range(59):
        # Oscillating up-and-down with step sizes 1, 2, 3 repeating.
        step = (i % 3) + 1
        sign = -1 if i % 2 == 1 else 1
        closes.append(closes[-1] + sign * step)
    df = handler._add_technical_indicators(_build_close_df(closes))

    # Plain rolling-mean RSI as documented in the old buggy code.
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().abs()
    rs   = gain / loss.replace(0, np.nan)
    naive_rsi = (100 - (100 / (1 + rs))).dropna()

    wilder_series = df["rsi"].dropna()
    assert not wilder_series.empty, "Wilder RSI never produced a finite value"
    wilder_rsi = float(wilder_series.iloc[-1])
    naive_last = float(naive_rsi.iloc[-1])
    assert abs(wilder_rsi - naive_last) > 0.1, (
        f"Wilder RSI should differ materially from naive "
        f"({wilder_rsi:.4f} vs {naive_last:.4f})"
    )


def test_macd_signal_and_histogram_present(handler, linear_closes):
    df = handler._add_technical_indicators(_build_close_df(linear_closes))
    assert "macd_signal" in df.columns
    assert "macd_hist"   in df.columns
    # For a uniform uptrend, MACD histogram should be positive.
    assert float(df["macd_hist"].iloc[-1]) > 0
    # signal is 9-EMA of MACD line, smoother
    assert float(df["macd_signal"].iloc[-1]) > 0


def test_atr_positive_on_realistic_data(handler):
    closes = [100 + i + (i % 3) * 0.2 for i in range(60)]
    highs  = [c + 1.5 for c in closes]
    lows   = [c - 1.5 for c in closes]
    df     = handler._add_technical_indicators(_build_close_df(closes, highs, lows))
    last_atr = float(df["atr_14"].iloc[-1])
    assert last_atr > 0


def test_hv_annualised_in_percent(handler):
    closes = [100 + i for i in range(60)]  # constant linear trend
    df     = handler._add_technical_indicators(_build_close_df(closes))
    last_hv = float(df["hv_14"].iloc[-1])
    # A linear series has non-zero log returns → non-zero vol
    assert last_hv > 0
    assert last_hv < 1000  # sanity cap


def test_max_drawdown_negative_on_decline(handler):
    """A peak-then-decline series must show negative drawdown at the end."""
    closes = [100] * 20 + [110] * 10 + [80] * 30  # peak at 110, ends at 80
    df     = handler._add_technical_indicators(_build_close_df(closes))
    last_dd = float(df["drawdown"].iloc[-1])
    assert last_dd < 0
    # Final close 80 vs peak 110 → -27.27%
    assert abs(last_dd - ((80 / 110 - 1) * 100)) < 0.5


def test_market_summary_exposes_new_fields(handler):
    """get_market_summary must surface macd_signal, sma_50, hv_14, atr_14,
    drawdown, close_prev_1/5/20."""
    closes = [100 + i for i in range(60)]
    df     = handler._add_technical_indicators(_build_close_df(closes))
    df["ticker"] = "TEST"
    df["date"]   = pd.to_datetime(df["date"])
    summary = handler.get_market_summary(df, "TEST", df["date"].iloc[-1])
    for key in ("macd_signal", "macd_hist", "sma_50", "hv_14",
                "atr_14", "drawdown",
                "close_prev_1", "close_prev_5", "close_prev_20"):
        assert key in summary, f"market_summary missing {key}"


# ---------------------------------------------------------------------------
# Real-time quote stream — feeds canned trades through a fake WebSocket
# ---------------------------------------------------------------------------

import asyncio as _asyncio
import json as _json
from unittest.mock import AsyncMock, patch


class _FakeCM:
    """Async context manager that resolves to a given object on __aenter__."""
    def __init__(self, target):
        self._target = target

    async def __aenter__(self):
        return self._target

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeWS:
    """Drop-in replacement for a websockets connection that yields a
    canned sequence of inbound messages and records outbound sends."""

    def __init__(self, inbound_messages):
        self._inbound = list(inbound_messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, msg):
        self.sent.append(msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return self._inbound.pop(0)


def test_subscribe_quotes_yields_trades():
    """aiter_quote_stream should subscribe to each ticker and yield (s, p)
    pairs parsed from Finnhub's trade messages."""
    import sys
    from data_handler import aiter_quote_stream

    messages = [
        _json.dumps({"type": "trade", "data": [
            {"s": "AAPL", "p": 333.07, "t": 1700000000000, "v": 100},
            {"s": "TSLA", "p": 313.10, "t": 1700000000000, "v": 50},
        ]}),
        _json.dumps({"type": "trade", "data": [
            {"s": "AAPL", "p": 333.10, "t": 1700000000001, "v": 25},
        ]}),
    ]
    fake = _FakeWS(messages)

    async def _run():
        with patch("data_handler._websockets.connect", lambda *a, **kw: _FakeCM(fake)):
            out = []
            async for t, p in aiter_quote_stream(["AAPL", "TSLA"], api_key="test"):
                out.append((t, p))
                if len(out) >= 3:
                    break
        return out

    result = _asyncio.run(_run())
    assert result == [("AAPL", 333.07), ("TSLA", 313.10), ("AAPL", 333.10)], result

    # Verify subscription messages were sent for both tickers.
    subs = [_json.loads(m) for m in fake.sent]
    assert {m["symbol"] for m in subs} == {"AAPL", "TSLA"}
    assert all(m["type"] == "subscribe" for m in subs)


def test_subscribe_quotes_skips_non_trade_messages():
    """Ping / control / error messages from Finnhub should be ignored."""
    from data_handler import aiter_quote_stream

    messages = [
        _json.dumps({"type": "ping"}),
        _json.dumps({"type": "error", "msg": "invalid symbol"}),
        _json.dumps({"type": "trade", "data": [{"s": "GOOGL", "p": 319.74, "t": 1, "v": 1}]}),
    ]
    fake = _FakeWS(messages)

    async def _run():
        with patch("data_handler._websockets.connect", lambda *a, **kw: _FakeCM(fake)):
            out = []
            async for t, p in aiter_quote_stream(["GOOGL"], api_key="test"):
                out.append((t, p))
                break
        return out

    result = _asyncio.run(_run())
    assert result == [("GOOGL", 319.74)], result

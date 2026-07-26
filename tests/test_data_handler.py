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

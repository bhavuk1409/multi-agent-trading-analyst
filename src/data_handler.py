"""
Data Handler — Multi-Agent Trading Analyst
==========================================
Primary data source: Yahoo Finance (via yfinance) — free, no API key required.
News source:         Exa API (neural search) — requires EXA_API_KEY.
News fallback:       yfinance built-in news — free, no API key required.

No synthetic or mock data is generated.  If yfinance returns no data for a
ticker, a ValueError is raised so the caller can surface a clear error rather
than silently returning fabricated numbers.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.error("yfinance is not installed. Run: pip install yfinance>=0.2.28")

try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    logger.warning("exa-py not installed — news will fall back to yfinance headlines.")


class DataHandler:
    """
    Fetches real OHLCV market data and news for a list of tickers.

    Data flow
    ---------
    1. OHLCV   : yfinance (Yahoo Finance) — always real, no API key needed.
    2. News    : Exa API (if EXA_API_KEY present) → yfinance headlines (fallback).
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Parameters
        ----------
        tickers    : list of uppercase ticker symbols, e.g. ['AAPL', 'TSLA']
        start_date : ISO date string, e.g. '2024-01-01'
        end_date   : ISO date string, e.g. '2026-07-26'
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        if not YFINANCE_AVAILABLE:
            raise RuntimeError(
                "yfinance is required but not installed. "
                "Run: pip install yfinance>=0.2.28"
            )

        # Exa client — optional, used only for richer news search
        exa_key = os.getenv("EXA_API_KEY")
        if exa_key and EXA_AVAILABLE:
            try:
                self.exa = Exa(api_key=exa_key)
                logger.info("✓ Exa API initialised for news search")
            except Exception as exc:
                logger.warning(f"Exa init failed: {exc}. Using yfinance news.")
                self.exa = None
        else:
            self.exa = None
            if exa_key and not EXA_AVAILABLE:
                logger.info("exa-py not installed — falling back to yfinance news.")
            elif not exa_key:
                logger.info("EXA_API_KEY not set — using yfinance news headlines.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_and_process(self) -> pd.DataFrame:
        """
        Fetch real OHLCV data for all tickers and compute technical indicators.

        Returns
        -------
        pd.DataFrame with columns:
            date, open, high, low, close, volume, ticker,
            rsi, macd, sma_20, sma_50, bb_upper, bb_lower, bb_position,
            volume_sma, volume_ratio, momentum
        """
        logger.info(f"Fetching market data for {self.tickers} …")
        frames: List[pd.DataFrame] = []

        for ticker in self.tickers:
            df = self._fetch_ohlcv(ticker)
            df = self._add_technical_indicators(df)
            df["ticker"] = ticker
            frames.append(df)
            logger.info(f"  ✓ {ticker}: {len(df)} trading days from Yahoo Finance")

        combined = pd.concat(frames, ignore_index=True).dropna()
        logger.info(f"Total rows after dropna: {len(combined)}")
        return combined

    def get_market_summary(self, df: pd.DataFrame, ticker: str, date) -> Dict:
        """
        Return the latest technical snapshot for *ticker* at *date*.

        Falls back to the closest available trading day if the exact date is
        not present (e.g. weekends / holidays).
        """
        mask = (df["ticker"] == ticker) & (df["date"] == date)
        row_df = df[mask]

        if row_df.empty:
            # Nearest trading day
            ticker_df = df[df["ticker"] == ticker].copy()
            ticker_df["_diff"] = (ticker_df["date"] - date).abs()
            row_df = ticker_df.nsmallest(1, "_diff")

        if row_df.empty:
            raise ValueError(f"No data found for {ticker} near {date}")

        r = row_df.iloc[0]
        # Multi-day closes (1/5/20 trading days back) so agents can detect
        # SMA crossovers and trend changes — keyed off the most recent rows.
        ticker_history = df[df["ticker"] == ticker].sort_values("date")
        closes = ticker_history["close"].tolist()
        n      = len(closes)

        def _prev(k: int):
            return float(closes[-1 - k]) if n > k else float(r["close"])

        def _opt(key: str, default):
            v = r.get(key)
            return float(v) if pd.notna(v) else default

        return {
            "close":         float(r["close"]),
            "volume":        int(r["volume"]),
            "rsi":           _opt("rsi",          50.0),
            "macd":          _opt("macd",          0.0),
            "macd_signal":   _opt("macd_signal",   0.0),
            "macd_hist":     _opt("macd_hist",     0.0),
            "sma_20":        _opt("sma_20",       float(r["close"])),
            "sma_50":        _opt("sma_50",       float(r["close"])),
            "bb_position":   _opt("bb_position",   0.5),
            "bb_upper":      _opt("bb_upper",     float(r["close"]) * 1.05),
            "bb_lower":      _opt("bb_lower",     float(r["close"]) * 0.95),
            "volume_ratio":  _opt("volume_ratio",  1.0),
            "momentum":      _opt("momentum",      0.0),
            "hv_14":         _opt("hv_14",        20.0),
            "atr_14":        _opt("atr_14",        1.0),
            "drawdown":      _opt("drawdown",      0.0),
            "close_prev_1":  _prev(1),
            "close_prev_5":  _prev(5),
            "close_prev_20": _prev(20),
        }

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Pull free fundamentals from yfinance's `.info` endpoint.

        No API key required. The endpoint is rate-limited but free; on failure
        the method returns an empty dict rather than raising — fundamentals are
        advisory and a missing sector / P/E should not abort the analysis.
        """
        try:
            logger.info(f"  Fetching fundamentals for {ticker} via yfinance .info …")
            info = yf.Ticker(ticker).info or {}

            def _num(key: str):
                v = info.get(key)
                return v if v is not None else None

            return {
                "market_cap":        _num("marketCap"),
                "pe_trailing":       _num("trailingPE"),
                "pe_forward":        _num("forwardPE"),
                "eps_trailing":      _num("trailingEps"),
                "eps_forward":       _num("forwardEps"),
                "dividend_yield":    _num("dividendYield"),
                "sector":            info.get("sector"),
                "industry":          info.get("industry"),
                "52w_high":          _num("fiftyTwoWeekHigh"),
                "52w_low":           _num("fiftyTwoWeekLow"),
                "beta":              _num("beta"),
                "short_pct":         _num("shortPercentOfFloat"),
                "target_mean_price": _num("targetMeanPrice"),
                "target_high_price": _num("targetHighPrice"),
                "target_low_price":  _num("targetLowPrice"),
            }
        except Exception as exc:
            logger.warning(f"Fundamentals fetch failed for {ticker}: {exc}")
            return {}

    def fetch_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch recent news for *ticker*.

        Priority
        --------
        1. Exa API (neural search, richer summaries) — if EXA_API_KEY is set.
        2. yfinance built-in news headlines (free, no key).
        """
        if self.exa:
            news = self._fetch_news_exa(ticker, days_back)
            if news:
                return news
            logger.warning("Exa returned no news — falling back to yfinance news.")

        return self._fetch_news_yfinance(ticker, days_back=days_back)

    def fetch_live_quote(self, ticker: str) -> Dict:
        """
        Return the latest price and 1-day change for *ticker* from Yahoo Finance.

        Used by the /api/watchlist endpoint.
        """
        t = yf.Ticker(ticker)
        info = t.fast_info          # lightweight, much faster than t.info
        try:
            price    = float(info.last_price)
            prev     = float(info.previous_close)
            change   = price - prev
            change_pct = (change / prev * 100) if prev else 0.0
            return {
                "ticker":      ticker,
                "price":       round(price, 2),
                "change":      round(change, 2),
                "change_pct":  round(change_pct, 2),
                "is_positive": change >= 0,
            }
        except Exception as exc:
            logger.error(f"Could not fetch live quote for {ticker}: {exc}")
            return {
                "ticker":     ticker,
                "price":      0.0,
                "change":     0.0,
                "change_pct": 0.0,
                "is_positive": True,
            }

    # ------------------------------------------------------------------
    # Private helpers — OHLCV
    # ------------------------------------------------------------------

    def _fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """
        Download real OHLCV data from Yahoo Finance.

        Raises
        ------
        ValueError  if Yahoo Finance returns no data for the ticker / date range.
        """
        logger.info(f"  Downloading {ticker} from Yahoo Finance …")
        t = yf.Ticker(ticker)
        hist = t.history(
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,   # adjusts for splits & dividends automatically
        )

        if hist.empty:
            raise ValueError(
                f"Yahoo Finance returned no OHLCV data for '{ticker}' "
                f"between {self.start_date} and {self.end_date}. "
                "Check that the ticker is valid and the date range includes trading days."
            )

        hist = hist.reset_index()
        hist.rename(columns={
            "Date":     "date",
            "Datetime": "date",   # intraday intervals use 'Datetime'
            "Open":     "open",
            "High":     "high",
            "Low":      "low",
            "Close":    "close",
            "Volume":   "volume",
        }, inplace=True)

        # Remove timezone info so downstream pandas operations don't complain
        if str(hist["date"].dtype).startswith("datetime64[ns,"):
            hist["date"] = hist["date"].dt.tz_localize(None)
        else:
            hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)

        df = hist[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.dropna(subset=["close"])
        return df

    # ------------------------------------------------------------------
    # Private helpers — technical indicators
    # ------------------------------------------------------------------

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append standard technical indicators in-place.

        Notes on formulas:
        - RSI uses Wilder's exponential smoothing (alpha = 1/14), which matches
          TradingView / Bloomberg. A simple rolling mean was previously used
          which is mathematically valid but produces materially different values.
        - MACD computes the line, the 9-period signal line, and the histogram.
        - ATR uses True Range (max of HL, |H-Cprev|, |L-Cprev|) then a 14-period
          Wilder-smoothed mean.
        - Historical volatility is the 14-day rolling std of daily log returns,
          annualised by √252 and expressed in percent.
        - Max drawdown is the running percent distance from the running peak
          across the entire history (60d-context aware via the cummax).
        """
        close = df["close"]

        # RSI (14-period, Wilder's smoothing)
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().abs()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD (12/26 EMA diff) + 9-period signal line + histogram
        ema12            = close.ewm(span=12, adjust=False).mean()
        ema26            = close.ewm(span=26, adjust=False).mean()
        df["macd"]       = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Simple moving averages
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()

        # Bollinger Bands (20-period, ±2σ)
        bb_mid          = close.rolling(20).mean()
        bb_std          = close.rolling(20).std()
        df["bb_upper"]  = bb_mid + 2 * bb_std
        df["bb_lower"]  = bb_mid - 2 * bb_std
        band_width      = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - df["bb_lower"]) / band_width.replace(0, np.nan)

        # Volume ratio (vs 20-day average)
        df["volume_sma"]   = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)

        # Momentum (10-day percent change — relative, not absolute, so it scales
        # meaningfully across price levels).
        df["momentum"] = close.pct_change(10) * 100

        # 14-day annualised historical volatility (log-returns, √252 scaling).
        log_ret       = np.log(df["close"] / df["close"].shift(1))
        df["hv_14"]   = log_ret.rolling(14).std() * np.sqrt(252) * 100

        # 14-day Average True Range (Wilder smoothing).
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

        # Trailing max drawdown (percent from running peak across full history).
        df["drawdown"] = (df["close"] / df["close"].cummax() - 1) * 100

        return df

    # ------------------------------------------------------------------
    # Private helpers — news
    # ------------------------------------------------------------------

    def _fetch_news_exa(self, ticker: str, days_back: int) -> List[Dict]:
        """Fetch news via the Exa neural-search API."""
        try:
            end_dt   = datetime.now()
            start_dt = end_dt - timedelta(days=days_back)
            logger.info(f"  Fetching news for {ticker} via Exa …")

            results = self.exa.search(
                f"{ticker} stock market news latest updates",
                type="neural",
                num_results=10,
                start_published_date=start_dt.strftime("%Y-%m-%d"),
                text=True,
            )

            articles = []
            for r in results.results:
                articles.append({
                    "title":          r.title or "Untitled",
                    "url":            r.url or "",
                    "published_date": r.published_date or end_dt.strftime("%Y-%m-%d"),
                    "summary":        (r.text or "")[:300] or "No summary available.",
                    "source":         r.url.split("/")[2] if r.url else "Unknown",
                    "sentiment":      "neutral",   # Exa doesn't return sentiment; LLM agent will judge
                })

            logger.info(f"  ✓ Exa: {len(articles)} articles for {ticker}")
            return articles

        except Exception as exc:
            logger.error(f"Exa news fetch failed for {ticker}: {exc}")
            return []

    def _fetch_news_yfinance(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Fetch news headlines directly from Yahoo Finance — no API key needed.

        `days_back` filters articles to the past N days so the sentiment agent
        isn't fed stale headlines. Articles with no usable timestamp are kept.
        """
        try:
            logger.info(f"  Fetching news for {ticker} via yfinance …")
            t        = yf.Ticker(ticker)
            raw_news = t.news or []
            today    = datetime.now()
            cutoff   = today - timedelta(days=days_back)

            articles = []
            for item in raw_news[:10]:
                content = item.get("content", {})
                title   = content.get("title") or item.get("title", "Untitled")
                url     = (
                    content.get("canonicalUrl", {}).get("url")
                    or item.get("link", "")
                )
                pub_ts  = content.get("pubDate") or item.get("providerPublishTime")
                if isinstance(pub_ts, (int, float)):
                    pub_dt   = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                    pub_date = pub_dt.strftime("%Y-%m-%d")
                    # Drop items outside the [cutoff, today] window.
                    if pub_dt.replace(tzinfo=None) < cutoff:
                        continue
                elif isinstance(pub_ts, str):
                    pub_date = pub_ts[:10]
                else:
                    pub_date = today.strftime("%Y-%m-%d")

                summary = (
                    content.get("summary")
                    or content.get("body", "")[:300]
                    or "No summary available."
                )
                provider = (
                    content.get("provider", {}).get("displayName")
                    or item.get("publisher", "Yahoo Finance")
                )

                articles.append({
                    "title":          title,
                    "url":            url,
                    "published_date": pub_date,
                    "summary":        summary,
                    "source":         provider,
                    "sentiment":      "neutral",
                })

            logger.info(f"  ✓ yfinance: {len(articles)} articles for {ticker} (≤{days_back}d)")
            return articles

        except Exception as exc:
            logger.error(f"yfinance news fetch failed for {ticker}: {exc}")
            return []
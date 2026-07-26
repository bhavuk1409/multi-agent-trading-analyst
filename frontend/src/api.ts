import type { AnalysisResults, Ticker, WatchlistItem } from './types';

const API_BASE = '/api';

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/**
 * Run the multi-agent analysis via the backend.
 * Throws an Error with the server's error message on failure.
 */
export async function runAnalysis(ticker: Ticker): Promise<AnalysisResults> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker }),
    signal: AbortSignal.timeout(120_000),   // 2-minute budget for LLM chain
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    try {
      const { error } = JSON.parse(text);
      throw new Error(error || `Server error ${res.status}`);
    } catch {
      throw new Error(text || `Server error ${res.status}`);
    }
  }

  return res.json() as Promise<AnalysisResults>;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3_000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Watchlist — live quotes for all supported tickers
// ---------------------------------------------------------------------------

export async function fetchWatchlist(): Promise<WatchlistItem[]> {
  try {
    const res = await fetch(`${API_BASE}/watchlist`, {
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) throw new Error(`Watchlist fetch failed: ${res.status}`);
    const data = await res.json() as { quotes: WatchlistItem[] };
    return data.quotes ?? [];
  } catch (err) {
    console.error('fetchWatchlist error:', err);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Real-time tick stream (Server-Sent Events from /api/stream/watchlist)
// ---------------------------------------------------------------------------

/**
 * Open a real-time tick stream for the watchlist. Returns a cleanup function
 * that closes the EventSource.  ``onTick`` is called with ``(ticker, price)``
 * for every trade Finnhub pushes.  The browser auto-reconnects on transient
 * network errors; on Vercel the stream is naturally reaped at the 60 s
 * function limit, which also triggers an auto-reconnect.
 */
export function connectWatchlistStream(
  onTick: (ticker: Ticker, price: number) => void,
  onStatus?: (status: 'open' | 'error') => void,
): () => void {
  const es = new EventSource(`${API_BASE}/stream/watchlist`);
  es.onopen = () => onStatus?.('open');
  es.onerror = () => onStatus?.('error');
  es.onmessage = (e) => {
    try {
      const { ticker, price } = JSON.parse(e.data) as { ticker: Ticker; price: number };
      onTick(ticker, price);
    } catch {
      /* ignore malformed event */
    }
  };
  return () => es.close();
}

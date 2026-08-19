import type { AnalysisResults, Ticker, WatchlistItem } from './types';

const API_BASE = '/api';

/**
 * Thrown when the backend returns 429. Carries the parsed Retry-After (seconds)
 * so the UI can show a friendly countdown and gate the next click.
 */
export class RateLimitError extends Error {
  retryAfter: number;
  constructor(message: string, retryAfter: number) {
    super(message);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/**
 * Run the multi-agent analysis via the backend.
 * Throws RateLimitError on 429 (carrying Retry-After), or a generic Error
 * with the server's error message on other failures.
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
    let message = text || `Server error ${res.status}`;
    try {
      const parsed = JSON.parse(text);
      if (parsed?.error) message = parsed.error;
    } catch {
      // Non-JSON body — keep raw text as the message.
    }

    if (res.status === 429) {
      const retryAfterHeader = res.headers.get('Retry-After');
      const retryAfter = retryAfterHeader ? Math.max(1, parseInt(retryAfterHeader, 10) || 10) : 10;
      throw new RateLimitError(message, retryAfter);
    }
    throw new Error(message);
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

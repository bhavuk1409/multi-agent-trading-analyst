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

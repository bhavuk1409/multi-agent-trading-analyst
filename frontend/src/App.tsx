import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import './App.css';

import type { AgentState, AnalysisResults, Ticker, WatchlistItem } from './types';
import { checkHealth, runAnalysis, fetchWatchlist, connectWatchlistStream } from './api';
import { Header } from './components/Header';
import { AgentCard } from './components/AgentCard';
import { MarketReadout } from './components/MarketReadout';
import { DecisionPanel } from './components/DecisionPanel';
import { NewsFeed } from './components/NewsFeed';
import { IconRun, IconRefresh, IconDownload } from './components/Icons';
import { CompanySelect } from './components/CompanySelect';

const TICKERS: Ticker[] = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'];

const TICKER_NAMES: Record<Ticker, string> = {
  TSLA:  'Tesla, Inc.',
  AAPL:  'Apple Inc.',
  MSFT:  'Microsoft Corp.',
  GOOGL: 'Alphabet Inc.',
  NVDA:  'NVIDIA Corp.',
};

const INITIAL_AGENTS: AgentState[] = [
  {
    id: 'technical',
    name: 'Technical',
    icon: 'technical',
    color: '#4ecdc4',
    status: 'idle',
    description: 'Chart patterns, RSI/MACD/Bollinger Bands, moving averages, momentum',
  },
  {
    id: 'fundamental',
    name: 'Fundamental',
    icon: 'fundamental',
    color: '#f5a623',
    status: 'idle',
    description: 'Valuation, market conditions, financial metrics, industry trends',
  },
  {
    id: 'sentiment',
    name: 'Sentiment',
    icon: 'sentiment',
    color: '#a78bfa',
    status: 'idle',
    description: 'Real-time news processing, market sentiment, social signals',
  },
  {
    id: 'risk',
    name: 'Risk',
    icon: 'risk',
    color: '#ff5e5b',
    status: 'idle',
    description: 'Portfolio risk, position sizing, stop-loss / take-profit levels',
  },
  {
    id: 'rl',
    name: 'Quant',
    icon: 'rl',
    color: '#8b5cf6',
    status: 'idle',
    description: 'PPO reinforcement-learning policy — defensive/risk-hedging signal',
  },
];

// Easing for the price-counter animation. easeOutCubic — price decelerates
// into the new value, which feels snappier than a linear ramp.
const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
const PRICE_TICK_MS = 420;          // duration of one counter animation
const PRICE_TICK_FPS = 60;          // smoothness target

export default function App() {
  const [ticker, setTicker] = useState<Ticker>('AAPL');
  const [apiUp, setApiUp] = useState<boolean | null>(null);
  const [agents, setAgents] = useState<AgentState[]>(INITIAL_AGENTS);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<'idle' | 'running' | 'synthesising' | 'done'>('idle');
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [watchlistLoading, setWatchlistLoading] = useState(true);
  const [watchlistFetchedAt, setWatchlistFetchedAt] = useState<number | null>(null);
  const [now, setNow] = useState<number>(Date.now());
  // Agent-card focus mode — when set to an agent id, the AGENT ANALYSIS
  // section swaps the 5-card grid for a single full-width card showing
  // that specialist's reasoning. Cleared on every analysis re-run.
  const [focusedAgentId, setFocusedAgentId] = useState<string | null>(null);
  const agentTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const runGenerationRef = useRef(0);
  // Live ticking prices — separate from `watchlist[].price` so we can animate
  // the displayed digits from the old value to the new one over ~400 ms
  // instead of jumping. The backend `price` is the source of truth; this
  // mirror just lags behind it briefly.
  const [displayedPrices, setDisplayedPrices] = useState<Record<Ticker, number>>(
    { AAPL: 0, GOOGL: 0, MSFT: 0, TSLA: 0, NVDA: 0 } as Record<Ticker, number>,
  );
  const tickAnimRef = useRef<{
    raf: number | null;
    rafTickEnd: number;   // ms timestamp when the current animation finishes
    rafFrom: number;
    rafTo: number;
  }>({ raf: null, rafTickEnd: 0, rafFrom: 0, rafTo: 0 });
  // Per-ticker rAF handles so animation for each ticker runs independently.
  const tickerRafRef = useRef<Partial<Record<Ticker, number>>>({});

  // Health-check every 10 s
  useEffect(() => {
    let active = true;
    const ping = async () => {
      const ok = await checkHealth();
      if (active) setApiUp(ok);
    };
    ping();
    const id = setInterval(ping, 10_000);
    return () => { active = false; clearInterval(id); };
  }, []);

  // Live watchlist:
  //   1. Initial snapshot via fetchWatchlist() — establishes the prev_close
  //      baseline so change_pct is meaningful.
  //   2. EventSource pushes per-tick prices from Finnhub (sub-second).
  //   3. If the SSE stream errors out (e.g. missing FINNHUB_API_KEY on the
  //      server), the frontend falls back to 5 s polling of fetchWatchlist()
  //      so the UI still feels alive.
  useEffect(() => {
    let active = true;
    let pollId: ReturnType<typeof setInterval> | null = null;
    let closeStream: (() => void) | null = null;

    const load = async () => {
      setWatchlistLoading(true);
      const data = await fetchWatchlist();
      if (active && data.length > 0) {
        setWatchlist(prev => {
          // Preserve any prior `_flash` markers if the new fetch returns the
          // same ticker (cheap way to avoid stripping transient flash state).
          const prevByTicker = new Map(prev.map(p => [p.ticker, p]));
          return data.map(d => ({ ...d, _flash: prevByTicker.get(d.ticker)?._flash }));
        });
        // Seed the displayed-price counter with the snapshot values so the
        // first render already shows real prices (no "0.00" flicker).
        setDisplayedPrices(prev => {
          const next = { ...prev };
          for (const d of data) next[d.ticker] = d.price;
          return next;
        });
      }
      setWatchlistFetchedAt(Date.now());
      setWatchlistLoading(false);
    };

    const startFallbackPolling = () => {
      if (pollId) return;
      pollId = setInterval(load, 5_000);
    };

    const onTick = (ticker: Ticker, price: number) => {
      if (!active) return;
      let didFlash = false;
      setWatchlist(prev => prev.map(item => {
        if (item.ticker !== ticker) return item;
        const prevClose = item.price - item.change;
        const dir: 'up' | 'down' | null =
          price > item.price ? 'up' :
          price < item.price ? 'down' : null;
        if (dir) didFlash = true;
        return {
          ...item,
          price,
          change:      price - prevClose,
          change_pct:  prevClose ? ((price - prevClose) / prevClose) * 100 : 0,
          is_positive: price >= prevClose,
          _flash:      dir ?? item._flash,
        };
      }));
      // bump freshness label so the "Xs ago" line stays accurate
      setWatchlistFetchedAt(Date.now());

      // Animate the displayed price from its current value to the new target.
      // If a tick arrives mid-animation, the new animation starts from wherever
      // the digits are now — so the counter keeps moving smoothly instead of
      // snapping to the new value.
      const existingHandle = tickerRafRef.current[ticker];
      if (existingHandle) cancelAnimationFrame(existingHandle);
      const startValue = displayedPrices[ticker] ?? price;
      const startTs = performance.now();
      const animate = (now: number) => {
        const elapsed = now - startTs;
        const t = Math.min(1, elapsed / PRICE_TICK_MS);
        const eased = easeOutCubic(t);
        const value = startValue + (price - startValue) * eased;
        setDisplayedPrices(prev => ({ ...prev, [ticker]: value }));
        if (t < 1) {
          tickerRafRef.current[ticker] = requestAnimationFrame(animate);
        } else {
          tickerRafRef.current[ticker] = undefined;
        }
      };
      tickerRafRef.current[ticker] = requestAnimationFrame(animate);

      // Clear the flash class after the CSS animation completes.
      if (didFlash) {
        setTimeout(() => {
          if (!active) return;
          setWatchlist(prev => prev.map(it => it.ticker === ticker ? { ...it, _flash: null } : it));
        }, 320);
      }
    };

    load();
    closeStream = connectWatchlistStream(
      onTick,
      (status) => {
        if (status === 'error') {
          // Stream failed (likely missing FINNHUB_API_KEY) — fall back to polling.
          startFallbackPolling();
        } else if (status === 'open' && pollId) {
          // Stream came back — stop polling, ticks handle refresh.
          clearInterval(pollId);
          pollId = null;
        }
      },
    );

    return () => {
      active = false;
      if (pollId) clearInterval(pollId);
      closeStream?.();
      // Cancel any in-flight price-counter animations so we don't call
      // setState on an unmounted component.
      for (const t of Object.keys(tickerRafRef.current) as Ticker[]) {
        const h = tickerRafRef.current[t];
        if (h) cancelAnimationFrame(h);
      }
      tickerRafRef.current = {};
    };
  }, []);

  // Tick a 'now' state every second so the watchlist freshness label
  // ("Ns ago") updates without us re-running the watcher effect.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  function resetAgents() {
    agentTimersRef.current.forEach(clearTimeout);
    agentTimersRef.current = [];
    setAgents(INITIAL_AGENTS);
  }

  /** Simulate the 4 agents lighting up one by one, then fire the real API. */
  async function handleRun() {
    if (loading) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setFocusedAgentId(null);
    setStep('running');
    resetAgents();

    // Bump generation so any in-flight timers / promises from a previous run
    // no-op when they eventually fire.
    const generation = ++runGenerationRef.current;

    // Animate agents lighting up almost simultaneously so all five are
    // "scanning" before the parallel backend resolves (~600 ms total).
    const agentIds = ['technical', 'fundamental', 'sentiment', 'risk', 'rl'];
    agentIds.forEach((id, i) => {
      const t = setTimeout(() => {
        if (runGenerationRef.current !== generation) return;
        setAgents(prev => prev.map(a => a.id === id ? { ...a, status: 'running' } : a));
      }, i * 150);
      agentTimersRef.current.push(t);
    });

    try {
      const data = await runAnalysis(ticker);

      // A newer run started while we were awaiting the API — discard this result.
      if (runGenerationRef.current !== generation) return;

      // Brief synthesising interstitial so the user sees the Coordinator step
      // — otherwise the parallel agents appear to "freeze" before the decision.
      setStep('synthesising');
      await new Promise<void>(r => setTimeout(r, 300));
      if (runGenerationRef.current !== generation) return;

      // Map results onto agent states
      setAgents(prev => prev.map(a => {
        const key = `${a.id}_analysis` as keyof AnalysisResults;
        const analysis = data[key] as AgentState['analysis'];
        return { ...a, status: analysis ? 'done' : 'error', analysis };
      }));

      setResults(data);
      setStep('done');
    } catch (e) {
      if (runGenerationRef.current !== generation) return;
      setError(e instanceof Error ? e.message : 'Analysis failed.');
      setAgents(prev => prev.map(a => ({ ...a, status: 'error' })));
      setStep('idle');
    } finally {
      if (runGenerationRef.current === generation) setLoading(false);
    }
  }

  function handleReset() {
    resetAgents();
    setResults(null);
    setFocusedAgentId(null);
    setStep('idle');
    setError(null);
  }

  const agentKeys = ['technical', 'fundamental', 'sentiment', 'risk', 'rl'] as const;

  // Watchlist header aggregate — drives "5 stocks · avg +0.42% · 12s ago".
  const watchlistAggregate = (() => {
    if (watchlistLoading || watchlist.length === 0) return null;
    const avg = watchlist.reduce((sum, item) => sum + item.change_pct, 0) / watchlist.length;
    const sign = avg >= 0 ? '+' : '';
    return {
      count: watchlist.length,
      avgPctStr: `${sign}${avg.toFixed(2)}%`,
      isPositive: avg >= 0,
    };
  })();

  const freshnessLabel = (() => {
    if (!watchlistFetchedAt) return null;
    const secAgo = Math.max(0, Math.floor((now - watchlistFetchedAt) / 1000));
    if (secAgo < 5)  return 'just now';
    if (secAgo < 60) return `${secAgo}s ago`;
    const minAgo = Math.floor(secAgo / 60);
    if (minAgo < 60) return `${minAgo}m ago`;
    return '1h+ ago';
  })();

  return (
    <>
      <Header apiUp={apiUp} />

      <main className="nexus-layout">
        {/* ====== LEFT COLUMN: Control + Neural Viz ====== */}
        <section className="nexus-left">
          <div className="control-pod glass">
            <label className="control-pod__label">SELECT COMPANY</label>
            <CompanySelect
              selected={ticker}
              onChange={(t) => {
                setTicker(t);
                if (step !== 'idle') handleReset();
              }}
              disabled={loading}
            />

            <div className="control-pod__actions">
              {step === 'idle' || step === 'running' || step === 'synthesising' ? (
                <button
                  id="run-analysis-btn"
                  className="btn btn--primary run-btn"
                  onClick={handleRun}
                  disabled={loading}
                >
                  {loading ? (
                    <><span className="spinner" />ANALYZING…</>
                  ) : (
                    <><IconRun size={15} color="currentColor" /> RUN ANALYSIS</>
                  )}
                </button>
              ) : (
                <button
                  id="new-analysis-btn"
                  className="btn btn--primary run-btn"
                  onClick={handleReset}
                >
                  <IconRefresh size={14} color="currentColor" /> NEW ANALYSIS
                </button>
              )}

              {error && (
                <motion.div
                  className="error-msg"
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  ⚠ {error}
                </motion.div>
              )}
            </div>
          </div>

          {/* Market Watchlist */}
          <div className="watchlist-pod glass">
            <div className="watchlist-pod__header">
              <span className="watchlist-pod__title">MARKET WATCHLIST</span>
              <span className="watchlist-pod__status-dot dot dot--live" />
            </div>
            <div className="watchlist-pod__meta">
              <span className="watchlist-pod__meta-item">
                <strong>{watchlistAggregate?.count ?? TICKERS.length}</strong>
                <span className="faint"> stocks</span>
              </span>
              {watchlistAggregate && (
                <>
                  <span className="watchlist-pod__meta-sep" aria-hidden="true">·</span>
                  <span className="watchlist-pod__meta-item">
                    <span className="faint">avg</span>{' '}
                    <strong className={watchlistAggregate.isPositive ? 'txt-buy' : 'txt-sell'}>
                      {watchlistAggregate.avgPctStr}
                    </strong>
                  </span>
                </>
              )}
              {freshnessLabel && (
                <>
                  <span className="watchlist-pod__meta-sep" aria-hidden="true">·</span>
                  <span className="watchlist-pod__meta-item faint">{freshnessLabel}</span>
                </>
              )}
            </div>
            <div className="watchlist-list">
              {watchlistLoading
                ? TICKERS.map(t => (
                    <div key={t} className="watchlist-item watchlist-item--skeleton">
                      <div className="watchlist-item__left">
                        <span className="watchlist-item__ticker">{t}</span>
                        <span className="watchlist-item__name">{TICKER_NAMES[t]}</span>
                      </div>
                      <div className="watchlist-item__right">
                        <span className="watchlist-item__price skeleton-text">——</span>
                        <span className="watchlist-item__change">——</span>
                      </div>
                    </div>
                  ))
                : watchlist.map((item) => {
                    // Prefer the animated `displayedPrice` (lags behind
                    // `item.price` for ~400 ms after a tick) so the digits
                    // visibly count up/down to the new value instead of
                    // jumping. Fall back to `item.price` if the counter
                    // hasn't been seeded yet (first render).
                    const displayed =
                      displayedPrices[item.ticker] > 0
                        ? displayedPrices[item.ticker]
                        : item.price;
                    const priceStr  = `$${displayed.toFixed(2)}`;
                    const sign      = item.is_positive ? '+' : '';
                    const changeStr = `${sign}${item.change_pct.toFixed(2)}%`;
                    return (
                      <div
                        key={item.ticker}
                        className={`watchlist-item ${ticker === item.ticker ? 'watchlist-item--active' : ''}`}
                        aria-label={`${item.ticker} ${item.is_positive ? 'up' : 'down'} ${changeStr} — informational only, use the selector above to analyse`}
                      >
                        <div className="watchlist-item__left">
                          <span className="watchlist-item__ticker">{item.ticker}</span>
                          <span className="watchlist-item__name">{TICKER_NAMES[item.ticker]}</span>
                        </div>
                        <div className="watchlist-item__right">
                          <span className={
                            'watchlist-item__price' +
                            (item._flash === 'up'   ? ' watchlist-item__price--flash-up'   : '') +
                            (item._flash === 'down' ? ' watchlist-item__price--flash-down' : '')
                          }>{priceStr}</span>
                          <span className={`watchlist-item__change ${item.is_positive ? 'txt-buy' : 'txt-sell'}`}>
                            {changeStr}
                          </span>
                        </div>
                      </div>
                    );
                  })
              }
            </div>
          </div>
          {/* Live agent status — only shown while a run is in progress.
              After completion, the per-agent cards and Final Decision panel
              already convey the verdict; the legend becomes redundant. */}
          {(step === 'running' || step === 'synthesising') && (
            <div className="agent-legend glass">
              {agents.map(agent => (
                <div key={agent.id} className="agent-legend__item">
                  <div className={`agent-legend__dot ${agent.status === 'running' ? 'dot--wait' : agent.status === 'done' ? 'dot--live' : 'dot'}`}
                    style={{ background: agent.status === 'idle' ? 'var(--text-3)' : undefined }}
                  />
                  <span className="agent-legend__name">{agent.name}</span>
                  <span className={`agent-legend__status agent-legend__status--${agent.status}`}>
                    {agent.status === 'running' ? 'SCANNING' : agent.status === 'done' ? 'DONE' : agent.status === 'error' ? 'ERROR' : 'STANDBY'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ====== RIGHT COLUMN: Results ====== */}
        <section className="nexus-right">
          <AnimatePresence mode="wait">
            {step === 'idle' && (
              <motion.div
                key="empty-state"
                className="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
              >
                <div className="empty-state__inner">
                  <div className="empty-state__grid" aria-hidden="true" />
                  <div className="pulse-logo pulse-logo--idle">
                    <svg width="112" height="112" viewBox="0 0 44 44" fill="none">
                      <polygon
                        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                        stroke="url(#hex-grad-large)"
                        strokeWidth="2.4"
                        fill="none"
                      />
                      <polygon
                        points="22,8 35,15.5 35,30.8 22,38.3 9,30.8 9,15.5"
                        stroke="rgba(255,255,255,0.18)"
                        strokeWidth="1"
                        fill="none"
                      />
                      <defs>
                        <linearGradient id="hex-grad-large" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
                          <stop offset="0%" stopColor="#ffffff" />
                          <stop offset="100%" stopColor="#1a1a1a" />
                        </linearGradient>
                      </defs>
                    </svg>
                  </div>
                  <h2 className="empty-state__title">INTELLIGENCE STANDBY</h2>
                  <p className="empty-state__desc">
                    Select a ticker and launch the multi-agent analysis.<br />
                    Five specialized models will simultaneously process technical signals,<br />
                    fundamental data, sentiment, risk, and a PPO quant model — then synthesize a final call.
                  </p>
                  <div className="empty-state__agents">
                    {INITIAL_AGENTS.map(a => (
                      <div key={a.id} className="empty-state__agent-pill">
                        {a.name}
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {step === 'running' && !results && (
              <motion.div
                key="analyzing-state"
                className="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
              >
                <div className="empty-state__inner">
                  <div className="empty-state__grid" aria-hidden="true" />
                  <div className="pulse-logo pulse-logo--active">
                    <svg width="112" height="112" viewBox="0 0 44 44" fill="none">
                      <polygon
                        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                        stroke="url(#hex-grad-analyzing)"
                        strokeWidth="2.4"
                        fill="none"
                      />
                      <polygon
                        points="22,8 35,15.5 35,30.8 22,38.3 9,30.8 9,15.5"
                        stroke="rgba(255,255,255,0.20)"
                        strokeWidth="1"
                        fill="none"
                      />
                      <defs>
                        <linearGradient id="hex-grad-analyzing" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
                          <stop offset="0%" stopColor="#ffffff" />
                          <stop offset="100%" stopColor="#1a1a1a" />
                        </linearGradient>
                      </defs>
                    </svg>
                  </div>
                  <h2 className="empty-state__title" style={{ letterSpacing: '0.24em' }}>ANALYZING MARKET</h2>
                  <p className="empty-state__desc">
                    Retrieving market statistics and running multi-agent consensus chains...<br />
                    This process compiles real-time feeds and technical signals.
                  </p>
                </div>
              </motion.div>
            )}

            {step === 'synthesising' && !results && (
              <motion.div
                key="synthesising-state"
                className="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
              >
                <div className="empty-state__inner">
                  <div className="empty-state__grid" aria-hidden="true" />
                  <div className="pulse-logo pulse-logo--active">
                    <svg width="112" height="112" viewBox="0 0 44 44" fill="none">
                      <polygon
                        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                        stroke="url(#hex-grad-synthesising)"
                        strokeWidth="2.4"
                        fill="none"
                      />
                      <polygon
                        points="22,8 35,15.5 35,30.8 22,38.3 9,30.8 9,15.5"
                        stroke="rgba(245,166,35,0.30)"
                        strokeWidth="1"
                        fill="none"
                      />
                      <defs>
                        <linearGradient id="hex-grad-synthesising" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
                          <stop offset="0%" stopColor="#f5a623" />
                          <stop offset="100%" stopColor="#1a1a1a" />
                        </linearGradient>
                      </defs>
                    </svg>
                  </div>
                  <h2 className="empty-state__title" style={{ letterSpacing: '0.24em' }}>SYNTHESISING FINAL CALL</h2>
                  <p className="empty-state__desc">
                    All five agents have returned their analyses.<br />
                    Coordinator Agent is merging them into a single recommendation…
                  </p>
                </div>
              </motion.div>
            )}

            {results && (
              <motion.div
                key="results"
                className="results-column"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {results?.market_data && (
                  <MarketReadout ticker={ticker} data={results.market_data} history={results.price_history} />
                )}

                {/* Agent analysis cards */}
                {step === 'done' && (
                  <div className="agent-cards-section">
                    <div className="section-header">
                      <span className="section-header__tag">
                        {focusedAgentId ? 'AGENT DETAIL' : 'AGENT ANALYSIS'}
                      </span>
                      <span className="faint" style={{ fontSize: 12 }}>
                        {focusedAgentId
                          ? 'Focused view · click Back to compare all 5'
                          : '5 specialized models · 4 AI analysts + 1 quant model · parallel inference'}
                      </span>
                    </div>

                    {focusedAgentId ? (() => {
                      // Resolve the focused agent + its analysis (mirrors the
                      // grid's resultKey handling — rl uses 'rl_analysis').
                      const agent = agents.find(a => a.id === focusedAgentId);
                      const resultKey = focusedAgentId === 'rl'
                        ? 'rl_analysis'
                        : `${focusedAgentId}_analysis` as keyof AnalysisResults;
                      const analysis = results?.[resultKey] as AgentState['analysis'];
                      if (!agent || !analysis) return null;
                      return (
                        // key={focusedAgentId} so framer-motion replays the
                        // enter animation when the user clicks a different
                        // card from the grid.
                        <div key={focusedAgentId} className="agent-card-focus">
                          <AgentCard
                            agent={agent}
                            analysis={analysis}
                            index={0}
                            variant="focus"
                            onBack={() => setFocusedAgentId(null)}
                          />
                        </div>
                      );
                    })() : (
                      // Original 5-card grid view.
                      <div className="agent-cards-grid">
                        {agentKeys.map((key, i) => {
                          const agent = agents.find(a => a.id === key);
                          const resultKey = key === 'rl'
                            ? 'rl_analysis'
                            : `${key}_analysis` as keyof AnalysisResults;
                          const analysis = results?.[resultKey] as AgentState['analysis'];
                          if (!agent || !analysis) return null;
                          return (
                            <AgentCard
                              key={key}
                              agent={agent}
                              analysis={analysis}
                              index={i}
                              variant="grid"
                              onFocus={() => setFocusedAgentId(key)}
                            />
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                {/* News feed */}
                {results?.news && results.news.length > 0 && (
                  <NewsFeed articles={results.news} />
                )}

                {/* Final decision */}
                {results?.final_decision && (
                  <div className="decision-section">
                    <div className="section-header">
                      <span className="section-header__tag section-header__tag--gold">FINAL DECISION</span>
                      <span className="faint" style={{ fontSize: 12 }}>Coordinator Agent synthesis</span>
                    </div>
                    <DecisionPanel decision={results.final_decision} ticker={ticker} />
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>

      <footer className="nexus-footer">
        <span className="mono faint">NEXUS · MULTI-AGENT TRADING INTELLIGENCE</span>
        <span className="mono faint">GROQ · LLaMA 3.3 70B</span>
      </footer>
    </>
  );
}

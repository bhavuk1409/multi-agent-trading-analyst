import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import './App.css';

import type { AgentState, AnalysisResults, Ticker, WatchlistItem } from './types';
import { checkHealth, runAnalysis, fetchWatchlist } from './api';
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
    color: '#ffffff',
    status: 'idle',
    description: 'Chart patterns, RSI/MACD/Bollinger Bands, moving averages, momentum',
  },
  {
    id: 'fundamental',
    name: 'Fundamental',
    icon: 'fundamental',
    color: '#bbbbbb',
    status: 'idle',
    description: 'Valuation, market conditions, financial metrics, industry trends',
  },
  {
    id: 'sentiment',
    name: 'Sentiment',
    icon: 'sentiment',
    color: '#888888',
    status: 'idle',
    description: 'Real-time news processing, market sentiment, social signals',
  },
  {
    id: 'risk',
    name: 'Risk',
    icon: 'risk',
    color: '#555555',
    status: 'idle',
    description: 'Portfolio risk, position sizing, stop-loss / take-profit levels',
  },
];

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
  const agentTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const runGenerationRef = useRef(0);

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

  // Live watchlist on mount — refetch every 60 s
  useEffect(() => {
    let active = true;
    const load = async () => {
      setWatchlistLoading(true);
      const data = await fetchWatchlist();
      if (active && data.length > 0) setWatchlist(data);
      setWatchlistLoading(false);
    };
    load();
    const id = setInterval(load, 60_000);
    return () => { active = false; clearInterval(id); };
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
    setStep('running');
    resetAgents();

    // Bump generation so any in-flight timers / promises from a previous run
    // no-op when they eventually fire.
    const generation = ++runGenerationRef.current;

    // Animate agents lighting up almost simultaneously so all four are
    // "scanning" before the parallel backend resolves (~450 ms total).
    const agentIds = ['technical', 'fundamental', 'sentiment', 'risk'];
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
    setStep('idle');
    setError(null);
  }

  const agentKeys = ['technical', 'fundamental', 'sentiment', 'risk'] as const;

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
                    const priceStr  = `$${item.price.toFixed(2)}`;
                    const sign      = item.is_positive ? '+' : '';
                    const changeStr = `${sign}${item.change_pct.toFixed(2)}%`;
                    return (
                      <button
                        key={item.ticker}
                        className={`watchlist-item ${ticker === item.ticker ? 'watchlist-item--active' : ''}`}
                        onClick={() => {
                          if (loading) return;
                          setTicker(item.ticker);
                          if (step !== 'idle') handleReset();
                        }}
                        disabled={loading}
                      >
                        <div className="watchlist-item__left">
                          <span className="watchlist-item__ticker">{item.ticker}</span>
                          <span className="watchlist-item__name">{TICKER_NAMES[item.ticker]}</span>
                        </div>
                        <div className="watchlist-item__right">
                          <span className="watchlist-item__price">{priceStr}</span>
                          <span className={`watchlist-item__change ${item.is_positive ? 'txt-buy' : 'txt-sell'}`}>
                            {changeStr}
                          </span>
                        </div>
                      </button>
                    );
                  })
              }
            </div>
          </div>
          <div className="agent-legend glass">
            {agents.map(agent => (
              <div key={agent.id} className="agent-legend__item">
                <div className={`agent-legend__dot ${agent.status === 'running' ? 'dot--wait' : agent.status === 'done' ? 'dot--live' : 'dot'}`}
                  style={{ background: agent.status === 'idle' ? 'var(--text-3)' : undefined }}
                />
                <span className="agent-legend__name">{agent.name}</span>
                <span className={`agent-legend__status agent-legend__status--${agent.status}`}>
                  {agent.status === 'running' ? 'SCANNING' : agent.status === 'done' ? agent.analysis?.recommendation?.toUpperCase() ?? 'DONE' : agent.status === 'error' ? 'ERROR' : 'STANDBY'}
                </span>
              </div>
            ))}
          </div>
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
                    <svg width="68" height="68" viewBox="0 0 44 44" fill="none">
                      <polygon
                        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                        stroke="url(#hex-grad-large)"
                        strokeWidth="3.0"
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
                    Four specialized AI agents will simultaneously process technical signals,<br />
                    fundamental data, sentiment, and risk — then synthesize a final call.
                  </p>
                  <div className="empty-state__agents">
                    {INITIAL_AGENTS.map(a => (
                      <div key={a.id} className="empty-state__agent-pill" style={{ borderColor: `${a.color}40`, color: a.color }}>
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
                    <svg width="72" height="72" viewBox="0 0 44 44" fill="none">
                      <polygon
                        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                        stroke="url(#hex-grad-analyzing)"
                        strokeWidth="3.5"
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
                    <svg width="72" height="72" viewBox="0 0 44 44" fill="none">
                      <polygon
                        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                        stroke="url(#hex-grad-synthesising)"
                        strokeWidth="3.5"
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
                    All four agents have returned their analyses.<br />
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
                      <span className="section-header__tag">AGENT ANALYSIS</span>
                      <span className="faint" style={{ fontSize: 12 }}>4 specialized models · parallel inference</span>
                    </div>
                    <div className="agent-cards-grid">
                      {agentKeys.map((key, i) => {
                        const agent = agents.find(a => a.id === key);
                        const analysis = results?.[`${key}_analysis` as keyof AnalysisResults] as AgentState['analysis'];
                        if (!agent || !analysis) return null;
                        return (
                          <AgentCard
                            key={key}
                            agent={agent}
                            analysis={analysis}
                            index={i}
                          />
                        );
                      })}
                    </div>
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
        <span className="mono faint">LangChain · LLaMA 3.3</span>
      </footer>
    </>
  );
}

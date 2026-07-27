import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { AgentAnalysis, AgentState } from '../types';
import { IconTechnical, IconFundamental, IconSentiment, IconRisk, IconRL, IconChevronDown } from './Icons';
import { HelpTip } from './HelpTip';
import { HELP_TEXT } from '../helpText';

interface AgentCardProps {
  agent: AgentState;
  analysis: AgentAnalysis;
  index: number;
}

const AGENT_FULL_NAMES: Record<string, string> = {
  technical:   'Technical Analyst',
  fundamental: 'Fundamental Analyst',
  sentiment:   'Sentiment Analyst',
  risk:        'Risk Manager',
  rl:          'Quant Model',
};

const AGENT_DESCS: Record<string, string> = {
  technical:   'Chart Patterns · RSI · MACD · Bollinger Bands',
  fundamental: 'Valuation · Market Conditions · Financial Metrics',
  sentiment:   'News Processing · Social Signals · Market Mood',
  risk:        'Position Sizing · Stop-Loss · Risk/Reward',
  rl:          'PPO Policy · Downtrend Shield · Indicator Patterns',
};

/** Whether the agent is an LLM (AI Analyst) vs a trained RL policy (Quant Model) */
const AGENT_TYPE: Record<string, 'AI ANALYST' | 'QUANT MODEL'> = {
  technical:   'AI ANALYST',
  fundamental: 'AI ANALYST',
  sentiment:   'AI ANALYST',
  risk:        'AI ANALYST',
  rl:          'QUANT MODEL',
};

/** Heuristic: detect a backend fallback so we can render a recovery UI instead
 * of pretending the model returned a 50/50/hold verdict. */
function isFallback(analysis: AgentAnalysis): boolean {
  const r = (analysis.reasoning || '').toLowerCase();
  return r.includes('unavailable') || r.includes('analysis error') || r.includes('analysis failed');
}

function AgentIcon({ id, size = 20 }: { id: string; size?: number }) {
  if (id === 'technical')   return <IconTechnical   size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'fundamental') return <IconFundamental size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'sentiment')   return <IconSentiment   size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'risk')        return <IconRisk        size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'rl')          return <IconRL          size={size} color="currentColor" strokeWidth={1.5} />;
  return null;
}

export function AgentCard({ agent, analysis, index }: AgentCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const rec = analysis.recommendation;
  const signalColor =
    rec === 'buy'
      ? 'var(--buy)'
      : rec === 'sell'
      ? 'var(--sell)'
      : 'var(--hold)';

  const recBg =
    rec === 'buy'
      ? 'var(--buy-dim)'
      : rec === 'sell'
      ? 'var(--sell-dim)'
      : 'var(--hold-dim)';

  const recBorder =
    rec === 'buy'
      ? 'rgba(0, 229, 160, 0.25)'
      : rec === 'sell'
      ? 'rgba(255, 59, 48, 0.25)'
      : 'rgba(245, 166, 35, 0.25)';

  const fallback = isFallback(analysis);

  return (
    <motion.div
      className={`agent-card glass ${fallback ? 'agent-card--error' : ''} ${isExpanded ? 'agent-card--expanded' : ''}`}
      style={{ '--agent-accent': agent.color, cursor: 'pointer', userSelect: 'none' } as React.CSSProperties}
      initial={{ opacity: 0, y: 24, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: index * 0.07, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      onClick={() => setIsExpanded(prev => !prev)}
      role="button"
      tabIndex={0}
      aria-expanded={isExpanded}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          setIsExpanded(prev => !prev);
        }
      }}
    >
      <div className="agent-card__header">
        <div className="agent-card__identity">
          <div
            className="agent-card__icon-wrap"
            style={{ color: fallback ? 'var(--text-3)' : signalColor, borderColor: fallback ? 'rgba(255,255,255,0.12)' : `${signalColor}30`, background: fallback ? 'rgba(255,255,255,0.03)' : `${signalColor}08` }}
          >
            <AgentIcon id={agent.id} size={18} />
          </div>
          <div style={{ minWidth: 0 }}>
            <div className="agent-card__name">
              {AGENT_FULL_NAMES[agent.id] ?? agent.name}
            </div>
            {/* Type badge — distinguishes Quant Model from AI Analyst cards.
                Sits on its own line below the name so it can't split "Technical
                / Analyst" mid-word when the card is narrow. */}
            <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span
                className="agent-card__type-badge"
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  letterSpacing: '0.08em',
                  padding: '2px 6px',
                  borderRadius: 4,
                  whiteSpace: 'nowrap',
                  background: agent.id === 'rl' ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.06)',
                  color: agent.id === 'rl' ? '#a78bfa' : 'var(--text-3)',
                  border: agent.id === 'rl' ? '1px solid rgba(139,92,246,0.3)' : '1px solid rgba(255,255,255,0.10)',
                }}
              >
                {AGENT_TYPE[agent.id] ?? 'AI ANALYST'}
              </span>
            </div>
            <div className="agent-card__desc faint" style={{ marginTop: 4 }}>{AGENT_DESCS[agent.id]}</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {fallback ? (
            <div
              className="agent-card__rec agent-card__rec--error"
              title={analysis.reasoning}
            >
              <span className="agent-card__rec-text" style={{ color: 'var(--text-3)' }}>
                NO DATA
              </span>
            </div>
          ) : (
            <div
              className="agent-card__rec"
              style={{ background: recBg, border: `1px solid ${recBorder}` }}
            >
              <span className="agent-card__rec-text" style={{ color: signalColor }}>
                {analysis.recommendation.toUpperCase()}
              </span>
            </div>
          )}
          <motion.div
            style={{ color: 'var(--text-3)', display: 'flex', alignItems: 'center' }}
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <IconChevronDown size={14} />
          </motion.div>
        </div>
      </div>

      {/* Body: confidence bar (always visible) */}
      {fallback ? (
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              className="agent-card__error"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
            >
              <div className="agent-card__error-icon" aria-hidden="true">⚠</div>
              <div className="agent-card__error-body">
                <div className="agent-card__error-title">Agent returned no data</div>
                <div className="agent-card__error-msg">
                  The {AGENT_FULL_NAMES[agent.id] ?? agent.name} did not respond —
                  most likely a temporary rate limit. Re-run the analysis to retry this agent.
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      ) : (
        <>
          {/* Confidence bar */}
          <div className="agent-card__confidence">
            <div className="agent-card__conf-header">
              <span className="faint" onClick={(e) => e.stopPropagation()}>
                Confidence
                <HelpTip id={`tip-confidence-${agent.id}`} text={HELP_TEXT.confidence} />
              </span>
              <span className="mono" style={{ color: signalColor }}>{analysis.confidence}%</span>
            </div>
            <div className="agent-card__conf-track">
              <motion.div
                className="agent-card__conf-fill"
                style={{ background: `linear-gradient(90deg, ${signalColor} 0%, ${signalColor}40 100%)` }}
                initial={{ width: 0 }}
                animate={{ width: `${analysis.confidence}%` }}
                transition={{ delay: index * 0.07 + 0.3, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
          </div>

          {/* Reasoning (collapsible) */}
          <AnimatePresence initial={false}>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, height: 0, marginTop: 0 }}
                animate={{ opacity: 1, height: 'auto', marginTop: 8 }}
                exit={{ opacity: 0, height: 0, marginTop: 0 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                style={{ overflow: 'hidden' }}
              >
                <p className="agent-card__reasoning">{analysis.reasoning}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </motion.div>
  );
}

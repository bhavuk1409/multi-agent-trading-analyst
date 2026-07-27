import { motion } from 'framer-motion';
import type { AgentAnalysis, AgentState } from '../types';
import { IconTechnical, IconFundamental, IconSentiment, IconRisk, IconRL } from './Icons';
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
      className={`agent-card glass ${fallback ? 'agent-card--error' : ''}`}
      style={{ '--agent-accent': agent.color } as React.CSSProperties}
      initial={{ opacity: 0, y: 24, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: index * 0.07, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >


      <div className="agent-card__header">
        <div className="agent-card__identity">
          <div
            className="agent-card__icon-wrap"
            style={{ color: fallback ? 'var(--text-3)' : signalColor, borderColor: fallback ? 'rgba(255,255,255,0.12)' : `${signalColor}30`, background: fallback ? 'rgba(255,255,255,0.03)' : `${signalColor}08` }}
          >
            <AgentIcon id={agent.id} size={18} />
          </div>
          <div>
            <div className="agent-card__name">
              {AGENT_FULL_NAMES[agent.id] ?? agent.name}
              {/* Type badge — distinguishes Quant Model from AI Analyst cards */}
              <span
                className="agent-card__type-badge"
                style={{
                  marginLeft: 7,
                  fontSize: 9,
                  fontWeight: 700,
                  letterSpacing: '0.08em',
                  padding: '2px 6px',
                  borderRadius: 4,
                  verticalAlign: 'middle',
                  background: agent.id === 'rl' ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.06)',
                  color: agent.id === 'rl' ? '#a78bfa' : 'var(--text-3)',
                  border: agent.id === 'rl' ? '1px solid rgba(139,92,246,0.3)' : '1px solid rgba(255,255,255,0.10)',
                }}
              >
                {AGENT_TYPE[agent.id] ?? 'AI ANALYST'}
              </span>
            </div>
            <div className="agent-card__desc faint">{AGENT_DESCS[agent.id]}</div>
          </div>
        </div>

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
      </div>

      {/* Body: either confidence + reasoning (normal) or a clear recovery prompt */}
      {fallback ? (
        <div className="agent-card__error">
          <div className="agent-card__error-icon" aria-hidden="true">⚠</div>
          <div className="agent-card__error-body">
            <div className="agent-card__error-title">Agent returned no data</div>
            <div className="agent-card__error-msg">
              The {AGENT_FULL_NAMES[agent.id] ?? agent.name} did not respond —
              most likely a temporary rate limit. Re-run the analysis to retry this agent.
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Confidence bar */}
          <div className="agent-card__confidence">
            <div className="agent-card__conf-header">
              <span className="faint">
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

          {/* Reasoning */}
          <p className="agent-card__reasoning">{analysis.reasoning}</p>
        </>
      )}
    </motion.div>
  );
}

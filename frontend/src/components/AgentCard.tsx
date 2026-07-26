import { motion } from 'framer-motion';
import type { AgentAnalysis, AgentState } from '../types';
import { IconTechnical, IconFundamental, IconSentiment, IconRisk } from './Icons';
import { HelpTip } from './HelpTip';
import { HELP_TEXT } from '../helpText';

interface AgentCardProps {
  agent: AgentState;
  analysis: AgentAnalysis;
  index: number;
}

const AGENT_FULL_NAMES: Record<string, string> = {
  technical: 'Technical Analyst',
  fundamental: 'Fundamental Analyst',
  sentiment: 'Sentiment Analyst',
  risk: 'Risk Manager',
};

const AGENT_DESCS: Record<string, string> = {
  technical: 'Chart Patterns · RSI · MACD · Bollinger Bands',
  fundamental: 'Valuation · Market Conditions · Financial Metrics',
  sentiment: 'News Processing · Social Signals · Market Mood',
  risk: 'Position Sizing · Stop-Loss · Risk/Reward',
};

function AgentIcon({ id, size = 20 }: { id: string; size?: number }) {
  if (id === 'technical') return <IconTechnical size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'fundamental') return <IconFundamental size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'sentiment') return <IconSentiment size={size} color="currentColor" strokeWidth={1.5} />;
  if (id === 'risk') return <IconRisk size={size} color="currentColor" strokeWidth={1.5} />;
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

  return (
    <motion.div
      className="agent-card glass"
      style={{ '--agent-accent': agent.color } as React.CSSProperties}
      initial={{ opacity: 0, y: 24, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: index * 0.07, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >


      <div className="agent-card__header">
        <div className="agent-card__identity">
          <div
            className="agent-card__icon-wrap"
            style={{ color: signalColor, borderColor: `${signalColor}30`, background: `${signalColor}08` }}
          >
            <AgentIcon id={agent.id} size={18} />
          </div>
          <div>
            <div className="agent-card__name">{AGENT_FULL_NAMES[agent.id] ?? agent.name}</div>
            <div className="agent-card__desc faint">{AGENT_DESCS[agent.id]}</div>
          </div>
        </div>

        <div
          className="agent-card__rec"
          style={{ background: recBg, border: `1px solid ${recBorder}` }}
        >
          <span className="agent-card__rec-text" style={{ color: signalColor }}>
            {analysis.recommendation.toUpperCase()}
          </span>
        </div>
      </div>

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
    </motion.div>
  );
}

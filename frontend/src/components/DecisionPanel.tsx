import { motion } from 'framer-motion';
import type { FinalDecision } from '../types';
import { IconTrendUp, IconTrendDown, IconHold } from './Icons';
import { HelpTip } from './HelpTip';
import { HELP_TEXT } from '../helpText';

interface DecisionPanelProps {
  decision: FinalDecision;
  ticker: string;
}

function RiskRewardBar({ entry, stop, target, action }: { entry: number; stop: number; target: number; action: string }) {
  const riskPct = Math.abs(entry - stop) / entry * 100;
  const rewardPct = Math.abs(target - entry) / entry * 100;
  const rr = rewardPct / (riskPct || 1);
  const total = riskPct + rewardPct;

  return (
    <div className="rr-bar">
      <div className="rr-bar__labels">
        <span style={{ color: 'var(--sell)' }}>
          SL ${stop.toFixed(2)} <span className="faint" style={{ fontSize: 10 }}>−{riskPct.toFixed(1)}%</span>
          <HelpTip id="tip-dp-sl" text={HELP_TEXT.stopLoss} />
        </span>
        <span style={{ color: 'rgba(255,255,255,0.8)' }}>ENTRY ${entry.toFixed(2)}</span>
        <span style={{ color: 'var(--buy)' }}>
          TP ${target.toFixed(2)} <span className="faint" style={{ fontSize: 10 }}>+{rewardPct.toFixed(1)}%</span>
          <HelpTip id="tip-dp-tp" text={HELP_TEXT.takeProfit} />
        </span>
      </div>
      <div className="rr-bar__track">
        <motion.div
          className="rr-bar__risk"
          style={{ width: `${(riskPct / total) * 100}%`, background: 'var(--sell)' }}
          initial={{ scaleX: 0, originX: action === 'sell' ? 0 : 1 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.5, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        />
        <motion.div
          className="rr-bar__reward"
          style={{ width: `${(rewardPct / total) * 100}%`, background: 'var(--buy)' }}
          initial={{ scaleX: 0, originX: action === 'sell' ? 1 : 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.6, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        />
        <div className="rr-bar__entry" style={{ left: `${(riskPct / total) * 100}%` }} />
      </div>
      <div className="rr-bar__ratio">
        <span className="faint">Risk / Reward</span>
        <span className="mono" style={{ color: rr >= 2 ? 'var(--buy)' : rr >= 1.5 ? 'var(--hold)' : 'var(--sell)' }}>
          1 : {rr.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function ConvictionMeter({ conviction }: { conviction: string }) {
  const levels = ['low', 'medium', 'high'];
  const idx = levels.indexOf(conviction);

  const activeColor =
    conviction === 'high'
      ? 'var(--buy)'
      : conviction === 'medium'
      ? 'var(--hold)'
      : 'var(--sell)';

  return (
    <div className="conviction-meter">
      {levels.map((lvl, i) => (
        <motion.div
          key={lvl}
          className="conviction-block"
          style={{
            background: i <= idx ? activeColor : 'rgba(255,255,255,0.07)',
            opacity: i <= idx ? 0.3 + i * 0.35 : 0.2,
            flex: 1,
          }}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ delay: 0.3 + i * 0.1, duration: 0.4 }}
        />
      ))}
      <span className="conviction-label" style={{ color: activeColor }}>{conviction.toUpperCase()}</span>
    </div>
  );
}

function ActionIcon({ action }: { action: string }) {
  if (action === 'buy') return <IconTrendUp size={28} color="currentColor" strokeWidth={1.8} />;
  if (action === 'sell') return <IconTrendDown size={28} color="currentColor" strokeWidth={1.8} />;
  return <IconHold size={28} color="currentColor" strokeWidth={1.8} />;
}

/** Detect a backend fallback verdict so we render an "incomplete" state
 * instead of pretending the Coordinator returned a HOLD.
 *
 * We check two signals:
 *   1. confidence === 0  — _default_decision() always returns 0; a real
 *      Coordinator response will be ≥ 1 even when capped.
 *   2. The exact fallback reasoning prefix the backend emits.
 *
 * We deliberately do NOT check for loose substrings like "unavailable"
 * because the Coordinator's OWN reasoning often says "No agents reported
 * unavailable", which is a healthy success message that contains the word.
 */
function isDecisionFallback(decision: FinalDecision): boolean {
  if (decision.confidence === 0) return true;
  const r = (decision.reasoning || '').toLowerCase();
  return (
    r.startsWith('decision unavailable') ||
    r.startsWith('decision error') ||
    r.startsWith('decision failed') ||
    r.includes('one or more agents failed. retry')
  );
}

export function DecisionPanel({ decision, ticker }: DecisionPanelProps) {
  const actionColor = decision.action === 'buy' ? 'var(--buy)' : decision.action === 'sell' ? 'var(--sell)' : 'var(--hold)';
  const actionBg = decision.action === 'buy' ? 'var(--buy-dim)' : decision.action === 'sell' ? 'var(--sell-dim)' : 'var(--hold-dim)';
  const actionBorder = decision.action === 'buy' ? 'rgba(0, 229, 160, 0.35)' : decision.action === 'sell' ? 'rgba(255, 59, 48, 0.35)' : 'rgba(245, 166, 35, 0.35)';

  const horizonLabel =
    decision.time_horizon === 'short-term' ? '1–4 weeks'
    : decision.time_horizon === 'medium-term' ? '1–6 months'
    : '6–24 months';

  const confidenceColor =
    decision.confidence > 75 ? 'var(--buy)' : decision.confidence > 55 ? 'var(--hold)' : 'var(--sell)';

  const fallback = isDecisionFallback(decision);

  return (
    <motion.div
      className="decision-panel glass-2"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Subtle top glow */}
      <div
        className="decision-panel__glow"
        style={{ background: `radial-gradient(ellipse 60% 30% at 50% 0%, ${actionColor}12, transparent)` }}
      />

      <div className="decision-panel__top">
        {/* Action badge */}
        <div className="decision-panel__action-wrap">
          {fallback ? (
            <div className="decision-action decision-action--error">
              <span className="decision-action__icon" aria-hidden="true">⚠</span>
              <span className="decision-action__text">NO DECISION</span>
              <span className="decision-action__ticker">{ticker}</span>
            </div>
          ) : (
            <motion.div
              className="decision-action"
              style={{
                background: actionBg,
                border: `1.5px solid ${actionBorder}`,
                color: '#ffffff',
                boxShadow: `0 0 24px -6px ${actionColor}66`,
              }}
              initial={{ scale: 0.7, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 250, damping: 18, delay: 0.1 }}
            >
              <ActionIcon action={decision.action} />
              <span className="decision-action__text">{decision.action.toUpperCase()}</span>
              <span className="decision-action__ticker">{ticker}</span>
            </motion.div>
          )}

          <div className="decision-panel__meta">
            {fallback ? (
              <>
                <span className="decision-panel__meta-line">
                  <span className="faint">Source:</span>{' '}
                  <strong style={{ color: 'var(--text)' }}>Coordinator Agent</strong>
                </span>
                <span className="decision-panel__meta-sub faint">
                  Coordinator did not respond. Retry the analysis once the rate limit clears.
                </span>
              </>
            ) : (
              <>
                <span className="faint">Final recommendation from</span>
                <strong style={{ color: 'var(--text)' }}> Coordinator Agent</strong>
              </>
            )}
          </div>
        </div>

        {/* Key metrics */}
        <div className="decision-metrics">
          <div className="decision-metric">
            <span className="decision-metric__label">
              CONFIDENCE
              <HelpTip id="tip-dp-confidence" text={HELP_TEXT.confidence} />
            </span>
            <span className="decision-metric__value" style={{ color: fallback ? 'var(--text-3)' : confidenceColor }}>
              {fallback ? '—' : `${decision.confidence}%`}
            </span>
          </div>
          <div className="decision-metric">
            <span className="decision-metric__label">CONVICTION</span>
            {fallback ? <span className="decision-metric__value decision-metric__value--sm" style={{ color: 'var(--text-3)' }}>—</span> : <ConvictionMeter conviction={decision.conviction} />}
          </div>
          <div className="decision-metric">
            <span className="decision-metric__label">
              POSITION SIZE
              <HelpTip id="tip-dp-position" text={HELP_TEXT.position} />
            </span>
            <span className="decision-metric__value" style={{ color: fallback ? 'var(--text-3)' : 'var(--text)' }}>
              {fallback ? '—' : `${(decision.position_size * 100).toFixed(1)}%`}
            </span>
          </div>
          <div className="decision-metric">
            <span className="decision-metric__label">
              TIME HORIZON
              <HelpTip id="tip-dp-horizon" text={HELP_TEXT.horizon} />
            </span>
            <span className="decision-metric__value decision-metric__value--sm" style={{ color: fallback ? 'var(--text-3)' : 'var(--text)' }}>
              {fallback ? '—' : decision.time_horizon.replace('-', ' ').toUpperCase()}
            </span>
            <span className="decision-metric__sub faint">{fallback ? 'No data' : horizonLabel}</span>
          </div>
        </div>
      </div>

      {!fallback && (
        <>
          {/* Price levels */}
          <div className="decision-panel__prices">
            <h3 className="decision-panel__section-title">PRICE TARGETS</h3>
            <RiskRewardBar
              entry={decision.entry_price}
              stop={decision.stop_loss_price}
              target={decision.take_profit_price}
              action={decision.action}
            />
          </div>

          <div className="decision-panel__reasoning">
            <h3 className="decision-panel__section-title">COORDINATOR SYNTHESIS</h3>
            <p className="decision-panel__reasoning-text">{decision.reasoning}</p>
          </div>
        </>
      )}
    </motion.div>
  );
}

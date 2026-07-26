import { motion } from 'framer-motion';
import type { AgentState } from '../types';
import { IconTechnical, IconFundamental, IconSentiment, IconRisk } from './Icons';

interface NeuralOrbProps {
  agent: AgentState;
  index: number;
  total: number;
  isActive: boolean;
}

function AgentIcon({ id, size = 20 }: { id: string; size?: number }) {
  const props = { size, color: 'currentColor', strokeWidth: 1.5 } as const;
  if (id === 'technical') return <IconTechnical {...props} />;
  if (id === 'fundamental') return <IconFundamental {...props} />;
  if (id === 'sentiment') return <IconSentiment {...props} />;
  if (id === 'risk') return <IconRisk {...props} />;
  return null;
}

export function NeuralOrb({ agent, index, total, isActive }: NeuralOrbProps) {
  const angle = (index / total) * 2 * Math.PI - Math.PI / 2;
  const radius = 120;
  const cx = Math.cos(angle) * radius;
  const cy = Math.sin(angle) * radius;

  const isDone = agent.status === 'done' && !!agent.analysis;
  const isRunning = agent.status === 'running';

  // Dynamic functional colors for recommendation highlights
  const rec = agent.analysis?.recommendation;
  const colorToken =
    rec === 'buy'
      ? 'var(--buy)'
      : rec === 'sell'
      ? 'var(--sell)'
      : rec === 'hold'
      ? 'var(--hold)'
      : 'rgba(255, 255, 255, 0.12)';

  const hexBorder = isDone
    ? colorToken
    : isRunning
    ? 'rgba(255, 255, 255, 0.45)'
    : 'rgba(255, 255, 255, 0.12)';

  const hexGlow = isDone
    ? `0 0 20px -2px ${colorToken}40`
    : isRunning
    ? '0 0 16px -6px rgba(255, 255, 255, 0.15)'
    : 'none';

  return (
    /* Outer static HTML div handles absolute positioning & translation centering */
    <div
      className="orb-node-wrap"
      style={{
        left: `calc(50% + ${cx}px)`,
        top: `calc(50% + ${cy}px)`,
        transform: 'translate(-50%, -50%)',
      }}
    >
      {/* Inner motion.div handles scaling/entrance animations without breaking centering translate */}
      <motion.div
        className={`orb-node orb-node--${agent.status}`}
        initial={{ opacity: 0, scale: 0.6 }}
        animate={{ opacity: 1, scale: isActive ? 1.05 : 1 }}
        transition={{ delay: index * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <div style={{ position: 'relative', width: 56, height: 56, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* Hex tile - simple, clean, pulsates when running */}
          <motion.div
            className="orb-node__hex"
            style={{
              width: 52,
              height: 52,
              borderRadius: 10,
              border: `1.5px solid ${hexBorder}`,
              background: isDone ? `${colorToken}06` : 'rgba(255, 255, 255, 0.02)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: isDone ? colorToken : 'rgba(255, 255, 255, 0.7)',
              boxShadow: hexGlow,
              backdropFilter: 'blur(8px)',
              transition: 'border-color 0.4s ease, box-shadow 0.4s ease, color 0.4s ease',
              position: 'relative',
              zIndex: 2,
            }}
            animate={isRunning ? { 
              opacity: [0.4, 1, 0.4],
              scale: [0.97, 1.03, 0.97]
            } : { 
              opacity: 1,
              scale: 1
            }}
            transition={isRunning ? { 
              duration: 1.2, 
              repeat: Infinity,
              ease: 'easeInOut'
            } : {}}
          >
            <div className="orb-node__icon">
              <AgentIcon id={agent.id} size={20} />
            </div>
          </motion.div>
        </div>

        {/* Labels positioned cleanly beneath */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <span className="orb-node__name">
            {agent.name}
          </span>

          {isDone && rec && (
            <motion.span
              className={`orb-node__rec orb-node__rec--${rec}`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              {rec.toUpperCase()}
            </motion.span>
          )}

          {isRunning && (
            <span className="orb-node__scanning">
              SCANNING
            </span>
          )}
        </div>
      </motion.div>
    </div>
  );
}

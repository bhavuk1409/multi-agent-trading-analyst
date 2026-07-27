import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

type EmptyStateVariant = 'idle' | 'running' | 'synthesising';

interface NexusHexLogoProps {
  /** Gradient ID suffix to keep multiple SVGs in the same DOM unique. */
  idSuffix: string;
  /** Stroke color for the inner hex (changes per state). */
  innerStroke?: string;
}

/** Reusable double-hexagon mark used in the header + all three empty states.
 * The two hexagons sit slightly offset, giving the "stamped" look without
 * adding a second gradient stop per call site. */
export function NexusHexLogo({ idSuffix, innerStroke = 'rgba(255,255,255,0.18)' }: NexusHexLogoProps) {
  return (
    <svg width="112" height="112" viewBox="0 0 44 44" fill="none">
      <polygon
        points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
        stroke={`url(#hex-grad-${idSuffix})`}
        strokeWidth="2.4"
        fill="none"
      />
      <polygon
        points="22,8 35,15.5 35,30.8 22,38.3 9,30.8 9,15.5"
        stroke={innerStroke}
        strokeWidth="1"
        fill="none"
      />
      <defs>
        <linearGradient id={`hex-grad-${idSuffix}`} x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="100%" stopColor="#1a1a1a" />
        </linearGradient>
      </defs>
    </svg>
  );
}

interface EmptyStateProps {
  variant: EmptyStateVariant;
  title: string;
  /** Already-JSX description block. Two-line break is the expected shape. */
  description: ReactNode;
  /** Optional row of pills (used in idle state to advertise the agents). */
  agentPills?: string[];
}

/** Empty / loading placeholder used while the user hasn't picked a ticker,
 *  while agents are scanning, and while the Coordinator is synthesising.
 *
 *  The variant flips .pulse-logo to its active form (rotating rings) for
 *  `running`/`synthesising`, and keeps the idle "sonar ping" otherwise. */
export function EmptyState({ variant, title, description, agentPills }: EmptyStateProps) {
  const isActive = variant !== 'idle';

  // Per-variant inner-hex stroke so the synthesising state can tint orange
  // without altering the outer gradient (which stays white-to-black).
  const innerStroke =
    variant === 'synthesising' ? 'rgba(245,166,35,0.30)' : 'rgba(255,255,255,0.20)';

  return (
    <motion.div
      className="empty-state"
      key={`empty-state-${variant}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.97 }}
    >
      <div className="empty-state__inner">
        <div className="empty-state__grid" aria-hidden="true" />
        <div className={`pulse-logo pulse-logo--${variant}`}>
          <NexusHexLogo idSuffix={variant} innerStroke={innerStroke} />
        </div>
        <h2 className="empty-state__title" style={isActive ? { letterSpacing: '0.24em' } : undefined}>
          {title}
        </h2>
        <p className="empty-state__desc">{description}</p>
        {agentPills && agentPills.length > 0 && (
          <div className="empty-state__agents">
            {agentPills.map(name => (
              <div key={name} className="empty-state__agent-pill">
                {name}
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
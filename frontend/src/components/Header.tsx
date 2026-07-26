import { motion } from 'framer-motion';

interface HeaderProps {
  apiUp: boolean | null;
}

export function Header({ apiUp }: HeaderProps) {
  const statusText = apiUp === null ? 'CONNECTING' : apiUp ? 'SYSTEMS ONLINE' : 'DEMO MODE';
  const dotClass = apiUp === null ? 'dot--wait' : apiUp ? 'dot--live' : 'dot--off';

  return (
    <header className="nexus-header">
      <div className="nexus-header__left">
        <div className="nexus-brand">
          <div className="nexus-brand__row">
            <div className="nexus-logo" aria-hidden="true">
              <svg width="22" height="22" viewBox="0 0 44 44" fill="none">
                <polygon
                  points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                  stroke="url(#hex-grad)"
                  strokeWidth="3"
                  fill="none"
                />
                <defs>
                  <linearGradient id="hex-grad" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#ffffff" />
                    <stop offset="100%" stopColor="#3a3a3a" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <h1 className="nexus-title">
              <span className="nexus-title__main">NEXUS</span>
            </h1>
          </div>
          <p className="nexus-byline">Multi-Agent LLM Analysis Platform</p>
        </div>
      </div>

      <div className="nexus-header__right">
        <div className="nexus-status">
          <span className={`dot ${dotClass}`} />
          <span className="nexus-status__text mono">{statusText}</span>
        </div>

        <div className="nexus-badges">
          <span className="nexus-badge">4 AGENTS</span>
          <span className="nexus-badge nexus-badge--accent">LLaMA 3.3</span>
        </div>
      </div>
    </header>
  );
}

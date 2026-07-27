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
              <svg width="24" height="24" viewBox="0 0 44 44" fill="none">
                <polygon
                  points="22,3.5 39,13.3 39,33 22,42.8 5,33 5,13.3"
                  stroke="url(#hex-grad)"
                  strokeWidth="3.2"
                  strokeLinejoin="round"
                  fill="none"
                />
                <polygon
                  points="22,9.5 33.5,16.1 33.5,29.9 22,36.5 10.5,29.9 10.5,16.1"
                  stroke="url(#hex-grad-inner)"
                  strokeWidth="1.8"
                  strokeOpacity="0.45"
                  strokeLinejoin="round"
                  fill="none"
                />
                <defs>
                  <linearGradient id="hex-grad" x1="5" y1="3.5" x2="39" y2="42.8" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#ffffff" />
                    <stop offset="100%" stopColor="#3a3a3a" />
                  </linearGradient>
                  <linearGradient id="hex-grad-inner" x1="10.5" y1="9.5" x2="33.5" y2="36.5" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#ffffff" />
                    <stop offset="100%" stopColor="#444444" />
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
          <span className="nexus-badge">5 AGENTS</span>
          <span className="nexus-badge nexus-badge--accent">LLaMA 3.3</span>
        </div>
      </div>
    </header>
  );
}

/** Premium SaaS-grade SVG icons — all monochrome line icons */

interface IconProps {
  size?: number;
  strokeWidth?: number;
  color?: string;
}

const defaults = { size: 22, strokeWidth: 1.6, color: 'currentColor' };

/** Candlestick / technical chart */
export function IconTechnical({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <line x1="6" y1="2" x2="6" y2="22" />
      <rect x="3" y="6" width="6" height="8" rx="0.5" />
      <line x1="18" y1="2" x2="18" y2="22" />
      <rect x="15" y="10" width="6" height="6" rx="0.5" />
      <line x1="12" y1="5" x2="12" y2="19" strokeDasharray="2 2" opacity="0.4" />
    </svg>
  );
}

/** Rising bar chart / fundamental */
export function IconFundamental({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <line x1="3" y1="20" x2="21" y2="20" />
      <rect x="4" y="14" width="4" height="6" rx="0.5" />
      <rect x="10" y="9" width="4" height="11" rx="0.5" />
      <rect x="16" y="4" width="4" height="16" rx="0.5" />
      <polyline points="5,12 11,7 17,2" opacity="0.4" strokeDasharray="2 2" />
    </svg>
  );
}

/** Pulse / sentiment waveform */
export function IconSentiment({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="2,12 6,12 8,5 10,19 12,9 14,15 16,12 22,12" />
    </svg>
  );
}

/** Shield / risk manager */
export function IconRisk({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.5C16.5 22.15 20 17.25 20 12V6L12 2z" />
      <polyline points="9,12 11,14 15,10" />
    </svg>
  );
}

/** Coordinator / final decision node */
export function IconCoordinator({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <circle cx="12" cy="12" r="8" opacity="0.4" />
      <line x1="12" y1="2" x2="12" y2="4" />
      <line x1="12" y1="20" x2="12" y2="22" />
      <line x1="2" y1="12" x2="4" y2="12" />
      <line x1="20" y1="12" x2="22" y2="12" />
    </svg>
  );
}

/** Activity / live status */
export function IconActivity({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
    </svg>
  );
}

/** Download / export */
export function IconDownload({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="7,10 12,15 17,10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

/** Refresh / new analysis */
export function IconRefresh({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="1,4 1,10 7,10" />
      <path d="M3.51 15a9 9 0 1 0 .49-4.1" />
    </svg>
  );
}

/** Play / run analysis */
export function IconRun({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polygon points="10,8 16,12 10,16" fill={color} stroke="none" />
    </svg>
  );
}

/** News / article */
export function IconNews({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 22h16a2 2 0 002-2V4a2 2 0 00-2-2H8a2 2 0 00-2 2v16a2 2 0 01-2 2zm0 0a2 2 0 01-2-2v-9c0-1.1.9-2 2-2h2" />
      <line x1="18" y1="14" x2="10" y2="14" />
      <line x1="18" y1="10" x2="10" y2="10" />
      <line x1="14" y1="6" x2="10" y2="6" />
    </svg>
  );
}

/** Chevron right arrow */
export function IconChevronRight({ size = 14, strokeWidth = 1.8, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9,18 15,12 9,6" />
    </svg>
  );
}

/** Trending up / buy signal */
export function IconTrendUp({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23,6 13.5,15.5 8.5,10.5 1,18" />
      <polyline points="17,6 23,6 23,12" />
    </svg>
  );
}

/** Trending down / sell signal */
export function IconTrendDown({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23,18 13.5,8.5 8.5,13.5 1,6" />
      <polyline points="17,18 23,18 23,12" />
    </svg>
  );
}

/** Minus / hold signal */
export function IconHold({ size = defaults.size, strokeWidth = defaults.strokeWidth, color = defaults.color }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round">
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

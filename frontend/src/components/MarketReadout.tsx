import { motion } from 'framer-motion';
import type { MarketData } from '../types';
import { HelpTip } from './HelpTip';
import { HELP_TEXT } from '../helpText';

interface MarketReadoutProps {
  ticker: string;
  data: MarketData;
  history?: { date: string; close: number }[];
}

function PriceChart({ history }: { history?: { date: string; close: number }[] }) {
  if (!history || history.length === 0) return null;

  const prices = history.map(h => h.close);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice || 1;

  const width = 1100;
  const height = 160;
  const padding = 16;
  const labelGutter = 60;  // left margin for Y-axis tick labels
  const rightGutter = 80;  // right margin for the latest-price label

  const points = history.map((pt, i) => {
    const x = labelGutter + (i / Math.max(history.length - 1, 1)) * (width - labelGutter - rightGutter);
    const y = padding + (1 - (pt.close - minPrice) / priceRange) * (height - padding * 2);
    return { x, y };
  });

  const linePath = points.reduce((path, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.y}` : `${path} L ${pt.x} ${pt.y}`;
  }, '');

  const firstPt = points[0];
  const lastPt = points[points.length - 1];
  const areaPath = `${linePath} L ${lastPt.x} ${height - padding} L ${firstPt.x} ${height - padding} Z`;

  // 30-day return determines the gradient + stroke tint.
  const firstClose = prices[0];
  const lastClose = prices[prices.length - 1];
  const isPositive = lastClose >= firstClose;
  const upStroke = 'rgba(0, 229, 160, 0.85)';
  const downStroke = 'rgba(255, 59, 48, 0.85)';
  const strokeColor = isPositive ? upStroke : downStroke;

  // 4 evenly-spaced Y-axis tick labels (rounded to 2 decimals).
  const ticks = [0, 0.33, 0.66, 1].map((r) => {
    const value = maxPrice - r * priceRange;
    return { value, y: padding + r * (height - padding * 2) };
  });

  return (
    <div className="price-chart">
      <div className="price-chart__meta">
        <span className="price-chart__label faint">30D PRICE TRAJECTORY</span>
        <span
          className="price-chart__range mono"
          style={{ color: isPositive ? 'var(--buy)' : 'var(--sell)' }}
        >
          {isPositive ? '+' : ''}{(((lastClose - firstClose) / firstClose) * 100).toFixed(2)}%
        </span>
      </div>
      <div className="price-chart__svg-container">
        <svg className="price-chart__svg" viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <linearGradient id="chart-area-grad-up" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(0, 229, 160, 0.28)" />
              <stop offset="100%" stopColor="rgba(0, 229, 160, 0.0)" />
            </linearGradient>
            <linearGradient id="chart-area-grad-down" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(255, 59, 48, 0.28)" />
              <stop offset="100%" stopColor="rgba(255, 59, 48, 0.0)" />
            </linearGradient>
          </defs>

          {/* Y-axis tick labels */}
          {ticks.map((t, i) => (
            <text
              key={i}
              x={labelGutter - 6}
              y={t.y + 3}
              textAnchor="end"
              fontFamily="var(--font-mono)"
              fontSize="9"
              fill="rgba(255, 255, 255, 0.45)"
            >
              ${t.value.toFixed(2)}
            </text>
          ))}

          {/* Horizontal Grid lines */}
          {ticks.map((t, i) => (
            <line
              key={i}
              x1={labelGutter}
              y1={t.y}
              x2={width - rightGutter}
              y2={t.y}
              stroke="rgba(255, 255, 255, 0.04)"
              strokeDasharray="2 4"
            />
          ))}

          {/* Area fill */}
          <path
            d={areaPath}
            fill={isPositive ? 'url(#chart-area-grad-up)' : 'url(#chart-area-grad-down)'}
          />

          {/* Sparkline glow */}
          <path
            d={linePath}
            fill="none"
            stroke={strokeColor}
            strokeOpacity="0.25"
            strokeWidth="3.5"
            style={{ filter: 'blur(2px)' }}
          />

          {/* Sparkline stroke */}
          <path
            d={linePath}
            fill="none"
            stroke={strokeColor}
            strokeWidth="1.5"
          />

          {/* Latest-price dashed line */}
          <line
            x1={labelGutter}
            y1={lastPt.y}
            x2={width - rightGutter}
            y2={lastPt.y}
            stroke="rgba(255, 255, 255, 0.45)"
            strokeDasharray="3 3"
          />

          {/* Latest price label (right margin) */}
          <text
            x={width - rightGutter + 6}
            y={lastPt.y + 3}
            textAnchor="start"
            fontFamily="var(--font-mono)"
            fontSize="9"
            fontWeight="700"
            fill="#ffffff"
          >
            ${lastClose.toFixed(2)}
          </text>
        </svg>
      </div>
    </div>
  );
}

interface StatBlockProps {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  delay?: number;
  /** When set, renders a `?` tooltip next to the label with this HELP_TEXT key. */
  tipKey?: keyof typeof HELP_TEXT;
}

function StatBlock({ label, value, sub, color, delay = 0, tipKey }: StatBlockProps) {
  return (
    <motion.div
      className="stat-block"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <span className="stat-block__label">
        {label}
        {tipKey && <HelpTip id={`tip-${tipKey}`} text={HELP_TEXT[tipKey]} />}
      </span>
      <span className="stat-block__value" style={{ color: color ?? 'var(--text)' }}>
        {value}
      </span>
      {sub && <span className="stat-block__sub">{sub}</span>}
    </motion.div>
  );
}

// Functional color mapping: green = buy/positive, red = sell/negative, orange = hold/caution
function signalColor(positive: boolean): string {
  return positive ? 'var(--buy)' : 'var(--sell)';
}

function rsiColor(rsi: number): string {
  if (rsi < 30) return 'var(--buy)';   // oversold  → green (buying opportunity)
  if (rsi > 70) return 'var(--sell)';  // overbought → red (selling indicator)
  return 'var(--hold)';                // neutral → orange
}

function bbColor(pct: number): string {
  if (pct < 20) return 'var(--buy)';
  if (pct > 80) return 'var(--sell)';
  return 'var(--text-2)';
}

function RSIMeter({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value));
  const color = rsiColor(value);
  const label = value < 30 ? 'OVERSOLD' : value > 70 ? 'OVERBOUGHT' : 'NEUTRAL';

  return (
    <div className="rsi-meter">
      <div className="rsi-meter__header">
        <span className="rsi-meter__label faint">RSI · 14</span>
        <span className="rsi-meter__value mono" style={{ color }}>{value.toFixed(1)}</span>
      </div>
      <div className="rsi-meter__track">
        <div className="rsi-meter__zones">
          <div className="rsi-zone rsi-zone--buy" />
          <div className="rsi-zone rsi-zone--neutral" />
          <div className="rsi-zone rsi-zone--sell" />
        </div>
        <motion.div
          className="rsi-meter__needle"
          style={{ left: `${pct}%`, background: color }}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        />
      </div>
      <div className="rsi-meter__footer">
        <span className="rsi-zone-label faint">OVERSOLD</span>
        <span className="rsi-zone-label faint mono" style={{ color }}>{label}</span>
        <span className="rsi-zone-label faint" style={{ textAlign: 'right' }}>OVERBOUGHT</span>
      </div>
    </div>
  );
}

function BBBar({ position }: { position: number }) {
  const pct = Math.min(100, Math.max(0, position * 100));
  const color = bbColor(pct);

  return (
    <div className="bb-bar">
      <div className="bb-bar__header">
        <span className="faint">
          BB POSITION
          <HelpTip id="tip-bb" text={HELP_TEXT.bb} />
        </span>
        <span className="mono" style={{ color }}>{(position * 100).toFixed(1)}%</span>
      </div>
      <div className="bb-bar__track">
        <motion.div
          className="bb-bar__fill"
          style={{ width: `${pct}%`, background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ delay: 0.5, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        />
        <div className="bb-bar__mid" />
      </div>
    </div>
  );
}

export function MarketReadout({ ticker, data, history }: MarketReadoutProps) {
  const volumeM = (data.volume / 1_000_000).toFixed(1);

  return (
    <motion.div
      className="market-readout glass"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="market-readout__header">
        <div>
          <div className="market-readout__ticker">{ticker}</div>
          <div className="market-readout__price">${data.close.toFixed(2)}</div>
        </div>
        <div className="market-readout__badge">LIVE DATA</div>
      </div>

      <PriceChart history={history} />

      <div className="market-readout__grid">
        <StatBlock
          label="RSI"
          value={data.rsi.toFixed(1)}
          sub={data.rsi < 30 ? 'Oversold' : data.rsi > 70 ? 'Overbought' : 'Neutral'}
          color={rsiColor(data.rsi)}
          delay={0.1}
          tipKey="rsi"
        />
        <StatBlock
          label="MACD"
          value={data.macd.toFixed(3)}
          color={signalColor(data.macd >= 0)}
          delay={0.15}
          tipKey="macd"
        />
        <StatBlock
          label="SMA 20"
          value={`$${data.sma_20.toFixed(2)}`}
          delay={0.2}
          tipKey="sma20"
        />
        <StatBlock
          label="MOMENTUM"
          value={data.momentum.toFixed(2)}
          color={signalColor(data.momentum >= 0)}
          delay={0.25}
          tipKey="momentum"
        />
        <StatBlock
          label="VOLUME"
          value={`${volumeM}M`}
          sub={`${data.volume_ratio.toFixed(2)}x avg`}
          delay={0.3}
          tipKey="volume"
        />
      </div>

      <RSIMeter value={data.rsi} />
      <BBBar position={data.bb_position} />
    </motion.div>
  );
}

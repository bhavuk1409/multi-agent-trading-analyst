/** Single source of truth for tooltip copy on every stat label. */

export const HELP_TEXT = {
  rsi:        'RSI (14-day) — below 30 = oversold, above 70 = overbought.',
  macd:       'MACD — momentum. Positive = bullish, negative = bearish.',
  sma20:      '20-day simple moving average. Price above = short-term uptrend.',
  momentum:   'Price change over the lookback window.',
  volume:     "Today's volume. Sub-line shows vs. 20-day average.",
  bb:         'Where close sits inside its 20-day Bollinger envelope (0%=lower, 100%=upper).',
  confidence: "This agent's stated confidence in its own recommendation.",
  position:   'Suggested fraction of capital to deploy (0–1.0).',
  stopLoss:   'Price level to exit if the trade moves against you.',
  takeProfit: 'Price level to exit with a profit.',
  horizon:    'Expected holding period for the trade.',
} as const;
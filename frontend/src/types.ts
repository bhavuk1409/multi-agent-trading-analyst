export type Ticker = 'AAPL' | 'GOOGL' | 'MSFT' | 'TSLA' | 'NVDA';

export type Action     = 'buy' | 'sell' | 'hold';
export type Conviction = 'low' | 'medium' | 'high';
export type TimeHorizon = 'short-term' | 'medium-term' | 'long-term';

export interface WatchlistItem {
  ticker:      Ticker;
  price:       number;   // e.g. 333.02
  change:      number;   // e.g. -4.12
  change_pct:  number;   // e.g. -1.22
  is_positive: boolean;
}

export interface AgentAnalysis {
  recommendation: Action;
  confidence:     number;   // 0-100
  reasoning:      string;
}

export interface FinalDecision {
  action:           Action;
  position_size:    number;   // 0.0 – 1.0
  confidence:       number;   // 0-100
  conviction:       Conviction;
  entry_price:      number;
  stop_loss_price:  number;
  take_profit_price: number;
  time_horizon:     TimeHorizon;
  reasoning:        string;
}

export interface MarketData {
  close:        number;
  volume:       number;
  rsi:          number;
  macd:         number;
  sma_20:       number;
  bb_position:  number;
  volume_ratio: number;
  momentum:     number;
}

export interface NewsArticle {
  title:          string;
  url:            string;
  published_date: string;
  summary:        string;
  source:         string;
  sentiment:      'positive' | 'negative' | 'neutral';
}

export interface AnalysisResults {
  technical_analysis?:   AgentAnalysis;
  fundamental_analysis?: AgentAnalysis;
  sentiment_analysis?:   AgentAnalysis;
  risk_analysis?:        AgentAnalysis;
  final_decision?:       FinalDecision;
  ticker?:               string;
  date?:                 string;
  market_data?:          MarketData;
  news?:                 NewsArticle[];
  price_history?:        { date: string; close: number }[];
}

export type AgentStatus = 'idle' | 'running' | 'done' | 'error';

export interface AgentState {
  id:           string;
  name:         string;
  icon:         string;
  color:        string;
  status:       AgentStatus;
  analysis?:    AgentAnalysis;
  description:  string;
}

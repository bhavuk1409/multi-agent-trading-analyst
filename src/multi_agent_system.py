"""
Advanced Multi-Agent System for Trading Decisions
==================================================
Uses the official lightweight `openai` SDK pointing to Groq's API endpoint.
Direct REST calls with `response_format={"type": "json_object"}` ensure fast,
reliable structured JSON output with zero heavy LangChain package overhead.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


def _clean_json_text(text: str) -> str:
    """Extract raw JSON text if response contains preamble or markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return text


class AdvancedMultiAgentSystem:
    """
    Multi-agent trading system using Groq LLaMA models.
    Each agent specializes in one analysis domain:
      1. Technical Analyst
      2. Fundamental Analyst
      3. Sentiment Analyst
      4. Risk Manager
      5. Coordinator (Final Synthesizer)
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        agent_config: Dict[str, Any] = None,
    ):
        self.model_name = model
        self.temperature = temperature
        self.agent_config = agent_config or {
            "technical_analyst":   {"enabled": True, "weight": 0.25},
            "fundamental_analyst": {"enabled": True, "weight": 0.25},
            "sentiment_analyst":   {"enabled": True, "weight": 0.25},
            "risk_manager":        {"enabled": True, "weight": 0.25},
        }

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

        logger.info(f"✓ Multi-agent system initialized with Groq model: {model}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Execute a single LLM request to Groq expecting structured JSON output."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        cleaned = _clean_json_text(content)
        return json.loads(cleaned)

    async def _acall_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Async variant of _call_llm — used for parallel agent execution."""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        cleaned = _clean_json_text(content)
        return json.loads(cleaned)

    async def _run_agent(self, cfg_key: str, sys_p: str, usr_p: str) -> Dict[str, Any]:
        """Run a single agent, honoring the `enabled` flag and falling back to
        _default_analysis() on any exception so that asyncio.gather() never sees
        a raised task (one agent's failure cannot break the others)."""
        if not self.agent_config.get(cfg_key, {}).get("enabled", True):
            return self._default_analysis()
        try:
            return await self._acall_llm(sys_p, usr_p)
        except Exception as e:
            logger.error(f"{cfg_key} error: {e}")
            return self._default_analysis()

    def analyze(
        self,
        ticker: str,
        date: str,
        market_data: Dict[str, Any],
        news: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run multi-agent analysis pipeline.

        Returns dict containing:
          - technical_analysis
          - fundamental_analysis
          - sentiment_analysis
          - risk_analysis
          - final_decision
        """
        logger.info(f"🤖 Running multi-agent analysis for {ticker} on {date}")
        results = {}

        # Run the four specialist agents concurrently via asyncio.gather.
        # Each agent has its own try/except inside _run_agent, so a single
        # failure surfaces as _default_analysis() rather than cancelling siblings.
        agent_specs = [
            (
                "technical_analyst",
                (
                    "You are an expert technical analyst for equities. Interpret the indicator snapshot using these rules:\n"
                    "  - RSI: <30 oversold (buy), >70 overbought (sell), 30–70 trend-following (use other signals).\n"
                    "  - MACD histogram positive+rising → bullish; negative+falling → bearish.\n"
                    "  - Bollinger band position <0.2 = near lower support (buy), >0.8 = near upper resistance (sell).\n"
                    "  - SMA-20 vs SMA-50: SMA-20 above SMA-50 = bullish trend; below = bearish.\n"
                    "  - Volume ratio >1.5 confirms breakouts; <0.5 suggests weak moves.\n"
                    "  - Compare close_prev_1 / close_prev_5 / close_prev_20 to detect SMA crossovers and momentum changes.\n"
                    "  - HV-14 (annualised vol %): quote this in your reasoning when it affects confidence.\n"
                    "Return ONLY a JSON object (no prose):\n"
                    '{{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<3-5 sentences citing specific numbers that drove your call>"}}'
                ),
                f"Ticker: {ticker}\nDate: {date}\nTechnical Indicators:\n{self._format_technical(market_data)}",
            ),
            (
                "fundamental_analyst",
                (
                    "You are an expert fundamental analyst. Interpret the metrics using these rules:\n"
                    "  - P/E trailing: <15 undervalued, 15–25 fair, >25 expensive (vs SPY ~22).\n"
                    "  - P/E forward vs trailing: forward << trailing → earnings expected to grow.\n"
                    "  - Dividend yield: >3% income play, <1% growth play. 0 = no dividend.\n"
                    "  - 52-week position ((price - low) / (high - low)): close to 0 = value zone, close to 1 = momentum zone.\n"
                    "  - Short interest >10% = bearish overhang; <3% = clean.\n"
                    "  - Beta >1.3 = amplifies market; <0.7 = defensive.\n"
                    "  - target_mean_price vs current price = analyst-implied upside/downside.\n"
                    "If a metric is null/absent, say so explicitly in reasoning — DO NOT invent values.\n"
                    "Return ONLY a JSON object:\n"
                    '{{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<3-5 sentences citing specific numbers that drove your call>"}}'
                ),
                f"Ticker: {ticker}\nDate: {date}\nFundamentals:\n{self._format_fundamental(market_data)}",
            ),
            (
                "sentiment_analyst",
                (
                    "You are an expert sentiment analyst. For each news article below, classify the headline + summary as positive, negative, or neutral for {ticker}. Weight recent items more heavily. Aggregate and decide:\n"
                    "  - Majority negative and no positive counter-evidence → sell.\n"
                    "  - Majority positive and no negative counter-evidence → buy.\n"
                    "  - Mixed or unclear → hold.\n"
                    "Confidence reflects how strongly the headline-set leans one way.\n"
                    "Return ONLY a JSON object:\n"
                    '{{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<aggregate sentiment, citing 2-3 headlines by source and date>"}}'
                ).format(ticker=ticker),
                f"Ticker: {ticker}\nDate: {date}\nRecent News Articles (newest first):\n{self._format_news(news)}",
            ),
            (
                "risk_manager",
                (
                    "You are an expert risk manager. Interpret the metrics using these rules:\n"
                    "  - HV-14 (annualised vol %): >40% = high risk, 20–40% = moderate, <15% = low.\n"
                    "  - ATR-14 (price units): 2 × ATR is a typical stop distance.\n"
                    "  - Max drawdown: <-15% recent pain = elevated risk; <-30% = severe.\n"
                    "  - Beta >1.3 amplifies market; <0.7 defensive.\n"
                    "Recommendation meaning: 'buy' = add risk, 'sell' = reduce, 'hold' = keep current exposure.\n"
                    "Confidence is INVERSELY related to volatility (high vol = lower confidence in any directional call).\n"
                    "Return ONLY a JSON object:\n"
                    '{{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<3-5 sentences citing specific numbers>"}}'
                ),
                f"Ticker: {ticker}\nDate: {date}\nRisk Snapshot:\n{self._format_risk(market_data)}",
            ),
        ]
        result_keys = [
            "technical_analysis",
            "fundamental_analysis",
            "sentiment_analysis",
            "risk_analysis",
        ]

        async def _gather_agents() -> list[Dict[str, Any]]:
            tasks = [self._run_agent(cfg_key, sys_p, usr_p)
                     for cfg_key, sys_p, usr_p in agent_specs]
            return await asyncio.gather(*tasks)

        gathered = asyncio.run(_gather_agents())
        for key, payload in zip(result_keys, gathered):
            results[key] = payload
        logger.info("  ✓ All 4 specialist agents complete (parallel)")

        # 5. Coordinator Decision
        try:
            current_price = float(market_data.get("close", 0))
            atr           = float(market_data.get("atr_14", current_price * 0.02) or current_price * 0.02)

            # Pre-compute a numeric weighted signal so the LLM has a concrete
            # starting point instead of guessing from the [weight=0.25] labels.
            cfg_map = {
                "technical_analysis":   "technical_analyst",
                "fundamental_analysis": "fundamental_analyst",
                "sentiment_analysis":   "sentiment_analyst",
                "risk_analysis":        "risk_manager",
            }
            raw_weights = {k: float(self.agent_config.get(v, {}).get("weight", 0.25))
                          for k, v in cfg_map.items()}
            w_sum = sum(raw_weights.values()) or 1.0
            norm_weights = {k: v / w_sum for k, v in raw_weights.items()}

            direction = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
            weighted_signal = sum(
                direction.get((results.get(k) or {}).get("recommendation", "hold"), 0.0)
                * float((results.get(k) or {}).get("confidence", 0)) / 100.0
                * norm_weights[k]
                for k in cfg_map
            )
            degraded_count = sum(
                1 for k in cfg_map
                if (results.get(k) or {}).get("degraded") or
                   "unavailable" in (results.get(k) or {}).get("reasoning", "").lower()
            )

            sys_p = (
                "You are the head trader coordinating the four specialist analyses below.\n"
                "Follow these synthesis rules strictly:\n"
                "  - If 3+ agents say 'buy' but the risk agent says 'sell' with confidence >70, default to 'hold'.\n"
                "  - If only 1 of 4 agents has confidence >70, cap final confidence at 60 and prefer 'hold'.\n"
                "  - Position size: Kelly-lite = (confidence/100) * 0.25, clamped to [0.02, 0.30].\n"
                "  - Stop-loss: entry - max(2 × ATR-14, 5% of entry) — the wider of the two.\n"
                "  - Take-profit: must keep risk:reward >= 1.5, anchored from the stop distance.\n"
                "  - Time horizon: short-term if momentum dominates; long-term if fundamental thesis.\n"
                "  - Use the pre-computed 'Weighted signal' below as the starting point, then justify.\n"
                f"  - {degraded_count} of 4 agents reported 'unavailable' — knock final confidence down by 10×{degraded_count} points and mention it in reasoning.\n"
                "Return ONLY a JSON object (no prose):\n"
                "{\n"
                '  "action": "buy|sell|hold",\n'
                '  "position_size": <0.02-0.30>,\n'
                '  "confidence": <0-100>,\n'
                '  "conviction": "low|medium|high",\n'
                '  "entry_price": <number>,\n'
                '  "stop_loss_price": <number>,\n'
                '  "take_profit_price": <number>,\n'
                '  "time_horizon": "short-term|medium-term|long-term",\n'
                '  "reasoning": "<short consolidated paragraph citing the agents and the rules you applied>"\n'
                "}"
            )
            usr_p = (
                f"Ticker: {ticker}\n"
                f"Date: {date}\n"
                f"Current Price: ${current_price:.2f}\n"
                f"ATR-14: ${atr:.2f}  (suggested 2× ATR stop = ${(2 * atr):.2f})\n"
                f"Pre-computed weighted signal: {weighted_signal:+.3f} (range [-1, +1])\n\n"
                f"Agent Analyses (relative weights shown, total 1.0):\n"
                f"{self._format_agent_results(results, include_weights=True)}"
            )
            results["final_decision"] = self._call_llm(sys_p, usr_p)
            logger.info("  ✓ Final decision made")
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            results["final_decision"] = self._default_decision(market_data.get("close", 0))

        return results

    def _format_technical(self, data: Dict[str, Any]) -> str:
        """Technical indicator block — gives the Technical analyst a multi-day view."""
        close       = data.get('close', 'N/A')
        prev_1      = data.get('close_prev_1', 'N/A')
        prev_5      = data.get('close_prev_5', 'N/A')
        prev_20     = data.get('close_prev_20', 'N/A')
        try:
            cross_5_20  = "BULLISH (momentum > 20d)" if float(prev_5) > float(prev_20) else "BEARISH (momentum < 20d)"
        except Exception:
            cross_5_20 = "N/A"
        return (
            f"- Close: ${close}\n"
            f"- RSI-14 (Wilder): {data.get('rsi', 'N/A')}\n"
            f"- MACD line: {data.get('macd', 'N/A')}  |  signal: {data.get('macd_signal', 'N/A')}  |  histogram: {data.get('macd_hist', 'N/A')}\n"
            f"- SMA-20: {data.get('sma_20', 'N/A')}  |  SMA-50: {data.get('sma_50', 'N/A')}\n"
            f"- Bollinger band position: {data.get('bb_position', 'N/A')}  (0=lower band, 1=upper band)\n"
            f"- Volume ratio (vs 20d avg): {data.get('volume_ratio', 'N/A')}\n"
            f"- Momentum (10d % change): {data.get('momentum', 'N/A')}\n"
            f"- HV-14 (annualised vol %): {data.get('hv_14', 'N/A')}\n"
            f"- 5d-close vs 20d-close: {cross_5_20}\n"
            f"- Close prev 1d: ${prev_1}  |  prev 5d: ${prev_5}  |  prev 20d: ${prev_20}\n"
        )

    def _format_fundamental(self, data: Dict[str, Any]) -> str:
        """Fundamentals block — sourced from yfinance `.info`."""
        def _fmt(key, unit=""):
            v = data.get(key)
            if v is None: return "N/A"
            try:    return f"{float(v):.2f}{unit}"
            except: return str(v)

        pos_52w = "N/A"
        try:
            low  = float(data.get("52w_low"))
            high = float(data.get("52w_high"))
            cur  = float(data.get("close"))
            pos_52w = f"{max(0.0, min(1.0, (cur - low) / (high - low) if high > low else 0)):.2f} (0=at low, 1=at high)"
        except Exception:
            pass

        short_pct = data.get("short_pct")
        try:
            short_pct = f"{float(short_pct) * 100:.1f}%" if short_pct is not None else "N/A"
        except Exception:
            short_pct = "N/A"

        target_upside = "N/A"
        try:
            tgt = float(data.get("target_mean_price"))
            cur = float(data.get("close"))
            target_upside = f"{(tgt - cur) / cur * 100:+.1f}% vs current"
        except Exception:
            pass

        return (
            f"- Sector / Industry: {data.get('sector', 'N/A')} / {data.get('industry', 'N/A')}\n"
            f"- Market cap: {_fmt('market_cap')}\n"
            f"- P/E (trailing): {_fmt('pe_trailing')}  |  P/E (forward): {_fmt('pe_forward')}\n"
            f"- EPS (trailing): {_fmt('eps_trailing')}  |  EPS (forward): {_fmt('eps_forward')}\n"
            f"- Dividend yield: {_fmt('dividend_yield', '%')}\n"
            f"- Beta: {_fmt('beta')}\n"
            f"- 52-week high/low: {_fmt('52w_high')} / {_fmt('52w_low')}  (position: {pos_52w})\n"
            f"- Short interest (% of float): {short_pct}\n"
            f"- Analyst mean target: {_fmt('target_mean_price')}  ({target_upside})\n"
            f"- Analyst targets: low {_fmt('target_low_price')} / high {_fmt('target_high_price')}\n"
            f"- Current price: ${data.get('close', 'N/A')}\n"
        )

    def _format_risk(self, data: Dict[str, Any]) -> str:
        """Risk block — uses real volatility / ATR / drawdown, not RSI mislabeled."""
        def _fmt(key, unit=""):
            v = data.get(key)
            if v is None: return "N/A"
            try:    return f"{float(v):.2f}{unit}"
            except: return str(v)

        return (
            f"- Current price: ${data.get('close', 'N/A')}\n"
            f"- HV-14 (annualised vol %): {_fmt('hv_14', '%')}\n"
            f"- ATR-14 (price units): {_fmt('atr_14')}\n"
            f"- Max drawdown (full history): {_fmt('drawdown', '%')}\n"
            f"- Beta: {_fmt('beta')}\n"
            f"- Volume ratio (vs 20d avg): {_fmt('volume_ratio')}\n"
        )

    def _format_news(self, news: List[Dict[str, Any]]) -> str:
        """News block — pass full summaries (newest first), not just titles."""
        if not news:
            return "No recent news available."
        lines = []
        # Cap at 8 to keep prompt size reasonable while preserving breadth.
        for item in news[:8]:
            date = item.get("published_date", "unknown date")
            source = item.get("source", "Unknown")
            title = item.get("title", "N/A")
            summary = (item.get("summary") or "").strip()
            if summary and summary != "No summary available.":
                lines.append(
                    f"- [{date} · {source}] {title}\n"
                    f"    Summary: {summary[:280]}"
                )
            else:
                lines.append(f"- [{date} · {source}] {title}")
        return "\n".join(lines)

    def _format_agent_results(self, results: Dict[str, Any], include_weights: bool = False) -> str:
        cfg_map = {
            "technical_analysis":   "technical_analyst",
            "fundamental_analysis": "fundamental_analyst",
            "sentiment_analysis":   "sentiment_analyst",
            "risk_analysis":        "risk_manager",
        }
        weights = {k: float(self.agent_config.get(v, {}).get("weight", 0.25))
                   for k, v in cfg_map.items()}
        total = sum(weights.values()) or 1.0
        lines = []
        for k in ["technical_analysis", "fundamental_analysis", "sentiment_analysis", "risk_analysis"]:
            if k in results:
                a = results[k]
                header = k.replace("_", " ").title()
                if include_weights:
                    header += f"  [weight={weights[k] / total:.2f}]"
                lines.append(
                    f"{header}:\n"
                    f"- Recommendation: {a.get('recommendation', 'N/A')}\n"
                    f"- Confidence: {a.get('confidence', 0)}%\n"
                    f"- Reasoning: {a.get('reasoning', 'N/A')}\n"
                )
        return "\n".join(lines)

    def _default_analysis(self) -> Dict[str, Any]:
        """Returned when an agent's LLM call fails. Confidence is 0 so the
        coordinator's weighted-signal is dragged toward zero, and `degraded`
        is surfaced to the user."""
        return {
            "recommendation": "hold",
            "confidence": 0,
            "reasoning": "Analysis unavailable — Groq API error or rate-limit. Retry shortly.",
            "degraded": True,
        }

    def _default_decision(self, current_price: float) -> Dict[str, Any]:
        p = float(current_price) if current_price else 100.0
        return {
            "action": "hold",
            "position_size": 0.0,
            "confidence": 0,
            "conviction": "low",
            "entry_price": p,
            "stop_loss_price": round(p * 0.95, 2),
            "take_profit_price": round(p * 1.05, 2),
            "time_horizon": "medium-term",
            "reasoning": "Decision unavailable — one or more agents failed. Retry shortly.",
            "degraded": True,
        }
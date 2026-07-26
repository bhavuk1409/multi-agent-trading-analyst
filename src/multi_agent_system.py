"""
Advanced Multi-Agent System for Trading Decisions
==================================================
Uses the official lightweight `openai` SDK pointing to Groq's API endpoint.
Direct REST calls with `response_format={"type": "json_object"}` ensure fast,
reliable structured JSON output with zero heavy LangChain package overhead.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

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

        # 1. Technical Analysis
        if self.agent_config.get("technical_analyst", {}).get("enabled", True):
            try:
                sys_p = (
                    "You are a technical analyst specializing in stock chart patterns and technical indicators.\n"
                    "Analyze the technical indicators and return ONLY a JSON object:\n"
                    '{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<explanation>"}'
                )
                usr_p = f"Ticker: {ticker}\nDate: {date}\nTechnical Indicators:\n{self._format_technical(market_data)}"
                results["technical_analysis"] = self._call_llm(sys_p, usr_p)
                logger.info("  ✓ Technical analysis complete")
            except Exception as e:
                logger.error(f"Technical analysis error: {e}")
                results["technical_analysis"] = self._default_analysis()

        # 2. Fundamental Analysis
        if self.agent_config.get("fundamental_analyst", {}).get("enabled", True):
            try:
                sys_p = (
                    "You are a fundamental analyst specializing in company valuation and market conditions.\n"
                    "Analyze the market metrics and return ONLY a JSON object:\n"
                    '{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<explanation>"}'
                )
                usr_p = f"Ticker: {ticker}\nDate: {date}\nMarket Data:\n{self._format_fundamental(market_data)}"
                results["fundamental_analysis"] = self._call_llm(sys_p, usr_p)
                logger.info("  ✓ Fundamental analysis complete")
            except Exception as e:
                logger.error(f"Fundamental analysis error: {e}")
                results["fundamental_analysis"] = self._default_analysis()

        # 3. Sentiment Analysis
        if self.agent_config.get("sentiment_analyst", {}).get("enabled", True):
            try:
                sys_p = (
                    "You are a sentiment analyst specializing in news processing and market sentiment.\n"
                    "Analyze recent news articles and return ONLY a JSON object:\n"
                    '{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<explanation>"}'
                )
                usr_p = f"Ticker: {ticker}\nDate: {date}\nRecent News Headlines:\n{self._format_news(news)}"
                results["sentiment_analysis"] = self._call_llm(sys_p, usr_p)
                logger.info("  ✓ Sentiment analysis complete")
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                results["sentiment_analysis"] = self._default_analysis()

        # 4. Risk Manager Assessment
        if self.agent_config.get("risk_manager", {}).get("enabled", True):
            try:
                sys_p = (
                    "You are a risk manager assessing portfolio volatility, beta, and downside risk.\n"
                    "Assess risk factors and return ONLY a JSON object:\n"
                    '{"recommendation": "buy|sell|hold", "confidence": <0-100>, "reasoning": "<explanation>"}'
                )
                usr_p = f"Ticker: {ticker}\nDate: {date}\nRisk Snapshot:\n{self._format_risk(market_data)}"
                results["risk_analysis"] = self._call_llm(sys_p, usr_p)
                logger.info("  ✓ Risk analysis complete")
            except Exception as e:
                logger.error(f"Risk analysis error: {e}")
                results["risk_analysis"] = self._default_analysis()

        # 5. Coordinator Decision
        try:
            current_price = float(market_data.get("close", 0))
            sys_p = (
                "You are the head trader coordinating all agent analyses.\n"
                "Synthesize the agent recommendations into a final trading decision with exact parameters.\n"
                "Return ONLY a JSON object with this structure:\n"
                "{\n"
                '  "action": "buy|sell|hold",\n'
                '  "position_size": <0.0-1.0>,\n'
                '  "confidence": <0-100>,\n'
                '  "conviction": "low|medium|high",\n'
                '  "entry_price": <number>,\n'
                '  "stop_loss_price": <number>,\n'
                '  "take_profit_price": <number>,\n'
                '  "time_horizon": "short-term|medium-term|long-term",\n'
                '  "reasoning": "<consolidated explanation>"\n'
                "}"
            )
            usr_p = (
                f"Ticker: {ticker}\n"
                f"Date: {date}\n"
                f"Current Price: ${current_price:.2f}\n\n"
                f"Agent Analyses:\n{self._format_agent_results(results)}"
            )
            results["final_decision"] = self._call_llm(sys_p, usr_p)
            logger.info("  ✓ Final decision made")
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            results["final_decision"] = self._default_decision(market_data.get("close", 0))

        return results

    def _format_technical(self, data: Dict[str, Any]) -> str:
        return (
            f"- RSI: {data.get('rsi', 'N/A')}\n"
            f"- MACD: {data.get('macd', 'N/A')}\n"
            f"- SMA 20: {data.get('sma_20', 'N/A')}\n"
            f"- Bollinger Band Position: {data.get('bb_position', 'N/A')}\n"
            f"- Volume Ratio: {data.get('volume_ratio', 'N/A')}\n"
            f"- Momentum: {data.get('momentum', 'N/A')}\n"
        )

    def _format_fundamental(self, data: Dict[str, Any]) -> str:
        vol = data.get("volume", 0)
        formatted_vol = f"{vol:,}" if isinstance(vol, (int, float)) else str(vol)
        return (
            f"- Close Price: ${data.get('close', 'N/A')}\n"
            f"- Volume: {formatted_vol}\n"
            f"- 20-day SMA: ${data.get('sma_20', 'N/A')}\n"
        )

    def _format_risk(self, data: Dict[str, Any]) -> str:
        return (
            f"- Current Price: ${data.get('close', 'N/A')}\n"
            f"- Volatility (RSI): {data.get('rsi', 'N/A')}\n"
            f"- Volume Ratio: {data.get('volume_ratio', 'N/A')}\n"
        )

    def _format_news(self, news: List[Dict[str, Any]]) -> str:
        if not news:
            return "No recent news available."
        lines = []
        for item in news[:5]:
            lines.append(f"- {item.get('title', 'N/A')} ({item.get('source', 'Unknown')})")
        return "\n".join(lines)

    def _format_agent_results(self, results: Dict[str, Any]) -> str:
        lines = []
        for k in ["technical_analysis", "fundamental_analysis", "sentiment_analysis", "risk_analysis"]:
            if k in results:
                a = results[k]
                name = k.replace("_", " ").title()
                lines.append(
                    f"{name}:\n"
                    f"- Recommendation: {a.get('recommendation', 'N/A')}\n"
                    f"- Confidence: {a.get('confidence', 0)}%\n"
                    f"- Reasoning: {a.get('reasoning', 'N/A')}\n"
                )
        return "\n".join(lines)

    def _default_analysis(self) -> Dict[str, Any]:
        return {"recommendation": "hold", "confidence": 50, "reasoning": "Analysis temporarily unavailable"}

    def _default_decision(self, current_price: float) -> Dict[str, Any]:
        p = float(current_price) if current_price else 100.0
        return {
            "action": "hold",
            "position_size": 0.0,
            "confidence": 50,
            "conviction": "low",
            "entry_price": p,
            "stop_loss_price": round(p * 0.95, 2),
            "take_profit_price": round(p * 1.05, 2),
            "time_horizon": "medium-term",
            "reasoning": "Decision unavailable — defaulting to neutral hold",
        }
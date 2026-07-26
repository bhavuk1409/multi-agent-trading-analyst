"""
Advanced Multi-Agent System for Trading Decisions
Uses LangChain with Groq for multi-agent collaboration
"""

import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class RobustJsonOutputParser(BaseOutputParser):
    """Robust JSON parser that extracts JSON content even if the LLM includes preamble or markdown."""
    def parse(self, text: Any) -> Any:
        if isinstance(text, dict):
            return text
        text_str = str(text).strip()
        try:
            # Strip markdown codeblock if present
            if text_str.startswith("```"):
                text_str = re.sub(r"^```(?:json)?", "", text_str)
                text_str = re.sub(r"```$", "", text_str).strip()
            
            # Extract json object using regex
            match = re.search(r'\{.*\}', text_str, re.DOTALL)
            if match:
                text_str = match.group(0)
            
            return json.loads(text_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON output: {e}. Output text: {text_str[:200]}")
            raise


# Pydantic schemas for structured output
class AgentAnalysis(BaseModel):
    """Schema for individual agent analysis."""
    recommendation: str = Field(description="buy, sell, or hold")
    confidence: int = Field(description="confidence level 0-100")
    reasoning: str = Field(description="explanation of the recommendation")


class FinalDecision(BaseModel):
    """Schema for final trading decision."""
    action: str = Field(description="buy, sell, or hold")
    position_size: float = Field(description="position size as decimal (0.0-1.0)")
    confidence: int = Field(description="overall confidence 0-100")
    conviction: str = Field(description="low, medium, or high")
    entry_price: float = Field(description="suggested entry price")
    stop_loss_price: float = Field(description="stop loss price")
    take_profit_price: float = Field(description="take profit target")
    time_horizon: str = Field(description="short-term, medium-term, or long-term")
    reasoning: str = Field(description="consolidated reasoning from all agents")


class AdvancedMultiAgentSystem:
    """
    Multi-agent trading system using LangChain + Groq.
    Each agent specializes in one analysis type.
    """
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        agent_config: Dict[str, Any] = None
    ):
        """
        Initialize multi-agent system.
        
        Args:
            model: Groq model name (e.g., llama-3.3-70b-versatile, llama-3.1-70b-versatile, gemma2-9b-it)
            temperature: LLM temperature
            agent_config: Agent weights and settings
        """
        self.model_name = model
        self.temperature = temperature
        self.agent_config = agent_config or {
            'technical_analyst': {'enabled': True, 'weight': 0.25},
            'fundamental_analyst': {'enabled': True, 'weight': 0.25},
            'sentiment_analyst': {'enabled': True, 'weight': 0.25},
            'risk_manager': {'enabled': True, 'weight': 0.25}
        }
        
        # Initialize LLM with Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            max_tokens=2000
        )
        
        # Setup agents with parsers
        self._setup_agents()
        
        logger.info(f"✓ Multi-agent system initialized with model: {model}")
    
    def _setup_agents(self):
        """Create specialized agent chains."""
        
        # Technical Analyst
        self.technical_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical analyst specializing in chart patterns and indicators.
Analyze the technical data and provide a structured recommendation.

Return ONLY a raw JSON object with no markdown formatting or intro text:
{{
    "recommendation": "buy|sell|hold",
    "confidence": <0-100>,
    "reasoning": "<your explanation>"
}}"""),
            ("user", """Ticker: {ticker}
Date: {date}

Technical Indicators:
{technical_data}

Provide your technical analysis.""")
        ])
        
        # Fundamental Analyst
        self.fundamental_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fundamental analyst specializing in company valuation and market conditions.
Analyze the fundamental data and provide a structured recommendation.

Return ONLY a raw JSON object with no markdown formatting or intro text:
{{
    "recommendation": "buy|sell|hold",
    "confidence": <0-100>,
    "reasoning": "<your explanation>"
}}"""),
            ("user", """Ticker: {ticker}
Date: {date}

Market Data:
{market_data}

Provide your fundamental analysis.""")
        ])
        
        # Sentiment Analyst
        self.sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sentiment analyst specializing in news and market sentiment.
Analyze the news sentiment and provide a structured recommendation.

Return ONLY a raw JSON object with no markdown formatting or intro text:
{{
    "recommendation": "buy|sell|hold",
    "confidence": <0-100>,
    "reasoning": "<your explanation>"
}}"""),
            ("user", """Ticker: {ticker}
Date: {date}

Recent News:
{news}

Provide your sentiment analysis.""")
        ])
        
        # Risk Manager
        self.risk_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a risk manager specializing in portfolio risk assessment.
Analyze the risk factors and provide a structured recommendation.

Return ONLY a raw JSON object with no markdown formatting or intro text:
{{
    "recommendation": "buy|sell|hold",
    "confidence": <0-100>,
    "reasoning": "<your explanation>"
}}"""),
            ("user", """Ticker: {ticker}
Date: {date}

Market Data:
{market_data}

Assess the risk and provide your recommendation.""")
        ])
        
        # Coordinator for final decision
        self.coordinator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the head trader coordinating all analyst recommendations.
Synthesize the agent analyses into a final trading decision with specific parameters.

Return ONLY a raw JSON object with no markdown formatting or intro text:
{{
    "action": "buy|sell|hold",
    "position_size": <0.0-1.0>,
    "confidence": <0-100>,
    "conviction": "low|medium|high",
    "entry_price": <price>,
    "stop_loss_price": <price>,
    "take_profit_price": <price>,
    "time_horizon": "short-term|medium-term|long-term",
    "reasoning": "<consolidated explanation>"
}}"""),
            ("user", """Ticker: {ticker}
Date: {date}
Current Price: {current_price}

Agent Analyses:
{agent_analyses}

Provide your final coordinated decision.""")
        ])
        
        # Setup parsers
        self.agent_parser = RobustJsonOutputParser()
        self.decision_parser = RobustJsonOutputParser()
    
    def analyze(
        self,
        ticker: str,
        date: str,
        market_data: Dict[str, Any],
        news: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run multi-agent analysis.
        
        Args:
            ticker: Stock ticker
            date: Analysis date
            market_data: Technical/fundamental data
            news: News articles
            
        Returns:
            Complete analysis with all agent outputs and final decision
        """
        logger.info(f"🤖 Running multi-agent analysis for {ticker} on {date}")
        
        results = {}
        
        # 1. Technical Analysis
        if self.agent_config['technical_analyst']['enabled']:
            try:
                tech_chain = self.technical_prompt | self.llm | self.agent_parser
                results['technical_analysis'] = tech_chain.invoke({
                    'ticker': ticker,
                    'date': date,
                    'technical_data': self._format_technical(market_data)
                })
                logger.info("  ✓ Technical analysis complete")
            except Exception as e:
                logger.error(f"Technical analysis failed: {e}")
                results['technical_analysis'] = self._default_analysis()
        
        # 2. Fundamental Analysis
        if self.agent_config['fundamental_analyst']['enabled']:
            try:
                fund_chain = self.fundamental_prompt | self.llm | self.agent_parser
                results['fundamental_analysis'] = fund_chain.invoke({
                    'ticker': ticker,
                    'date': date,
                    'market_data': self._format_fundamental(market_data)
                })
                logger.info("  ✓ Fundamental analysis complete")
            except Exception as e:
                logger.error(f"Fundamental analysis failed: {e}")
                results['fundamental_analysis'] = self._default_analysis()
        
        # 3. Sentiment Analysis
        if self.agent_config['sentiment_analyst']['enabled']:
            try:
                sentiment_chain = self.sentiment_prompt | self.llm | self.agent_parser
                results['sentiment_analysis'] = sentiment_chain.invoke({
                    'ticker': ticker,
                    'date': date,
                    'news': self._format_news(news)
                })
                logger.info("  ✓ Sentiment analysis complete")
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                results['sentiment_analysis'] = self._default_analysis()
        
        # 4. Risk Assessment
        if self.agent_config['risk_manager']['enabled']:
            try:
                risk_chain = self.risk_prompt | self.llm | self.agent_parser
                results['risk_analysis'] = risk_chain.invoke({
                    'ticker': ticker,
                    'date': date,
                    'market_data': self._format_risk(market_data)
                })
                logger.info("  ✓ Risk analysis complete")
            except Exception as e:
                logger.error(f"Risk analysis failed: {e}")
                results['risk_analysis'] = self._default_analysis()
        
        # 5. Coordinator Decision
        try:
            coord_chain = self.coordinator_prompt | self.llm | self.decision_parser
            results['final_decision'] = coord_chain.invoke({
                'ticker': ticker,
                'date': date,
                'current_price': market_data.get('close', 0),
                'agent_analyses': self._format_agent_results(results)
            })
            logger.info("  ✓ Final decision made")
        except Exception as e:
            logger.error(f"Coordinator failed: {e}")
            results['final_decision'] = self._default_decision(market_data.get('close', 0))
        
        return results
    
    def _format_technical(self, data: Dict[str, Any]) -> str:
        """Format technical indicators."""
        return f"""
- RSI: {data.get('rsi', 'N/A')}
- MACD: {data.get('macd', 'N/A')}
- SMA 20: {data.get('sma_20', 'N/A')}
- Bollinger Band Position: {data.get('bb_position', 'N/A')}
- Volume Ratio: {data.get('volume_ratio', 'N/A')}
- Momentum: {data.get('momentum', 'N/A')}
"""
    
    def _format_fundamental(self, data: Dict[str, Any]) -> str:
        """Format fundamental data."""
        return f"""
- Close Price: ${data.get('close', 'N/A')}
- Volume: {data.get('volume', 'N/A'):,}
- 20-day SMA: ${data.get('sma_20', 'N/A')}
"""
    
    def _format_risk(self, data: Dict[str, Any]) -> str:
        """Format risk data."""
        return f"""
- Current Price: ${data.get('close', 'N/A')}
- Volatility (RSI): {data.get('rsi', 'N/A')}
- Volume: {data.get('volume', 'N/A'):,}
"""
    
    def _format_news(self, news: List[Dict[str, Any]]) -> str:
        """Format news articles."""
        if not news:
            return "No recent news available"
        
        formatted = []
        for item in news[:5]:  # Limit to 5 articles
            formatted.append(f"- {item.get('title', 'N/A')} ({item.get('sentiment', 'neutral')})")
        
        return "\n".join(formatted)
    
    def _format_agent_results(self, results: Dict[str, Any]) -> str:
        """Format agent analyses for coordinator."""
        formatted = []
        
        for key in ['technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'risk_analysis']:
            if key in results:
                analysis = results[key]
                name = key.replace('_', ' ').title()
                formatted.append(f"""
{name}:
- Recommendation: {analysis.get('recommendation', 'N/A')}
- Confidence: {analysis.get('confidence', 0)}%
- Reasoning: {analysis.get('reasoning', 'N/A')}
""")
        
        return "\n".join(formatted)
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis on error."""
        return {
            'recommendation': 'hold',
            'confidence': 50,
            'reasoning': 'Analysis unavailable'
        }
    
    def _default_decision(self, current_price: float) -> Dict[str, Any]:
        """Return default decision on error."""
        return {
            'action': 'hold',
            'position_size': 0.0,
            'confidence': 50,
            'conviction': 'low',
            'entry_price': current_price,
            'stop_loss_price': current_price * 0.95,
            'take_profit_price': current_price * 1.05,
            'time_horizon': 'medium-term',
            'reasoning': 'Decision unavailable - defaulting to hold'
        }
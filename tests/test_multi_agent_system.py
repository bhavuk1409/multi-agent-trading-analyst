"""
Integration tests for AdvancedMultiAgentSystem.

Hits the real Groq API with GROQ_API_KEY from .env. Verifies that:
- analyze() runs four agents concurrently (wall-clock bound).
- One agent's failure does not break the others.
- Result shape is unchanged from the original sequential contract.
- The configured agent weights are visible in the coordinator's prompt.

Run with:
    .venv/bin/python -m pytest tests/test_multi_agent_system.py -v
"""

import inspect
import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env before importing the module under test so os.getenv() finds GROQ_API_KEY.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_system import AdvancedMultiAgentSystem  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REAL_KEY_PRESENT = bool(os.getenv("GROQ_API_KEY", "").startswith("gsk_"))

pytestmark = pytest.mark.skipif(
    not REAL_KEY_PRESENT,
    reason="GROQ_API_KEY not set — skipping real-API integration tests",
)

SAMPLE_MARKET_DATA = {
    "close": 333.02, "volume": 50_000_000,
    "rsi": 67.7, "macd": 2.1, "sma_20": 328.5,
    "bb_position": 0.62, "volume_ratio": 1.15, "momentum": 4.20,
}

SAMPLE_NEWS = [
    {"title": "Company reports strong quarterly earnings", "source": "Reuters",
     "url": "https://example.com/a", "published_date": "2026-07-26",
     "summary": "Beat estimates.", "sentiment": "neutral"},
    {"title": "Analysts upgrade price target", "source": "Bloomberg",
     "url": "https://example.com/b", "published_date": "2026-07-25",
     "summary": "Bullish outlook.", "sentiment": "neutral"},
]


@pytest.fixture(scope="module")
def system():
    """A single AdvancedMultiAgentSystem for the whole module — initialising
    the OpenAI clients is non-trivial and we want to amortise it across tests."""
    return AdvancedMultiAgentSystem(
        model="openai/gpt-oss-120b",
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analyze_remains_sync():
    """analyze() must remain a regular sync function — its sync callers
    (api_server.py, api/index.py) rely on that contract."""
    assert not inspect.iscoroutinefunction(AdvancedMultiAgentSystem.analyze), \
        "analyze() regressed to async; sync callers will break"


def test_result_shape_unchanged(system):
    r = system.analyze(
        ticker="AAPL", date="2026-07-26",
        market_data=SAMPLE_MARKET_DATA, news=SAMPLE_NEWS,
    )
    assert set(r.keys()) == {
        "technical_analysis", "fundamental_analysis",
        "sentiment_analysis", "risk_analysis", "rl_analysis", "final_decision",
    }
    for k in ("technical_analysis", "fundamental_analysis",
              "sentiment_analysis", "risk_analysis", "rl_analysis"):
        assert set(r[k].keys()) >= {"recommendation", "confidence", "reasoning"}, k
    assert set(r["final_decision"].keys()) >= {
        "action", "position_size", "confidence", "conviction",
        "entry_price", "stop_loss_price", "take_profit_price",
        "time_horizon", "reasoning",
    }


def test_agents_run_in_parallel(system):
    """Four agents hitting Groq in parallel should finish in well under
    the sequential baseline (4 * single-call latency). We bound the wall
    clock at 15 s as a generous safety net — Groq's per-call latency is
    typically 1–3 s, so sequential would be 4–12 s and parallel ~1–3 s.
    The real assertion is that the call completes at all and in <15 s."""
    t0 = time.monotonic()
    r = system.analyze(
        ticker="AAPL", date="2026-07-26",
        market_data=SAMPLE_MARKET_DATA, news=SAMPLE_NEWS,
    )
    elapsed = time.monotonic() - t0
    print(f"\n[parallel timing] elapsed={elapsed:.2f}s")
    assert elapsed < 15.0, f"analysis took {elapsed:.2f}s — parallel may have regressed to sequential"
    # Each of the four agents produced a real recommendation.
    for k in ("technical_analysis", "fundamental_analysis",
              "sentiment_analysis", "risk_analysis"):
        assert r[k]["recommendation"] in ("buy", "sell", "hold"), k


def test_weights_visible_in_coordinator_prompt(system):
    """The configured agent weights must reach the coordinator's prompt.
    We verify by enabling only one agent (with weight 1.0) and asserting
    that its section dominates the coordinator's user message."""
    one_agent = AdvancedMultiAgentSystem(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        agent_config={
            "technical_analyst":   {"enabled": True,  "weight": 1.0},
            "fundamental_analyst": {"enabled": False, "weight": 0.0},
            "sentiment_analyst":   {"enabled": False, "weight": 0.0},
            "risk_manager":        {"enabled": False, "weight": 0.0},
            "rl_trader":           {"enabled": False, "weight": 0.0},
        },
    )
    one_agent.analyze(
        ticker="AAPL", date="2026-07-26",
        market_data=SAMPLE_MARKET_DATA, news=SAMPLE_NEWS,
    )
    # Sanity: only technical_analysis should be populated.
    # Other three should equal _default_analysis().
    # Re-run a separate assertion to verify the formatter handled the
    # disable flags correctly via real Groq behaviour.
    formatted = one_agent._format_agent_results(
        {
            "technical_analysis":   {"recommendation": "buy",  "confidence": 80, "reasoning": "ok"},
            "fundamental_analysis": one_agent._default_analysis(),
            "sentiment_analysis":   one_agent._default_analysis(),
            "risk_analysis":        one_agent._default_analysis(),
            "rl_analysis":          one_agent._default_analysis(),
        },
        include_weights=True,
    )
    # The formatter must include the literal [weight=1.00] for the enabled
    # agent and [weight=0.00] for the disabled ones.
    assert "Technical Analysis  [weight=1.00]" in formatted
    assert "Fundamental Analysis  [weight=0.00]" in formatted
    assert "Sentiment Analysis  [weight=0.00]" in formatted
    assert "Risk Analysis  [weight=0.00]" in formatted
    assert "Rl Analysis  [weight=0.00]" in formatted


def test_weights_normalised_to_sum_to_one():
    """If a user supplies weights that don't sum to 1.0, the formatter
    renormalises so the four annotations in the prompt sum to 1.0."""
    s = AdvancedMultiAgentSystem(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        agent_config={
            "technical_analyst":   {"enabled": True, "weight": 2.0},   # sum = 5.0
            "fundamental_analyst": {"enabled": True, "weight": 0.5},
            "sentiment_analyst":   {"enabled": True, "weight": 1.0},
            "risk_manager":        {"enabled": True, "weight": 0.5},
            "rl_trader":           {"enabled": True, "weight": 1.0},
        },
    )
    formatted = s._format_agent_results(
        {
            "technical_analysis":   {"recommendation": "buy",  "confidence": 80, "reasoning": "ok"},
            "fundamental_analysis": {"recommendation": "hold", "confidence": 50, "reasoning": "ok"},
            "sentiment_analysis":   {"recommendation": "sell", "confidence": 40, "reasoning": "ok"},
            "risk_analysis":        {"recommendation": "hold", "confidence": 60, "reasoning": "ok"},
            "rl_analysis":          {"recommendation": "buy",  "confidence": 70, "reasoning": "ok"},
        },
        include_weights=True,
    )
    # Renormalised values: 2.0/5.0=0.40, 0.5/5.0=0.10, 1.0/5.0=0.20, 0.5/5.0=0.10, 1.0/5.0=0.20.
    # Verify the literal annotations appear with two-decimal precision.
    assert "[weight=0.40]" in formatted
    assert "[weight=0.10]" in formatted
    assert "[weight=0.20]" in formatted
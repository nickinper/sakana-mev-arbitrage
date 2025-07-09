"""
Profit Generation Module - For Immediate MEV Arbitrage Profits

This module provides a minimal, profit-focused approach to MEV arbitrage
that prioritizes immediate execution and real-world feedback over complexity.
"""

from .minimal_arbitrage_detector import (
    ArbitrageOpportunity,
    MinimalDataPipeline,
    ProfitProjector,
    SimulationEngine
)

from .profit_focused_evolution import (
    ProfitAgent,
    ProfitFocusedEvolution,
    MarketFeedback
)

from .demo_execution_interface import (
    DemoGenerator,
    ExecutionTracker,
    ProfitSystem
)

__all__ = [
    'ArbitrageOpportunity',
    'MinimalDataPipeline',
    'ProfitProjector',
    'SimulationEngine',
    'ProfitAgent',
    'ProfitFocusedEvolution',
    'MarketFeedback',
    'DemoGenerator',
    'ExecutionTracker',
    'ProfitSystem'
]
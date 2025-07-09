"""
Training Module - Interactive visualization for MEV arbitrage training

This module provides multiple interfaces for monitoring and controlling
the evolutionary training process in real-time.
"""

from .training_dashboard import TrainingDashboard
from .training_terminal_ui import TerminalTrainingUI

__all__ = [
    'TrainingDashboard',
    'TerminalTrainingUI'
]
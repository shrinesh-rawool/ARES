"""
marl — Multi-Agent Reinforcement Learning module for ARES.
Implements MAPPO (Multi-Agent PPO) with Centralized Training and Decentralized Execution.
"""

from marl.environment import MultiAgentWarehouseEnv
from marl.policy import MAPPOPolicy

__all__ = ["MultiAgentWarehouseEnv", "MAPPOPolicy"]

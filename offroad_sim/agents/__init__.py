"""Agent implementations and interfaces."""

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import (
    KeyboardAgent,
    RandomAgent,
    RuleBasedGoalAgent,
    StopAgent,
)
from offroad_sim.agents.registry import AgentRegistry, AgentSpec, default_agent_registry, make_agent
from offroad_sim.agents.model_mpc import ModelMPCAgent
from offroad_sim.agents.route_world_model import RouteWorldModelAgent
from offroad_sim.agents.world_model import WorldModelAgent

__all__ = [
    "AgentRegistry",
    "AgentSpec",
    "KeyboardAgent",
    "ModelMPCAgent",
    "OffroadAgent",
    "RandomAgent",
    "RuleBasedGoalAgent",
    "RouteWorldModelAgent",
    "StopAgent",
    "WorldModelAgent",
    "default_agent_registry",
    "make_agent",
]

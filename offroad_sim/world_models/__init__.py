"""World model interfaces and baselines."""

from offroad_sim.world_models.base import BaseWorldModel, WorldModelPrediction
from offroad_sim.world_models.kinematic import SimpleKinematicWorldModel

__all__ = [
    "BaseWorldModel",
    "SimpleKinematicWorldModel",
    "WorldModelPrediction",
]

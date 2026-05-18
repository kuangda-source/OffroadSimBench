from __future__ import annotations

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning.cem import WorldModelCEMPlanner
from offroad_sim.planning.region_cost import RegionTrajectoryCost
from offroad_sim.world_models.base import BaseWorldModel, WorldModelPrediction


def _region_info() -> dict[str, object]:
    return {
        "region": {"polygon": [[0.0, -2.0], [10.0, -2.0], [10.0, 2.0], [0.0, 2.0]]},
        "cost": {"out_of_region_weight": 100.0, "boundary_weight": 0.0},
    }


def test_region_trajectory_cost_penalizes_out_of_bounds_states() -> None:
    cost = RegionTrajectoryCost.from_task(_region_info())

    inside = [VehicleState(x=2.0, y=0.0), VehicleState(x=4.0, y=0.0)]
    outside = [VehicleState(x=2.0, y=0.0), VehicleState(x=4.0, y=5.0)]

    assert cost.evaluate(inside) == 0.0
    assert cost.evaluate(outside) >= 100.0


def test_world_model_cem_adds_region_cost_from_observation_info() -> None:
    class SplitWorldModel(BaseWorldModel):
        def predict(self, observation, action, horizon=10):
            actions = list(action)
            y = 5.0 if actions[0].steer > 0.0 else 0.0
            states = [VehicleState(x=2.0, y=y, speed=1.0), VehicleState(x=4.0, y=y, speed=1.0)]
            return WorldModelPrediction(states=states, actions=actions, metadata={"max_risk": 0.0})

    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(5.0, 0.0),
        info={"navigation_region": _region_info()},
    )
    planner = WorldModelCEMPlanner(horizon=2, num_samples=4, iterations=1, region_weight=1.0)

    inside_cost = planner._candidate_cost(observation, SplitWorldModel(), np.asarray([[0.0, 0.4, 0.0], [0.0, 0.4, 0.0]]))
    outside_cost = planner._candidate_cost(observation, SplitWorldModel(), np.asarray([[1.0, 0.4, 0.0], [1.0, 0.4, 0.0]]))

    assert outside_cost > inside_cost + 50.0

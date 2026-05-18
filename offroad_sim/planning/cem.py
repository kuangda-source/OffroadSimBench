"""CEM path planner over the OffroadSimBench world-model interface."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning.region_cost import navigation_region_from_observation_info
from offroad_sim.planning.types import ActionPlanner, PlanningResult
from offroad_sim.world_models import BaseWorldModel


ACTION_LOW = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
ACTION_HIGH = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)


class WorldModelCEMPlanner(ActionPlanner):
    """Cross-entropy planner using any local `BaseWorldModel.predict()`."""

    planner_type = "world_model_cem"

    def __init__(
        self,
        *,
        horizon: int = 10,
        num_samples: int = 128,
        iterations: int = 4,
        elite_fraction: float = 0.15,
        seed: int = 0,
        goal_weight: float = 1.0,
        risk_weight: float = 2.0,
        region_weight: float = 1.0,
        smoothness_weight: float = 0.08,
        action_weight: float = 0.04,
    ) -> None:
        self.horizon = max(1, int(horizon))
        self.num_samples = max(4, int(num_samples))
        self.iterations = max(1, int(iterations))
        self.elite_fraction = float(np.clip(elite_fraction, 0.02, 0.8))
        self.seed = int(seed)
        self.goal_weight = float(goal_weight)
        self.risk_weight = float(risk_weight)
        self.region_weight = float(region_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.action_weight = float(action_weight)
        self.rng = np.random.default_rng(self.seed)

    def plan(
        self,
        observation: Observation,
        world_model: BaseWorldModel,
        *,
        reference_action: Action | None = None,
    ) -> PlanningResult:
        mean = self._initial_mean(reference_action)
        std = np.tile(np.asarray([0.55, 0.35, 0.18], dtype=np.float64), (self.horizon, 1))
        elite_count = max(2, int(round(self.num_samples * self.elite_fraction)))
        final_costs = np.zeros(self.num_samples, dtype=np.float64)
        candidates = np.zeros((self.num_samples, self.horizon, 3), dtype=np.float64)

        for _ in range(self.iterations):
            candidates = self.rng.normal(loc=mean, scale=std, size=(self.num_samples, self.horizon, 3))
            candidates = np.clip(candidates, ACTION_LOW, ACTION_HIGH)
            candidates[0] = mean
            final_costs = np.asarray([self._candidate_cost(observation, world_model, item) for item in candidates])
            elite_indices = np.argsort(final_costs)[:elite_count]
            elites = candidates[elite_indices]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 0.03)

        best_index = int(np.argmin(final_costs))
        best = candidates[best_index]
        actions = [_row_to_action(row) for row in best]
        prediction = world_model.predict(observation, actions, horizon=self.horizon)
        return PlanningResult(
            actions=actions,
            predicted_states=prediction.states,
            costs=final_costs.tolist(),
            best_cost=float(final_costs[best_index]),
            metadata={
                "planner": self.planner_type,
                "horizon": self.horizon,
                "num_samples": self.num_samples,
                "iterations": self.iterations,
                "elite_count": elite_count,
                "prediction": prediction.metadata,
            },
        )

    def _initial_mean(self, reference_action: Action | None) -> np.ndarray:
        action = reference_action or Action(throttle=0.35)
        row = np.asarray([action.steer, action.throttle, action.brake], dtype=np.float64)
        row = np.clip(row, ACTION_LOW, ACTION_HIGH)
        return np.tile(row, (self.horizon, 1))

    def _candidate_cost(self, observation: Observation, world_model: BaseWorldModel, candidate: np.ndarray) -> float:
        actions = [_row_to_action(row) for row in candidate]
        try:
            prediction = world_model.predict(observation, actions, horizon=self.horizon)
        except Exception:
            return float("inf")

        final_state = prediction.final_state
        if final_state is None:
            return float("inf")

        goal_x, goal_y = observation.goal
        goal_distance = math.hypot(final_state.x - goal_x, final_state.y - goal_y)
        start_distance = math.hypot(observation.vehicle_state.x - goal_x, observation.vehicle_state.y - goal_y)
        progress_bonus = max(0.0, start_distance - goal_distance)
        risk = float(prediction.metadata.get("max_risk", prediction.metadata.get("mean_risk", 0.0)) or 0.0)
        region_cost = 0.0
        region = navigation_region_from_observation_info(observation.info)
        if region is not None:
            region_cost = region.evaluate(prediction.states)
        smoothness = float(np.mean(np.abs(np.diff(candidate, axis=0)))) if len(candidate) > 1 else 0.0
        effort = float(np.mean(np.abs(candidate)))
        return (
            self.goal_weight * goal_distance
            - 0.25 * progress_bonus
            + self.risk_weight * risk
            + self.region_weight * region_cost
            + self.smoothness_weight * smoothness
            + self.action_weight * effort
        )


def _row_to_action(row: np.ndarray) -> Action:
    return Action(steer=float(row[0]), throttle=float(row[1]), brake=float(row[2]))

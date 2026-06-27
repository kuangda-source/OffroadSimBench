"""Navigation MPC planner with task-region costs and optional model scoring."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning.region_cost import navigation_region_from_observation_info
from offroad_sim.planning.types import PlanningResult
from offroad_sim.world_models import BaseWorldModel, SimpleKinematicWorldModel


ActionScorer = Callable[[list[list[Action]]], list[float]]


class NavigationMPCPlanner:
    """Receding-horizon controller for bounded region navigation."""

    planner_type = "navigation_mpc"

    def __init__(
        self,
        *,
        horizon: int = 10,
        num_samples: int = 64,
        iterations: int = 1,
        seed: int = 0,
        goal_weight: float = 1.0,
        progress_weight: float = 0.35,
        risk_weight: float = 2.0,
        region_weight: float = 1.0,
        smoothness_weight: float = 0.05,
        action_weight: float = 0.03,
        model_score_weight: float = 0.35,
        heading_weight: float = 0.45,
    ) -> None:
        self.horizon = max(1, int(horizon))
        self.num_samples = max(4, int(num_samples))
        self.iterations = max(1, int(iterations))
        self.seed = int(seed)
        self.goal_weight = float(goal_weight)
        self.progress_weight = float(progress_weight)
        self.risk_weight = float(risk_weight)
        self.region_weight = float(region_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.action_weight = float(action_weight)
        self.model_score_weight = float(model_score_weight)
        self.heading_weight = float(heading_weight)
        self._fallback_model = SimpleKinematicWorldModel()
        self.rng = np.random.default_rng(self.seed)

    def plan(
        self,
        observation: Observation,
        world_model: BaseWorldModel,
        *,
        reference_action: Action | None = None,
        score_actions: ActionScorer | None = None,
    ) -> PlanningResult:
        candidates = self._candidate_sequences(reference_action)
        external_scores = self._external_scores(candidates, score_actions)
        scored: list[tuple[float, int, list[Action], list[VehicleState], dict[str, Any]]] = []

        for index, candidate in enumerate(candidates):
            states, metadata = self._predict_states(observation, world_model, candidate)
            cost, cost_parts = self._trajectory_cost(observation, candidate, states, metadata, external_scores[index])
            scored.append((cost, index, candidate, states, cost_parts))

        scored.sort(key=lambda row: (row[0], row[1]))
        best_cost, best_index, best_actions, best_states, best_parts = scored[0]
        return PlanningResult(
            actions=best_actions,
            predicted_states=best_states,
            costs=[float(row[0]) for row in scored],
            best_cost=float(best_cost),
            metadata={
                "planner": self.planner_type,
                "horizon": self.horizon,
                "num_samples": len(candidates),
                "iterations": self.iterations,
                "best_candidate_index": best_index,
                "external_score_used": score_actions is not None,
                **best_parts,
            },
        )

    def _candidate_sequences(self, reference_action: Action | None) -> list[list[Action]]:
        reference = reference_action or Action(throttle=0.45)
        candidates: list[list[Action]] = [[_clamp_action(reference) for _ in range(self.horizon)]]
        for brake in (0.18, 0.35):
            rows = []
            for step in range(self.horizon):
                decay = 1.0 - step / max(1, self.horizon - 1)
                rows.append(Action(steer=float(reference.steer * decay), throttle=0.0, brake=float(brake)))
            candidates.append(rows)
            if len(candidates) >= self.num_samples:
                return candidates
        steer_values = np.linspace(-0.9, 0.9, 9)
        throttle_values = [0.15, 0.3, 0.45, 0.65]
        for throttle in throttle_values:
            for steer in steer_values:
                candidates.append([Action(steer=float(steer), throttle=float(throttle), brake=0.0) for _ in range(self.horizon)])
                if len(candidates) >= self.num_samples:
                    return candidates

        for throttle in (0.45, 0.65):
            for start_steer in steer_values:
                rows = []
                for step in range(self.horizon):
                    decay = 1.0 - step / max(1, self.horizon - 1)
                    rows.append(Action(steer=float(start_steer * decay), throttle=float(throttle), brake=0.0))
                candidates.append(rows)
                if len(candidates) >= self.num_samples:
                    return candidates

        while len(candidates) < self.num_samples:
            steer = float(self.rng.uniform(-0.9, 0.9))
            throttle = float(self.rng.uniform(0.2, 0.85))
            candidates.append([Action(steer=steer, throttle=throttle, brake=0.0) for _ in range(self.horizon)])
        return candidates

    def _external_scores(self, candidates: list[list[Action]], score_actions: ActionScorer | None) -> list[float]:
        if score_actions is None:
            return [0.0 for _ in candidates]
        values = [float(value) for value in score_actions(candidates)]
        if len(values) != len(candidates):
            raise ValueError("Action scorer must return one cost per candidate sequence.")
        return values

    def _predict_states(
        self,
        observation: Observation,
        world_model: BaseWorldModel,
        candidate: list[Action],
    ) -> tuple[list[VehicleState], dict[str, Any]]:
        prediction_error: str | None = None
        try:
            prediction = world_model.predict(observation, candidate, horizon=self.horizon)
            if prediction.states:
                return prediction.states, dict(prediction.metadata)
            prediction_error = "world model returned no predicted states"
        except Exception as exc:
            prediction_error = f"{type(exc).__name__}: {exc}"
        prediction = self._fallback_model.predict(observation, candidate, horizon=self.horizon)
        metadata = dict(prediction.metadata)
        metadata["prediction_fallback"] = "simple_kinematic"
        if prediction_error:
            metadata["prediction_error"] = prediction_error
        return prediction.states, metadata

    def _trajectory_cost(
        self,
        observation: Observation,
        candidate: list[Action],
        states: list[VehicleState],
        prediction_metadata: dict[str, Any],
        external_score: float,
    ) -> tuple[float, dict[str, Any]]:
        final_state = states[-1] if states else observation.vehicle_state
        goal_x, goal_y = observation.goal
        goal_distance = math.hypot(float(final_state.x) - goal_x, float(final_state.y) - goal_y)
        start_distance = math.hypot(float(observation.vehicle_state.x) - goal_x, float(observation.vehicle_state.y) - goal_y)
        progress = start_distance - goal_distance
        desired_heading = math.atan2(goal_y - float(final_state.y), goal_x - float(final_state.x))
        heading_error = abs(_wrap_angle(desired_heading - float(final_state.yaw)))
        heading_alignment = heading_error / math.pi
        risk = float(prediction_metadata.get("max_risk", prediction_metadata.get("mean_risk", 0.0)) or 0.0)
        region_cost = 0.0
        region = navigation_region_from_observation_info(observation.info)
        if region is not None:
            region_cost = region.evaluate(states)
        actions = np.asarray([[item.steer, item.throttle, item.brake] for item in candidate], dtype=np.float64)
        smoothness = float(np.mean(np.abs(np.diff(actions, axis=0)))) if len(actions) > 1 else 0.0
        effort = float(np.mean(np.abs(actions))) if len(actions) else 0.0
        total = (
            self.goal_weight * goal_distance
            - self.progress_weight * progress
            + self.risk_weight * risk
            + self.region_weight * region_cost
            + self.smoothness_weight * smoothness
            + self.action_weight * effort
            + self.model_score_weight * external_score
            + self.heading_weight * heading_alignment
        )
        cost_parts: dict[str, Any] = {
            "goal_distance": float(goal_distance),
            "progress": float(progress),
            "risk_cost": float(risk),
            "region_cost": float(region_cost),
            "smoothness_cost": float(smoothness),
            "action_cost": float(effort),
            "external_model_cost": float(external_score),
            "heading_alignment_cost": float(heading_alignment),
            "heading_error_rad": float(heading_error),
        }
        if "prediction_fallback" in prediction_metadata:
            cost_parts["prediction_fallback"] = str(prediction_metadata["prediction_fallback"])
        if "prediction_error" in prediction_metadata:
            cost_parts["prediction_error"] = str(prediction_metadata["prediction_error"])
        return float(total), cost_parts


def _clamp_action(action: Action) -> Action:
    return Action(
        steer=float(np.clip(action.steer, -1.0, 1.0)),
        throttle=float(np.clip(action.throttle, 0.0, 1.0)),
        brake=float(np.clip(action.brake, 0.0, 1.0)),
    )


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

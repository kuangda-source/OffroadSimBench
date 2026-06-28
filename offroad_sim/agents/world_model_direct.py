"""Route-free world-model MPC agent."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.agents.model_mpc import _action_dict, _goal_radius_from_info, _speed_steer_limit, _turn_speed_target
from offroad_sim.core import Action, Observation
from offroad_sim.planning import make_planner
from offroad_sim.planning.navigation_mpc import NavigationMPCPlanner
from offroad_sim.world_models import make_world_model


class WorldModelDirectAgent(OffroadAgent):
    """Drive toward the task goal with a world-model planner and no expert route."""

    def __init__(
        self,
        world_model_name: str = "simple_kinematic",
        world_model_path: str | Path | None = None,
        planner_name: str | None = None,
        planner_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.world_model_name = str(world_model_name or "simple_kinematic")
        self.world_model_path = str(world_model_path) if world_model_path else None
        self.world_model = make_world_model(self.world_model_name, path=world_model_path) if world_model_path else make_world_model(self.world_model_name)
        self.reference_agent = RuleBasedGoalAgent(cruise_throttle=0.5)
        self.planner_name = str(planner_name or "navigation_mpc")
        planner_kwargs = dict(planner_config or {})
        if self.planner_name == "le_wm_cem" and world_model_path is not None:
            planner_kwargs.setdefault("checkpoint_path", world_model_path)
        self.planner = make_planner(self.planner_name, **planner_kwargs) if planner_name else NavigationMPCPlanner(**planner_kwargs)
        self._last_diagnostics: dict[str, Any] = {}
        self._stuck_steps = 0
        self._last_goal_distance: float | None = None
        self._goal_hold_latched = False

    def reset(self, scenario_info: Any) -> None:
        self.world_model.reset(scenario_info if isinstance(scenario_info, dict) else {"scenario": scenario_info})
        self.reference_agent.reset(scenario_info)
        self._last_diagnostics = {}
        self._stuck_steps = 0
        self._last_goal_distance = None
        self._goal_hold_latched = False

    def act(self, obs: Observation) -> Action:
        terminal_stop = self._terminal_stop_action(obs)
        if terminal_stop is not None:
            self._last_diagnostics = {
                "agent": "world_model_direct",
                "planner": self.planner_name,
                "world_model": self.world_model_name,
                "world_model_path": self.world_model_path,
                "target_goal": [float(obs.goal[0]), float(obs.goal[1])],
                "route_used": False,
                "goal_stop": True,
                "goal_hold_latched": self._goal_hold_latched,
                "executed_action": _action_dict(terminal_stop),
            }
            return terminal_stop
        reference_action = self.reference_agent.act(obs)
        planning = self.planner.plan(obs, self.world_model, reference_action=reference_action)
        stabilized = _stabilize_action(planning.first_action, reference_action, obs)
        action, stuck_recovery = self._progress_filter(stabilized, reference_action, obs)
        self._last_diagnostics = {
            "agent": "world_model_direct",
            "planner": planning.metadata.get("planner") or self.planner_name,
            "world_model": self.world_model_name,
            "world_model_path": self.world_model_path,
            "best_cost": planning.best_cost,
            "planning": planning.metadata,
            "target_goal": [float(obs.goal[0]), float(obs.goal[1])],
            "route_used": False,
            "reference_action": _action_dict(reference_action),
            "executed_action": _action_dict(action),
            "stuck_recovery": stuck_recovery,
            "goal_stop": False,
            "goal_hold_latched": self._goal_hold_latched,
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        return None

    def _progress_filter(self, action: Action, reference_action: Action, obs: Observation) -> tuple[Action, bool]:
        state = obs.vehicle_state
        goal_distance = math.hypot(float(state.x) - float(obs.goal[0]), float(state.y) - float(obs.goal[1]))
        speed = max(0.0, float(state.speed))
        improving = self._last_goal_distance is None or goal_distance < self._last_goal_distance - 0.05
        if speed < 0.4 and not improving and action.throttle > 0.35:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self._last_goal_distance = goal_distance
        if self._stuck_steps < 18:
            return action, False
        reference_steer = max(min(float(reference_action.steer), 1.0), -1.0)
        turn_demand = abs(reference_steer)
        recovery_phase = ((self._stuck_steps - 18) // 12) % 3
        if turn_demand > 0.35 and recovery_phase == 0:
            return (
                Action(steer=0.0, throttle=0.55, brake=0.0, gear=-1),
                True,
            )
        if turn_demand > 0.35 and recovery_phase == 2:
            return (
                Action(
                    steer=max(min(reference_steer, 0.45), -0.45),
                    throttle=0.75,
                    brake=0.0,
                    gear=1,
                ),
                True,
            )
        if turn_demand > 0.75:
            return (
                Action(
                    steer=max(min(reference_steer, 0.9), -0.9),
                    throttle=min(max(float(reference_action.throttle), 0.55), 0.7),
                    brake=0.0,
                    gear=1,
                ),
                True,
            )
        if turn_demand > 0.35:
            return (
                Action(
                    steer=max(min(reference_steer, 0.65), -0.65),
                    throttle=min(max(float(reference_action.throttle), 0.65), 0.8),
                    brake=0.0,
                    gear=1,
                ),
                True,
            )
        if recovery_phase == 0:
            return (Action(steer=0.0, throttle=0.55, brake=0.0, gear=-1), True)
        return (
            Action(
                steer=max(min(reference_steer, 0.25), -0.25),
                throttle=1.0,
                brake=0.0,
                gear=1,
            ),
            True,
        )

    def _terminal_stop_action(self, obs: Observation) -> Action | None:
        radius = _goal_radius_from_info(obs.info)
        if radius <= 0.0:
            return None
        state = obs.vehicle_state
        distance = math.hypot(float(state.x) - float(obs.goal[0]), float(state.y) - float(obs.goal[1]))
        if distance <= radius:
            self._goal_hold_latched = True
        if not self._goal_hold_latched:
            return None
        self._stuck_steps = 0
        self._last_goal_distance = distance
        return Action(steer=0.0, throttle=0.0, brake=1.0, gear=1)


def _stabilize_action(action: Action, reference_action: Action, obs: Observation) -> Action:
    speed = max(0.0, float(obs.vehicle_state.speed))
    steer = max(min(float(action.steer), 1.0), -1.0)
    throttle = max(min(float(action.throttle), 1.0), 0.0)
    brake = max(min(float(action.brake), 1.0), 0.0)
    gear = int(action.gear) if action.gear is not None else 1
    reference_steer = max(min(float(reference_action.steer), 1.0), -1.0)
    turn_demand = max(abs(steer), abs(reference_steer))
    sharp_turn = abs(reference_steer) > 0.75
    if speed < 3.5:
        steer_conflict = abs(steer - reference_steer) > 0.45
        planner_stalling = throttle < 0.35 or brake > 0.01 or abs(steer) > 0.75
        if steer_conflict or planner_stalling:
            if sharp_turn:
                steer_limit = 0.9 if speed < 1.0 else 0.75
                throttle_floor = 0.42 if speed < 1.0 else 0.35
                throttle_cap = 0.5 if speed < 1.0 else 0.6
                return Action(
                    steer=max(min(reference_steer, steer_limit), -steer_limit),
                    throttle=min(max(throttle, float(reference_action.throttle), throttle_floor), throttle_cap),
                    brake=0.0,
                    gear=gear,
                )
            steer_limit = 0.25 if speed < 1.0 else 0.5
            return Action(
                steer=max(min(reference_steer, steer_limit), -steer_limit),
                throttle=max(throttle, float(reference_action.throttle), 0.78),
                brake=0.0,
                gear=gear,
            )
    if abs(reference_steer) > 0.35 and steer * reference_steer < -0.05:
        steer_limit = _speed_steer_limit(speed, sharp_turn=sharp_turn)
        steer = max(min(reference_steer, steer_limit), -steer_limit)
        turn_demand = max(abs(steer), abs(reference_steer))
    if turn_demand > 0.35:
        steer_limit = _speed_steer_limit(speed, sharp_turn=sharp_turn)
        steer = max(min(steer, steer_limit), -steer_limit)
        target_speed = _turn_speed_target(turn_demand)
        if speed > target_speed:
            overspeed = speed - target_speed
            throttle = 0.0 if overspeed >= 1.0 else min(throttle, 0.15)
            brake = max(brake, min(0.45, 0.08 + 0.08 * overspeed))
        elif speed > target_speed - 0.4:
            throttle = min(throttle, 0.2)
    if brake > 0.2 and throttle > 0.2:
        throttle = min(throttle, 0.2)
    return Action(steer=steer, throttle=throttle, brake=brake, gear=gear)

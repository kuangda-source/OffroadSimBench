"""Route-following world-model agent for visible simulator demos."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.world_model import WorldModelAgent
from offroad_sim.core import Action, Observation


class RouteWorldModelAgent(OffroadAgent):
    """Follow route waypoints while keeping model/planner selection switchable."""

    def __init__(
        self,
        route: list[tuple[float, float]] | None = None,
        waypoint_radius_m: float = 6.0,
        **world_model_kwargs: Any,
    ) -> None:
        self.route = _normalize_route(route)
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.cursor = 0
        world_model_kwargs.pop("seed", None)
        self.inner = WorldModelAgent(**world_model_kwargs)
        self._last_diagnostics: dict[str, Any] = {}

    def reset(self, scenario_info: Any) -> None:
        if isinstance(scenario_info, dict):
            route = scenario_info.get("route") or scenario_info.get("beamng_route")
            if route:
                self.route = _normalize_route(route)
        self.cursor = 0
        self.inner.reset(scenario_info)
        self._last_diagnostics = {}

    def act(self, obs: Observation) -> Action:
        if not self.route and obs.info.get("route"):
            self.route = _normalize_route(obs.info.get("route"))
            self.cursor = 0
        target = self._target_for(obs)
        routed_obs = replace(obs, goal=target) if target is not None else obs
        action = self.inner.act(routed_obs)
        inner_diagnostics = self.inner.diagnostics()
        self._last_diagnostics = {
            **inner_diagnostics,
            "route_length": len(self.route),
            "target_waypoint_index": self.cursor if self.route else None,
            "target_waypoint": list(target) if target is not None else None,
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        self.inner.close()

    def _target_for(self, obs: Observation) -> tuple[float, float] | None:
        if not self.route:
            return None
        state = obs.vehicle_state
        if self.cursor < len(self.route) - 1:
            waypoint = self.route[self.cursor]
            if math.hypot(state.x - waypoint[0], state.y - waypoint[1]) <= self.waypoint_radius_m:
                self.cursor += 1
        return self.route[min(self.cursor, len(self.route) - 1)]


def _normalize_route(route: Any) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    if route is None:
        return rows
    for point in route:
        try:
            rows.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return rows

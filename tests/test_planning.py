from __future__ import annotations

import numpy as np
import pytest

from offroad_sim.agents import WorldModelAgent
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning import NavigationMPCPlanner, StableWorldModelUnavailableError, WorldModelCEMPlanner, default_planner_registry
from offroad_sim.planning.stablewm import LeWMCEMPlanner
from offroad_sim.world_models import SimpleKinematicWorldModel


def _observation() -> Observation:
    return Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(8.0, 0.0),
    )


def test_planner_registry_reports_local_and_lewm_planners() -> None:
    registry = default_planner_registry()

    assert {"navigation_mpc", "world_model_cem", "le_wm_cem"}.issubset(set(registry.names()))
    assert registry.status("navigation_mpc").available is True
    assert registry.status("world_model_cem").available is True


def test_world_model_cem_plans_toward_goal() -> None:
    planner = WorldModelCEMPlanner(horizon=4, num_samples=16, iterations=2, seed=4)
    result = planner.plan(_observation(), SimpleKinematicWorldModel(), reference_action=Action(throttle=0.4))

    assert len(result.actions) == 4
    assert result.first_action.throttle >= 0.0
    assert result.predicted_states[-1].x > 0.0
    assert result.metadata["planner"] == "world_model_cem"


def test_navigation_mpc_plans_toward_off_axis_goal() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(0.0, 12.0),
    )
    planner = NavigationMPCPlanner(horizon=8, num_samples=32, seed=4)

    result = planner.plan(observation, SimpleKinematicWorldModel(), reference_action=Action(throttle=0.5))

    assert result.first_action.steer > 0.1
    assert result.first_action.throttle > 0.0
    assert result.predicted_states[-1].y > 0.0
    assert result.metadata["planner"] == "navigation_mpc"


def test_navigation_mpc_region_penalty_overrides_external_score() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=2.5),
        goal=(18.0, 0.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-2.0, -2.0], [25.0, -2.0], [25.0, 2.0], [-2.0, 2.0]]},
                "cost": {"out_of_region_weight": 5000.0, "boundary_weight": 20.0, "boundary_margin_m": 1.0},
            }
        },
    )
    planner = NavigationMPCPlanner(horizon=18, num_samples=40, seed=2, model_score_weight=1.0)

    def score_actions(candidates: list[list[Action]]) -> list[float]:
        return [-1000.0 if candidate[0].steer < -0.75 else 0.0 for candidate in candidates]

    result = planner.plan(
        observation,
        SimpleKinematicWorldModel(),
        reference_action=Action(throttle=0.6),
        score_actions=score_actions,
    )

    assert result.first_action.steer > -0.75
    assert result.metadata["external_score_used"] is True
    assert result.metadata["region_cost"] >= 0.0


def test_navigation_mpc_samples_deceleration_candidates() -> None:
    planner = NavigationMPCPlanner(horizon=5, num_samples=24, seed=2)

    candidates = planner._candidate_sequences(Action(steer=0.7, throttle=0.55))

    assert any(any(step.brake > 0.0 for step in candidate) for candidate in candidates)
    assert any(candidate[0].throttle == 0.0 for candidate in candidates)


def test_navigation_mpc_reports_world_model_prediction_fallback() -> None:
    class FailingWorldModel:
        def predict(self, observation, action, horizon=10):  # noqa: ANN001, ANN202
            raise RuntimeError("predict failed")

    planner = NavigationMPCPlanner(horizon=3, num_samples=8, seed=1)

    result = planner.plan(_observation(), FailingWorldModel(), reference_action=Action(throttle=0.4))  # type: ignore[arg-type]

    assert result.metadata["prediction_fallback"] == "simple_kinematic"
    assert "RuntimeError: predict failed" in result.metadata["prediction_error"]


def test_world_model_agent_can_use_planner() -> None:
    agent = WorldModelAgent(planner_name="world_model_cem", planner_config={"horizon": 3, "num_samples": 12, "iterations": 1})

    action = agent.act(_observation())
    diagnostics = agent.diagnostics()

    assert isinstance(action, Action)
    assert diagnostics["planner"] == "world_model_cem"
    assert diagnostics["best_cost"] >= 0.0


def test_lewm_planner_requires_checkpoint() -> None:
    planner = LeWMCEMPlanner()

    with pytest.raises(StableWorldModelUnavailableError, match="checkpoint"):
        planner.plan(_observation(), SimpleKinematicWorldModel())


def test_lewm_planner_accepts_direct_object_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from offroad_sim.planning.lewm_cost_model import LeWMCostModel

    checkpoint = tmp_path / "lewm_direct_object.ckpt"
    torch.save(LeWMCostModel().eval(), checkpoint)
    planner = LeWMCEMPlanner(checkpoint_path=checkpoint, horizon=2, num_samples=8, iterations=1, topk=2)

    result = planner.plan(_observation(), SimpleKinematicWorldModel())

    assert len(result.actions) == 2
    assert result.metadata["checkpoint_source_kind"] == "stablewm_object_file"
    assert result.metadata["checkpoint_object_path"] == str(checkpoint)


def test_lewm_planner_passes_navigation_region_to_cost_model() -> None:
    torch = pytest.importorskip("torch")
    planner = LeWMCEMPlanner()
    observation = _observation()
    observation.info["navigation_region"] = {
        "region": {"polygon": [[0.0, -2.0], [10.0, -2.0], [10.0, 2.0], [0.0, 2.0]]}
    }

    info = planner._observation_to_info(observation, torch)

    assert "region_polygon" in info
    assert tuple(info["region_polygon"].shape) == (1, 1, 4, 2)


def test_lewm_planner_clips_solver_actions_to_normalized_space(monkeypatch) -> None:
    class FakeTensor:
        def float(self):
            return self

    class FakeTorch:
        @staticmethod
        def from_numpy(value):
            return FakeTensor()

    class FakeSolver:
        def __call__(self, info):
            return {
                "actions": [np.asarray([[2.0, -0.5, -0.4], [-2.0, 1.5, 2.0]], dtype=np.float32)],
                "costs": [3.0],
            }

    planner = LeWMCEMPlanner(checkpoint_path="fake.ckpt", horizon=2)
    monkeypatch.setattr(planner, "_solver_for_checkpoint", lambda checkpoint: (FakeSolver(), FakeTorch()))

    result = planner.plan(_observation(), SimpleKinematicWorldModel())

    assert result.actions[0] == Action(steer=1.0, throttle=0.0, brake=0.0)
    assert result.actions[1] == Action(steer=-1.0, throttle=1.0, brake=1.0)

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from desktop_app import services
from offroad_sim.algorithms import DataPrepResult, TrainResult
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.replay import EpisodeRecorder


def _write_task(path: Path) -> None:
    path.write_text(
        """
task_id: beamng_region_nav_test
task_type: navigation_region_v1
map_id: gridmap_v2_region_001
backend_targets: [beamng]
level: gridmap_v2
region:
  polygon:
    - [0.0, -160.0]
    - [30.0, -160.0]
    - [30.0, -260.0]
    - [0.0, -260.0]
start_pose:
  pos: [1.0, -170.0, 100.6]
  yaw: -1.57
goal:
  pos: [4.0, -240.0]
  radius: 6.0
expert_route:
  - [1.0, -170.0]
  - [2.0, -205.0]
  - [4.0, -240.0]
constraints:
  max_steps: 120
  max_collision_count: 0
beamng:
  vehicle_model: pickup
  ai_line_speed: 10.0
  evaluation_route_mode: expert
""".strip(),
        encoding="utf-8",
    )


def _save_episode(path: Path, x: float, y: float, *, extra_points: list[tuple[float, float]] | None = None) -> Path:
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": path.name, "backend": "beamng"})
    points = [(x, y), *(extra_points or [])]
    for index, (px, py) in enumerate(points):
        recorder.record_step(
            observation=Observation(timestamp=float(index), vehicle_state=VehicleState(x=px, y=py, z=100.0, speed=0.0), goal=(4.0, -240.0)),
            action=Action(),
            reward=0.0,
            done=False,
            info={},
        )
    recorder.end_episode({"steps": len(points)})
    return recorder.save(path)


def test_region_navigation_closed_loop_uses_expert_only_for_collection(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(tmp_path / "collection", 2.0, -205.0)
    evaluation_episode = _save_episode(tmp_path / "evaluation", 4.5, -240.5, extra_points=[(15.0, -245.0)])
    seen_scenarios: list[dict[str, object]] = []

    def fake_run_episode(**kwargs):
        scenario = kwargs["scenario"]
        seen_scenarios.append(scenario)

        class Result:
            def to_dict(self):
                if scenario["scenario_id"].endswith("_collection"):
                    return {"episode_id": "collect", "episode_path": str(collection_episode), "metrics": {"horizontal_distance_traveled": 10.0}}
                return {"episode_id": "eval", "episode_path": str(evaluation_episode), "metrics": {"horizontal_distance_traveled": 20.0, "collision_count": 0}}

        return Result()

    class FakeAlgorithm:
        algorithm_id = "fake_region"

        def prepare_data(self, request):
            return DataPrepResult(output_path=str(tmp_path / "region.h5"), metadata={"total_frames": 3})

        def train(self, request):
            return TrainResult(output_dir=str(tmp_path / "model"), checkpoint_path=str(tmp_path / "model" / "lewm_cost_object.ckpt"))

    with (
        patch("desktop_app.services.run_episode", side_effect=fake_run_episode),
        patch("desktop_app.services.make_algorithm_adapter", return_value=FakeAlgorithm()),
    ):
        payload = services.run_region_navigation_closed_loop(
            services.RegionNavigationClosedLoopRequest(
                task_path=str(task_path),
                algorithm="fake_region",
                output_dir=str(tmp_path / "out"),
                collect_steps=12,
                eval_steps=8,
                close_beamng=True,
            )
        )

    collection_scenario = seen_scenarios[0]
    evaluation_scenario = seen_scenarios[1]
    assert collection_scenario["metadata"]["beamng"]["route"] == [[1.0, -170.0], [2.0, -205.0], [4.0, -240.0]]
    assert collection_scenario["metadata"]["beamng"]["drive_mode"] == "ai_line"
    assert evaluation_scenario["metadata"]["beamng"]["route"] == [[1.0, -170.0], [2.0, -205.0], [4.0, -240.0]]
    assert evaluation_scenario["metadata"]["beamng"]["drive_mode"] == "manual"
    assert payload["acceptance"]["goal_success"] is True
    assert payload["acceptance"]["model_controlled"] is True
    assert payload["acceptance"]["min_goal_distance"] < 1.0
    assert payload["acceptance"]["final_goal_distance"] > 10.0
    assert payload["acceptance"]["min_goal_step"] == 0
    assert payload["task"]["task_id"] == "beamng_region_nav_test"


def test_region_navigation_closed_loop_passes_planner_settings(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(tmp_path / "collection", 2.0, -205.0)
    evaluation_episode = _save_episode(tmp_path / "evaluation", 4.5, -240.5)
    seen_agent_options: list[dict[str, object]] = []

    def fake_run_episode(**kwargs):
        seen_agent_options.append(kwargs["agent_options"])

        class Result:
            def to_dict(self):
                if len(seen_agent_options) == 1:
                    return {"episode_id": "collect", "episode_path": str(collection_episode), "metrics": {"horizontal_distance_traveled": 10.0}}
                return {
                    "episode_id": "eval",
                    "episode_path": str(evaluation_episode),
                    "metrics": {"horizontal_distance_traveled": 20.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    class FakeAlgorithm:
        algorithm_id = "fake_region"

        def prepare_data(self, request):
            return DataPrepResult(output_path=str(tmp_path / "region.h5"), metadata={"total_frames": 3})

        def train(self, request):
            return TrainResult(output_dir=str(tmp_path / "model"), checkpoint_path=str(tmp_path / "model" / "lewm_cost_object.ckpt"))

    with (
        patch("desktop_app.services.run_episode", side_effect=fake_run_episode),
        patch("desktop_app.services.make_algorithm_adapter", return_value=FakeAlgorithm()),
    ):
        services.run_region_navigation_closed_loop(
            services.RegionNavigationClosedLoopRequest(
                task_path=str(task_path),
                algorithm="fake_region",
                output_dir=str(tmp_path / "out"),
                collect_steps=12,
                eval_steps=8,
                planner_horizon=7,
                planner_samples=24,
                planner_iterations=3,
                close_beamng=True,
            )
        )

    assert seen_agent_options[1]["planner_config"] == {"horizon": 7, "num_samples": 24, "iterations": 3}

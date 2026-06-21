from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from desktop_app import services
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.replay import EpisodeRecorder


def _write_task(path: Path) -> None:
    path.write_text(
        """
task_id: self_supervised_region_test
task_type: navigation_region_v1
map_id: gridmap_v2_region_001
backend_targets: [beamng]
level: gridmap_v2
region:
  polygon:
    - [0.0, 0.0]
    - [40.0, 0.0]
    - [40.0, 40.0]
    - [0.0, 40.0]
start_pose:
  pos: [2.0, 2.0, 1.0]
  yaw: 0.0
goal:
  pos: [35.0, 35.0]
  radius: 5.0
expert_route:
  - [2.0, 2.0]
  - [10.0, 30.0]
  - [35.0, 35.0]
constraints:
  max_steps: 120
  max_collision_count: 0
beamng:
  evaluation_route_mode: expert
  evaluation_drive_mode: manual
""".strip(),
        encoding="utf-8",
    )


def _save_episode(path: Path, points: list[tuple[float, float, float, float]]) -> Path:
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": path.name, "backend": "beamng"})
    for index, (x, y, yaw, speed) in enumerate(points):
        recorder.record_step(
            observation=Observation(
                timestamp=float(index),
                vehicle_state=VehicleState(x=x, y=y, z=1.0, yaw=yaw, speed=speed),
                goal=(35.0, 35.0),
            ),
            action=Action(steer=0.1, throttle=0.3, brake=0.0),
            reward=0.0,
            done=False,
            info={},
        )
    recorder.end_episode({"steps": len(points), "collision_count": 0})
    return recorder.save(path)


def test_region_self_supervised_world_model_trains_and_evaluates_without_route(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (4.0, 2.2, 0.1, 1.0), (7.0, 2.8, 0.18, 1.4), (10.0, 4.0, 0.25, 1.2)],
    )
    evaluation_episode = _save_episode(
        tmp_path / "evaluation",
        [(2.0, 2.0, 0.0, 0.5), (20.0, 18.0, 0.4, 2.0), (34.0, 34.0, 0.6, 1.0)],
    )
    seen_scenarios: list[dict[str, object]] = []
    seen_agents: list[str] = []
    seen_agent_options: list[dict[str, object]] = []

    def fake_run_episode(**kwargs):
        seen_scenarios.append(kwargs["scenario"])
        seen_agents.append(kwargs["agent_name"])
        seen_agent_options.append(kwargs["agent_options"])

        class Result:
            def to_dict(self):
                if len(seen_agents) == 1:
                    return {
                        "episode_id": "collect",
                        "episode_path": str(collection_episode),
                        "metrics": {"horizontal_distance_traveled": 8.0, "collision_count": 0, "drive_mode": "manual"},
                    }
                return {
                    "episode_id": "eval",
                    "episode_path": str(evaluation_episode),
                    "metrics": {"horizontal_distance_traveled": 40.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.run_region_self_supervised_world_model(
            services.RegionSelfSupervisedWorldModelRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "out"),
                collect_steps=20,
                eval_steps=20,
                close_beamng=True,
            )
        )

    assert seen_agents == ["region_explorer", "world_model_direct"]
    assert "route" not in seen_scenarios[0]["metadata"]["beamng"]
    assert "route" not in seen_scenarios[1]["metadata"]["beamng"]
    assert seen_agent_options[1]["world_model_name"] == "tiny_learned"
    assert seen_agent_options[1]["planner_config"] == {"horizon": 6, "num_samples": 32, "iterations": 3}
    assert Path(payload["training"]["model_path"]).exists()
    assert payload["training"]["model_type"] == "tiny_learned"
    assert payload["region_navigation"]["evaluation_agent"] == "world_model_direct"
    assert payload["acceptance"]["goal_success"] is True
    assert payload["acceptance"]["route_waypoint_count"] == 0
    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert training_record["preset_id"] == "region_self_supervised_world_model"
    assert training_record["artifact_path"] == payload["model_dir"]
    assert training_record["metrics"]["goal_success"] is True
    assert training_record["metrics"]["min_goal_distance"] <= 5.0
    assert training_record["metrics"]["collection_min_goal_distance"] > 30.0
    assert training_record["metrics"]["collection_distance_traveled"] == 8.0
    assert training_record["history"]["collection_min_goal_distance"] == [training_record["metrics"]["collection_min_goal_distance"]]


def test_region_self_supervised_world_model_trains_from_multiple_collection_rollouts(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_a = _save_episode(
        tmp_path / "collection_a",
        [(2.0, 2.0, 0.0, 0.5), (8.0, 4.0, 0.2, 1.2), (15.0, 8.0, 0.35, 1.5)],
    )
    collection_b = _save_episode(
        tmp_path / "collection_b",
        [(2.0, 2.0, 0.0, 0.5), (12.0, 18.0, 0.4, 1.5), (28.0, 30.0, 0.55, 1.8)],
    )
    evaluation_episode = _save_episode(
        tmp_path / "evaluation",
        [(2.0, 2.0, 0.0, 0.5), (22.0, 22.0, 0.4, 2.0), (35.0, 35.0, 0.6, 1.0)],
    )
    episodes = [collection_a, collection_b, evaluation_episode]
    distances = [14.0, 35.0, 46.0]
    seen_agents: list[str] = []

    def fake_run_episode(**kwargs):
        seen_agents.append(kwargs["agent_name"])
        index = len(seen_agents) - 1

        class Result:
            def to_dict(self):
                return {
                    "episode_id": f"episode_{index}",
                    "episode_path": str(episodes[index]),
                    "metrics": {"horizontal_distance_traveled": distances[index], "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.run_region_self_supervised_world_model(
            services.RegionSelfSupervisedWorldModelRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "out"),
                collect_steps=20,
                collect_rollouts=2,
                eval_steps=20,
                close_beamng=True,
            )
        )

    assert seen_agents == ["region_explorer", "region_explorer", "world_model_direct"]
    assert payload["training"]["metrics"]["sequence_count"] == 2
    assert len(payload["collections"]) == 2
    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert training_record["metrics"]["collection_rollout_count"] == 2
    assert training_record["metrics"]["collection_distance_traveled"] == 49.0
    assert training_record["metrics"]["collection_min_goal_distance"] <= 10.0


def test_region_self_supervised_world_model_stops_when_collection_makes_no_goal_progress(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (3.0, 2.0, 0.0, 0.8), (4.0, 2.0, 0.0, 0.8)],
    )
    seen_agents: list[str] = []

    def fake_run_episode(**kwargs):
        seen_agents.append(kwargs["agent_name"])

        class Result:
            def to_dict(self):
                return {
                    "episode_id": "collect",
                    "episode_path": str(collection_episode),
                    "metrics": {"horizontal_distance_traveled": 2.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.run_region_self_supervised_world_model(
            services.RegionSelfSupervisedWorldModelRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "out"),
                collect_steps=20,
                collect_rollouts=1,
                min_collection_goal_progress_ratio=0.5,
                eval_steps=20,
                close_beamng=True,
            )
        )

    assert seen_agents == ["region_explorer"]
    assert payload["status"] == "collection_insufficient"
    assert payload["quality_gate"]["passed"] is False
    assert payload["training"]["status"] == "skipped"
    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert training_record["status"] == "collection_insufficient"
    assert training_record["metrics"]["collection_progress_ratio"] < 0.5
    assert training_record["summary"]["quality_gate"]["reason"] == "collection_goal_progress_below_threshold"

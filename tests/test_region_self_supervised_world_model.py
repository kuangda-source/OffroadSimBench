from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.run_region_self_supervised_world_model as region_self_supervised_script
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


def _save_episode_with_controls(
    path: Path,
    points: list[tuple[float, float, float, float, int | None, bool]],
) -> Path:
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": path.name, "backend": "beamng"})
    for index, (x, y, yaw, speed, gear, recovery) in enumerate(points):
        recorder.record_step(
            observation=Observation(
                timestamp=float(index),
                vehicle_state=VehicleState(x=x, y=y, z=1.0, yaw=yaw, speed=speed),
                goal=(35.0, 35.0),
            ),
            action=Action(steer=0.1, throttle=0.3, brake=0.0, gear=gear),
            reward=0.0,
            done=False,
            info={"agent_diagnostics": {"stuck_recovery": recovery, "executed_action": {"gear": gear}}},
        )
    recorder.end_episode({"steps": len(points), "collision_count": 0, "drive_mode": "manual"})
    return recorder.save(path)


def test_region_self_supervised_world_model_defaults_match_route_free_demo_probe() -> None:
    request = services.RegionSelfSupervisedWorldModelRequest(task_path="configs/tasks/beamng_johnson_valley_nav_test.yaml")

    assert request.eval_steps == 1200
    assert request.evaluation_local_subgoal_distance_m == 12.0
    assert request.evaluation_allow_reverse_recovery is False
    assert request.evaluation_reverse_recovery_after_steps == 96
    assert request.use_experience_corridor is True


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
                register_world_model_config=True,
                world_model_config_path=str(tmp_path / "world_model_configs.json"),
                collect_steps=20,
                eval_steps=20,
                close_beamng=True,
            )
        )

    assert seen_agents == ["region_explorer", "world_model_direct"]
    assert "route" not in seen_scenarios[0]["metadata"]["beamng"]
    assert "route" not in seen_scenarios[1]["metadata"]["beamng"]
    assert seen_agent_options[0]["goal_bias_interval"] == 1
    assert seen_agent_options[0]["goal_corridor_interval"] == 1
    assert seen_agent_options[0]["coverage_grid_size"] == 0
    assert seen_agent_options[0]["coverage_target_interval"] == 0
    assert seen_agent_options[0]["max_target_steps"] == 80
    assert seen_agent_options[1]["world_model_name"] == "tiny_learned"
    assert seen_agent_options[1]["planner_config"] == {"horizon": 6, "num_samples": 32, "iterations": 3}
    assert seen_agent_options[1]["allow_reverse_recovery"] is False
    assert seen_agent_options[1]["reverse_recovery_after_steps"] == 96
    assert seen_agent_options[1]["local_subgoal_distance_m"] == 12.0
    assert Path(payload["training"]["model_path"]).exists()
    assert payload["training"]["model_type"] == "tiny_learned"
    assert payload["region_navigation"]["evaluation_agent"] == "world_model_direct"
    assert payload["acceptance"]["goal_success"] is True
    assert payload["acceptance"]["route_waypoint_count"] == 0
    assert Path(payload["trajectory_plot_path"]).exists()
    trajectory_svg = Path(payload["trajectory_plot_path"]).read_text(encoding="utf-8")
    assert "collection" in trajectory_svg
    assert "route_free" in trajectory_svg
    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert training_record["preset_id"] == "region_self_supervised_world_model"
    assert training_record["artifact_path"] == payload["model_dir"]
    assert training_record["summary"]["trajectory_plot_path"] == payload["trajectory_plot_path"]
    assert training_record["metrics"]["goal_success"] is True
    assert training_record["metrics"]["min_goal_distance"] <= 5.0
    assert training_record["metrics"]["collection_min_goal_distance"] > 30.0
    assert training_record["metrics"]["collection_distance_traveled"] == 8.0
    assert training_record["metrics"]["validation_rmse"] == payload["training"]["metrics"]["validation_rmse"]
    assert training_record["metrics"]["validation_sample_count"] == payload["training"]["metrics"]["validation_sample_count"]
    assert training_record["metrics"]["segment_rmse"] == payload["training"]["metrics"]["segment_rmse"]
    assert training_record["metrics"]["segment_sample_count"] == payload["training"]["metrics"]["segment_sample_count"]
    assert training_record["history"]["validation_rmse"] == [payload["training"]["metrics"]["validation_rmse"]]
    assert training_record["history"]["collection_min_goal_distance"] == [training_record["metrics"]["collection_min_goal_distance"]]
    config = payload["world_model_config"]
    saved_config = next(row for row in services.world_model_config_entries(tmp_path / "world_model_configs.json") if row["id"] == config["id"])
    assert config["algorithm"] == "world_model_direct"
    assert saved_config["world_model"] == "tiny_learned"
    assert saved_config["model_path"] == payload["model_dir"]
    assert saved_config["source_training_run_path"] == payload["training_run_path"]
    assert saved_config["demo_ready"] is False
    assert saved_config["validation"]["demo_ready"] is False
    assert saved_config["validation"]["goal_success"] is True
    assert saved_config["validation"]["route_free"] is True
    assert saved_config["validation"]["route_free_direct"] is False
    assert saved_config["validation"]["experience_corridor"] is True
    assert saved_config["validation"]["evaluation_route_mode"] == "route_free"
    assert saved_config["validation"]["route_waypoint_count"] == 0
    assert saved_config["validation"]["model_controlled"] is True
    assert saved_config["validation"]["validation_rmse"] == payload["training"]["metrics"]["validation_rmse"]
    assert saved_config["validation"]["segment_rmse"] == payload["training"]["metrics"]["segment_rmse"]
    refreshed_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert refreshed_record["summary"]["world_model_config"]["id"] == config["id"]
    assert refreshed_record["summary"]["world_model_config"]["model_path"] == payload["model_dir"]
    assert refreshed_record["summary"]["world_model_config"]["validation"]["goal_success"] is True
    assert refreshed_record["summary"]["world_model_config"]["validation"]["validation_rmse"] == payload["training"]["metrics"]["validation_rmse"]


def test_region_self_supervised_world_model_adds_experience_corridor_without_expert_route_leakage(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [
            (2.0, 2.0, 0.0, 0.5),
            (2.0, 12.0, 0.4, 1.0),
            (2.0, 24.0, 0.6, 1.2),
            (12.0, 30.0, 0.4, 1.4),
            (24.0, 34.0, 0.2, 1.5),
            (34.0, 35.0, 0.0, 0.8),
        ],
    )
    evaluation_episode = _save_episode(
        tmp_path / "evaluation",
        [(2.0, 2.0, 0.0, 0.5), (12.0, 30.0, 0.4, 2.0), (34.0, 35.0, 0.6, 1.0)],
    )
    seen_scenarios: list[dict[str, object]] = []

    def fake_run_episode(**kwargs):
        seen_scenarios.append(kwargs["scenario"])

        class Result:
            def to_dict(self):
                episode_path = collection_episode if len(seen_scenarios) == 1 else evaluation_episode
                return {
                    "episode_id": f"episode_{len(seen_scenarios)}",
                    "episode_path": str(episode_path),
                    "metrics": {"horizontal_distance_traveled": 45.0, "collision_count": 0, "drive_mode": "manual"},
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

    evaluation_scenario = seen_scenarios[1]
    evaluation_beamng = evaluation_scenario["metadata"]["beamng"]
    evaluation_task = evaluation_scenario["metadata"]["task"]
    experience_route = evaluation_task["experience_route"]

    assert "route" not in evaluation_beamng
    assert "expert_route" not in evaluation_task
    assert len(experience_route) >= 4
    assert experience_route[0] == [2.0, 2.0]
    assert experience_route[-1] == [35.0, 35.0]
    assert [2.0, 24.0] in experience_route
    assert payload["region_navigation"]["experience_corridor"] is True
    assert payload["region_navigation"]["experience_route_point_count"] == len(experience_route)
    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert training_record["metrics"]["experience_route_point_count"] == len(experience_route)
    assert training_record["parameters"]["use_experience_corridor"] is True


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
                collection_coverage_grid_size=4,
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
    assert training_record["metrics"]["collection_coverage_cell_count"] == 4
    assert training_record["metrics"]["collection_coverage_total_cells"] == 16
    assert training_record["metrics"]["collection_coverage_ratio"] == 0.25
    assert training_record["summary"]["coverage"]["ratio"] == 0.25


def test_region_training_data_collection_writes_reusable_collection_manifest(tmp_path: Path) -> None:
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
    episodes = [collection_a, collection_b]
    seen_agents: list[str] = []

    def fake_run_episode(**kwargs):
        seen_agents.append(kwargs["agent_name"])
        index = len(seen_agents) - 1

        class Result:
            def to_dict(self):
                return {
                    "episode_id": f"collection_{index}",
                    "episode_path": str(episodes[index]),
                    "metrics": {"horizontal_distance_traveled": 20.0 + index, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.collect_region_training_data(
            services.RegionTrainingDataCollectionRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "collection_run"),
                collect_steps=20,
                collect_rollouts=2,
                close_beamng=True,
            )
        )

    manifest_path = Path(payload["collection_manifest_path"])
    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert seen_agents == ["region_explorer", "region_explorer"]
    assert payload["status"] == "completed"
    assert manifest_path.exists()
    assert len(payload["episode_paths"]) == 2
    assert manifest["episode_paths"] == payload["episode_paths"]
    assert manifest["task"]["task_id"] == "self_supervised_region_test"
    assert training_record["preset_id"] == "beamng_region_training_data"
    assert training_record["artifact_type"] == "beamng_collection"
    assert training_record["artifact_path"] == str(manifest_path.resolve())
    assert training_record["metrics"]["collection_rollout_count"] == 2
    assert training_record["metrics"]["collection_distance_traveled"] == 41.0
    assert training_record["metrics"]["collection_coverage_cell_count"] == 4
    assert training_record["metrics"]["collection_coverage_total_cells"] == 16
    assert training_record["metrics"]["collection_coverage_ratio"] == 0.25
    assert manifest["metrics"]["collection_coverage_ratio"] == 0.25


def test_region_training_data_collection_retries_beamng_reconnect_errors(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (12.0, 12.0, 0.2, 1.2), (25.0, 25.0, 0.45, 1.5)],
    )
    calls = 0

    def fake_run_region_beamng_episode(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("BNGDisconnectedError: Connecting to the simulator failed.")
        return {
            "episode_id": "collection_retry",
            "episode_path": str(collection_episode),
            "metrics": {"horizontal_distance_traveled": 30.0, "collision_count": 0, "drive_mode": "manual"},
        }

    with (
        patch("desktop_app.services._run_region_beamng_episode", side_effect=fake_run_region_beamng_episode),
        patch("desktop_app.services.time.sleep", return_value=None),
    ):
        payload = services.collect_region_training_data(
            services.RegionTrainingDataCollectionRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "collection_run"),
                collect_steps=20,
                collect_rollouts=1,
                close_beamng=True,
            )
        )

    assert calls == 2
    assert payload["status"] == "completed"
    assert payload["episode_paths"] == [str(collection_episode.resolve())]


def test_region_world_model_training_from_collection_registers_model_config(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (12.0, 12.0, 0.2, 1.2), (25.0, 25.0, 0.45, 1.5)],
    )
    collection_payload = {
        "status": "completed",
        "task": services.load_navigation_region_task(str(task_path)).to_dict(),
        "task_path": str(task_path),
        "output_dir": str(tmp_path / "collection_run"),
        "episode_paths": [str(collection_episode.resolve())],
        "collection_acceptances": [{"min_goal_distance": 14.0, "final_goal_distance": 14.0, "collision_count": 0}],
        "metrics": {"collection_rollout_count": 1, "collection_distance_traveled": 25.0},
    }
    manifest_path = tmp_path / "collection_run" / "region_training_collection.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(collection_payload, indent=2), encoding="utf-8")

    payload = services.train_region_world_model_from_collection(
        services.RegionWorldModelTrainingRequest(
            collection_manifest_path=str(manifest_path),
            output_dir=str(tmp_path / "trained_model"),
            register_world_model_config=True,
            world_model_config_path=str(tmp_path / "world_model_configs.json"),
        )
    )

    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    saved_config = next(
        row for row in services.world_model_config_entries(tmp_path / "world_model_configs.json") if row["id"] == payload["world_model_config"]["id"]
    )

    assert payload["status"] == "completed"
    assert payload["training"]["model_type"] == "tiny_learned"
    assert Path(payload["model_dir"], "model.json").exists()
    assert training_record["preset_id"] == "region_world_model_training"
    assert training_record["dataset_root"] == str(manifest_path.resolve())
    assert training_record["artifact_type"] == "world_model"
    assert training_record["artifact_path"] == payload["model_dir"]
    assert training_record["metrics"]["train_rmse"] == payload["training"]["metrics"]["train_rmse"]
    assert training_record["metrics"]["validation_rmse"] == payload["training"]["metrics"]["validation_rmse"]
    assert training_record["metrics"]["validation_sample_count"] == payload["training"]["metrics"]["validation_sample_count"]
    assert training_record["metrics"]["segment_rmse"] == payload["training"]["metrics"]["segment_rmse"]
    assert training_record["metrics"]["segment_sample_count"] == payload["training"]["metrics"]["segment_sample_count"]
    assert training_record["history"]["validation_rmse"] == [payload["training"]["metrics"]["validation_rmse"]]
    assert saved_config["algorithm"] == "world_model_direct"
    assert saved_config["world_model"] == "tiny_learned"
    assert saved_config["model_path"] == payload["model_dir"]
    assert saved_config["source_training_run_path"] == payload["training_run_path"]
    assert saved_config["validation"]["validation_rmse"] == payload["training"]["metrics"]["validation_rmse"]
    assert saved_config["validation"]["segment_rmse"] == payload["training"]["metrics"]["segment_rmse"]


def test_region_world_model_training_refuses_insufficient_collection_manifest(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_payload = {
        "status": "completed",
        "task": services.load_navigation_region_task(str(task_path)).to_dict(),
        "task_path": str(task_path),
        "output_dir": str(tmp_path / "collection_run"),
        "episode_paths": [],
        "quality_gate": {"passed": False, "reason": "collection_route_coverage_below_threshold"},
        "metrics": {"route_coverage_ratio": 0.2, "goal_zone_coverage": 0.0},
    }
    manifest_path = tmp_path / "collection_run" / "region_training_collection.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(collection_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Collection quality gate failed"):
        services.train_region_world_model_from_collection(
            services.RegionWorldModelTrainingRequest(
                collection_manifest_path=str(manifest_path),
                output_dir=str(tmp_path / "trained_model"),
            )
        )


def test_region_self_supervised_world_model_records_navigation_diagnostics_when_eval_misses_goal(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (12.0, 12.0, 0.2, 1.2), (25.0, 25.0, 0.45, 1.5)],
    )
    evaluation_episode = _save_episode(
        tmp_path / "evaluation",
        [(2.0, 2.0, 0.0, 0.5), (7.0, 3.0, 0.05, 1.0), (9.0, 4.0, 0.02, 0.7)],
    )
    episodes = [collection_episode, evaluation_episode]
    seen_agents: list[str] = []

    def fake_run_episode(**kwargs):
        seen_agents.append(kwargs["agent_name"])
        index = len(seen_agents) - 1

        class Result:
            def to_dict(self):
                return {
                    "episode_id": f"episode_{index}",
                    "episode_path": str(episodes[index]),
                    "metrics": {"horizontal_distance_traveled": 25.0, "collision_count": 0, "drive_mode": "manual"},
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

    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    diagnostics = payload["diagnostics"]

    assert payload["status"] == "completed"
    assert payload["acceptance"]["goal_success"] is False
    assert diagnostics["status"] == "training_coverage_insufficient"
    assert diagnostics["evidence"]["route_free"] is True
    assert diagnostics["evidence"]["model_controlled"] is True
    assert diagnostics["evidence"]["min_goal_distance"] > payload["acceptance"]["goal_radius"]
    assert diagnostics["evidence"]["segment_sample_count"]["middle"] == 0
    assert diagnostics["evidence"]["segment_sample_count"]["goal"] == 0
    assert any("middle/goal" in item.lower() for item in diagnostics["next_actions"])
    assert training_record["summary"]["diagnostics"] == diagnostics


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
                register_world_model_config=True,
                world_model_config_path=str(tmp_path / "world_model_configs.json"),
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
    assert "world_model_config" not in training_record["summary"]
    assert payload.get("world_model_config") in ({}, None)
    assert not (tmp_path / "world_model_configs.json").exists()


def test_region_self_supervised_world_model_marks_too_short_collection_episode_insufficient(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection_short",
        [(35.0, 35.0, 0.0, 0.0)],
    )
    seen_agents: list[str] = []

    def fake_run_episode(**kwargs):
        seen_agents.append(kwargs["agent_name"])

        class Result:
            def to_dict(self):
                return {
                    "episode_id": "collect_short",
                    "episode_path": str(collection_episode),
                    "metrics": {"horizontal_distance_traveled": 0.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.run_region_self_supervised_world_model(
            services.RegionSelfSupervisedWorldModelRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "out"),
                collect_steps=20,
                collect_rollouts=1,
                close_beamng=True,
            )
        )

    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))

    assert seen_agents == ["region_explorer"]
    assert payload["status"] == "collection_insufficient"
    assert payload["quality_gate"]["reason"] == "collection_episode_too_short"
    assert payload["training"]["status"] == "skipped"
    assert training_record["status"] == "collection_insufficient"
    assert training_record["summary"]["diagnostics"]["status"] == "collection_insufficient"


def test_region_episode_sequence_records_global_task_segment_reference(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    episode = _save_episode(
        tmp_path / "collection_middle",
        [(10.0, 30.0, 0.0, 0.5), (22.0, 32.0, 0.2, 1.2), (35.0, 35.0, 0.45, 1.5)],
    )
    task = services.load_navigation_region_task(str(task_path))

    sequence = services._episode_trace_to_dataset_sequence(episode, task)

    assert sequence.metadata["task_start_pos"] == [2.0, 2.0]
    assert sequence.metadata["task_goal_pos"] == [35.0, 35.0]
    assert sequence.goal == (35.0, 35.0)


def test_region_world_model_evaluation_loads_existing_model_without_training(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    evaluation_episode = _save_episode(
        tmp_path / "evaluation",
        [(2.0, 2.0, 0.0, 0.5), (20.0, 18.0, 0.4, 2.0), (34.0, 34.0, 0.6, 1.0)],
    )
    seen: dict[str, object] = {}

    def fake_run_episode(**kwargs):
        seen.update(kwargs)

        class Result:
            def to_dict(self):
                return {
                    "episode_id": "eval",
                    "episode_path": str(evaluation_episode),
                    "metrics": {"horizontal_distance_traveled": 40.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.run_region_world_model_evaluation(
            services.RegionWorldModelEvaluationRequest(
                task_path=str(task_path),
                world_model_type="tiny_learned",
                world_model_path=str(tmp_path / "model"),
                output_dir=str(tmp_path / "out"),
                eval_steps=20,
                close_beamng=True,
            )
        )

    assert seen["agent_name"] == "world_model_direct"
    assert seen["agent_options"]["world_model_name"] == "tiny_learned"
    assert seen["agent_options"]["world_model_path"] == str(tmp_path / "model")
    assert "route" not in seen["scenario"]["metadata"]["beamng"]
    assert payload["status"] == "completed"
    assert payload["acceptance"]["goal_success"] is True
    assert payload["region_navigation"]["evaluation_agent"] == "world_model_direct"


def test_region_world_model_evaluation_compares_route_free_and_route_guided_baselines(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    route_free_episode = _save_episode_with_controls(
        tmp_path / "route_free",
        [
            (2.0, 2.0, 0.0, 0.5, None, False),
            (6.0, 3.0, 0.1, 0.8, -1, True),
            (8.0, 4.0, 0.1, 0.6, None, False),
        ],
    )
    route_guided_episode = _save_episode_with_controls(
        tmp_path / "route_guided",
        [
            (2.0, 2.0, 0.0, 0.5, None, False),
            (10.0, 30.0, 0.6, 1.5, None, False),
            (35.0, 35.0, 0.6, 0.8, None, False),
        ],
    )
    episodes = [route_free_episode, route_guided_episode]
    seen_agents: list[str] = []
    seen_scenarios: list[dict[str, object]] = []

    def fake_run_episode(**kwargs):
        seen_agents.append(kwargs["agent_name"])
        seen_scenarios.append(kwargs["scenario"])
        index = len(seen_agents) - 1

        class Result:
            def to_dict(self):
                return {
                    "episode_id": f"eval_{index}",
                    "episode_path": str(episodes[index]),
                    "metrics": {
                        "horizontal_distance_traveled": 10.0 + index * 60.0,
                        "collision_count": 0,
                        "drive_mode": "manual",
                        "route_waypoint_count": 0 if index == 0 else 3,
                    },
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.run_region_world_model_evaluation(
            services.RegionWorldModelEvaluationRequest(
                task_path=str(task_path),
                world_model_type="tiny_learned",
                world_model_path=str(tmp_path / "model"),
                output_dir=str(tmp_path / "out"),
                eval_steps=20,
                close_beamng=True,
                include_route_guided_baseline=True,
            )
        )

    assert seen_agents == ["world_model_direct", "route_world_model"]
    assert "route" not in seen_scenarios[0]["metadata"]["beamng"]
    assert seen_scenarios[1]["metadata"]["beamng"]["route"] == [[2.0, 2.0], [10.0, 30.0], [35.0, 35.0]]
    assert payload["baselines"]["route_free"]["acceptance"]["goal_success"] is False
    assert payload["baselines"]["route_guided"]["acceptance"]["goal_success"] is True
    assert payload["comparison"]["route_free_goal_success"] is False
    assert payload["comparison"]["route_guided_goal_success"] is True
    assert payload["comparison"]["route_free_reverse_count"] == 1
    assert payload["comparison"]["route_free_stuck_recovery_count"] == 1
    assert payload["comparison"]["route_guided_final_goal_distance"] <= 5.0
    assert Path(payload["trajectory_plot_path"]).exists()
    assert "route_free" in Path(payload["trajectory_plot_path"]).read_text(encoding="utf-8")
    assert json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))["comparison"] == payload["comparison"]


def test_region_training_data_collection_can_use_route_aware_curriculum(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (10.0, 30.0, 0.2, 1.2), (35.0, 35.0, 0.45, 1.5)],
    )
    seen_scenarios: list[dict[str, object]] = []
    seen_agent_options: list[dict[str, object]] = []

    def fake_run_episode(**kwargs):
        seen_scenarios.append(kwargs["scenario"])
        seen_agent_options.append(kwargs["agent_options"])

        class Result:
            def to_dict(self):
                return {
                    "episode_id": "collection",
                    "episode_path": str(collection_episode),
                    "metrics": {"horizontal_distance_traveled": 50.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.collect_region_training_data(
            services.RegionTrainingDataCollectionRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "collection_run"),
                collect_steps=20,
                collect_rollouts=1,
                collection_strategy="route_aware",
                collection_route_target_interval=1,
                collection_route_lateral_m=1.5,
                min_route_coverage_ratio=0.8,
                min_goal_zone_coverage=1.0,
                close_beamng=True,
            )
        )

    assert seen_scenarios[0]["metadata"]["beamng"]["route"] == [[2.0, 2.0], [10.0, 30.0], [35.0, 35.0]]
    assert seen_agent_options[0]["route_target_interval"] == 1
    assert seen_agent_options[0]["route_lateral_m"] == 1.5
    assert payload["quality_gate"]["passed"] is True
    assert payload["metrics"]["route_coverage_ratio"] == 1.0
    assert payload["metrics"]["goal_zone_coverage"] == 1.0
    assert payload["metrics"]["collection_min_goal_distance"] <= 5.0


def test_route_aware_collection_can_spawn_multiple_starts_along_expert_route(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    episodes = [
        _save_episode(tmp_path / f"collection_{index}", [(2.0, 2.0, 0.0, 0.5), (35.0, 35.0, 0.45, 1.5)])
        for index in range(3)
    ]
    seen_starts: list[list[float]] = []
    seen_routes: list[list[list[float]]] = []
    seen_task_routes: list[list[list[float]]] = []

    def fake_run_episode(**kwargs):
        beamng = kwargs["scenario"]["metadata"]["beamng"]
        task_metadata = kwargs["scenario"]["metadata"]["task"]
        seen_starts.append(list(beamng["vehicle_start"]["pos"]))
        seen_routes.append([list(point) for point in beamng["route"]])
        seen_task_routes.append([list(point) for point in task_metadata["expert_route"]])
        index = len(seen_starts) - 1

        class Result:
            def to_dict(self):
                return {
                    "episode_id": f"collection_{index}",
                    "episode_path": str(episodes[index]),
                    "metrics": {"horizontal_distance_traveled": 50.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        services.collect_region_training_data(
            services.RegionTrainingDataCollectionRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "collection_run"),
                collect_steps=20,
                collect_rollouts=3,
                collection_strategy="route_aware",
                collection_route_target_interval=1,
                collection_multi_start=True,
                close_beamng=True,
            )
        )

    assert [point[:2] for point in seen_starts] == [[2.0, 2.0], [10.0, 30.0], [10.0, 30.0]]
    assert all(math.hypot(point[0] - 35.0, point[1] - 35.0) > 5.0 for point in seen_starts)
    assert seen_routes == [
        [[2.0, 2.0], [10.0, 30.0], [35.0, 35.0]],
        [[10.0, 30.0], [35.0, 35.0]],
        [[10.0, 30.0], [35.0, 35.0]],
    ]
    assert seen_task_routes == seen_routes


def test_region_training_data_collection_marks_insufficient_quality_gate(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    collection_episode = _save_episode(
        tmp_path / "collection",
        [(2.0, 2.0, 0.0, 0.5), (3.0, 2.0, 0.0, 0.5), (4.0, 2.0, 0.0, 0.5)],
    )

    def fake_run_episode(**kwargs):
        class Result:
            def to_dict(self):
                return {
                    "episode_id": "collection",
                    "episode_path": str(collection_episode),
                    "metrics": {"horizontal_distance_traveled": 2.0, "collision_count": 0, "drive_mode": "manual"},
                }

        return Result()

    with patch("desktop_app.services.run_episode", side_effect=fake_run_episode):
        payload = services.collect_region_training_data(
            services.RegionTrainingDataCollectionRequest(
                task_path=str(task_path),
                output_dir=str(tmp_path / "collection_run"),
                collect_steps=20,
                collect_rollouts=1,
                collection_strategy="route_aware",
                collection_route_target_interval=1,
                min_route_coverage_ratio=0.8,
                min_goal_zone_coverage=1.0,
                close_beamng=True,
            )
        )

    training_record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))

    assert payload["status"] == "collection_insufficient"
    assert payload["quality_gate"]["passed"] is False
    assert payload["quality_gate"]["reason"] in {
        "collection_goal_progress_below_threshold",
        "collection_route_coverage_below_threshold",
        "collection_goal_zone_coverage_below_threshold",
    }
    assert training_record["status"] == "collection_insufficient"


def test_region_self_supervised_script_exposes_gui_training_options(monkeypatch, tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    _write_task(task_path)
    captured: dict[str, services.RegionSelfSupervisedWorldModelRequest] = {}

    def fake_run(request: services.RegionSelfSupervisedWorldModelRequest) -> dict[str, object]:
        captured["request"] = request
        return {"status": "ok", "training_run_path": "run.json"}

    monkeypatch.setattr(region_self_supervised_script, "run_region_self_supervised_world_model", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_region_self_supervised_world_model.py",
            str(task_path),
            "--collect-rollouts",
            "3",
            "--min-collection-goal-progress-ratio",
            "0.35",
            "--collection-goal-bias-interval",
            "2",
            "--collection-goal-corridor-interval",
            "4",
            "--collection-goal-corridor-lateral-m",
            "3.5",
            "--collection-coverage-grid-size",
            "5",
            "--collection-coverage-target-interval",
            "2",
            "--collection-max-target-steps",
            "35",
            "--collection-strategy",
            "route_aware",
            "--collection-route-target-interval",
            "1",
            "--collection-route-lateral-m",
            "2.5",
            "--collection-multi-start",
            "--collection-multi-start-lateral-m",
            "1.25",
            "--min-route-coverage-ratio",
            "0.55",
            "--min-goal-zone-coverage",
            "0.25",
            "--register-world-model-config",
            "--world-model-config-path",
            str(tmp_path / "world_model_configs.json"),
        ],
    )

    region_self_supervised_script.main()

    request = captured["request"]
    assert request.collect_rollouts == 3
    assert request.min_collection_goal_progress_ratio == 0.35
    assert request.collection_goal_bias_interval == 2
    assert request.collection_goal_corridor_interval == 4
    assert request.collection_goal_corridor_lateral_m == 3.5
    assert request.collection_coverage_grid_size == 5
    assert request.collection_coverage_target_interval == 2
    assert request.collection_max_target_steps == 35
    assert request.collection_strategy == "route_aware"
    assert request.collection_route_target_interval == 1
    assert request.collection_route_lateral_m == 2.5
    assert request.collection_multi_start is True
    assert request.collection_multi_start_lateral_m == 1.25
    assert request.min_route_coverage_ratio == 0.55
    assert request.min_goal_zone_coverage == 0.25
    assert request.register_world_model_config is True
    assert request.world_model_config_path == str(tmp_path / "world_model_configs.json")
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

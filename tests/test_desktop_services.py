from __future__ import annotations

import math
from unittest.mock import Mock, patch

from offroad_sim.datasets import create_mock_orfd_dataset
from offroad_sim.tasks import load_navigation_region_task

from desktop_app import services


def test_desktop_catalog_snapshot_exposes_runtime_choices() -> None:
    catalog = services.catalog_snapshot()

    assert any(item["name"] == "gym_heightmap" for item in catalog["backends"])
    assert any(item["name"] == "world_model" for item in catalog["agents"])
    assert any(item["name"] == "world_model_cem" for item in catalog["planners"])
    assert any(item["name"] == "tiny_learned" for item in catalog["world_models"])
    assert any(item["name"] == "local_lewm_cost" for item in catalog["algorithms"])


def test_desktop_request_builds_dataset_and_planner_options() -> None:
    request = services.RunRequest(
        backend="dataset_replay",
        agent="world_model",
        dataset_root="outputs/mock_orfd_phase3",
        adapter="orfd",
        sequence_id="training/seq_0001",
        load_assets=True,
        world_model_type="tiny_learned",
        world_model_path="outputs/models/phase3_tiny_world_model",
        planner="world_model_cem",
        planner_horizon=4,
        planner_samples=16,
        planner_iterations=2,
    )

    assert services.backend_options(request) == {
        "dataset_root": "outputs/mock_orfd_phase3",
        "sequence_id": "training/seq_0001",
        "adapter": "orfd",
        "load_assets": True,
    }
    assert services.agent_options(request)["planner_config"] == {
        "horizon": 4,
        "num_samples": 16,
        "iterations": 2,
    }
    assert services.scenario_for_request(request)["scenario_id"] == "dataset_mock_orfd_phase3"


def test_desktop_display_value_uses_nan_for_missing_values() -> None:
    assert services.display_value(math.nan) == "NaN"
    assert services.display_value(None) == "NaN"
    assert services.display_value(False) == "false"


def test_desktop_lewm_command_services_require_paths() -> None:
    try:
        services.export_lewm_hdf5("", "out.h5")
    except ValueError as exc:
        assert "Dataset root" in str(exc)
    else:
        raise AssertionError("Expected dataset root validation.")

    try:
        services.train_lewm_cost_model("", "out")
    except ValueError as exc:
        assert "Input HDF5" in str(exc)
    else:
        raise AssertionError("Expected HDF5 path validation.")


def test_run_json_command_ignores_surrounding_logs() -> None:
    completed = Mock(returncode=0, stdout="log before\n{\"ok\": true}\nlog after", stderr="")
    with patch("desktop_app.services.subprocess.run", return_value=completed):
        assert services._run_json_command(["python", "script.py"]) == {"ok": True}


def test_desktop_preview_and_terrain_draft_export(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "orfd", frame_count=3)

    preview = services.preview_dataset_frame(str(dataset_root), adapter="orfd", sequence_id="training/seq_0001")
    assert preview["frame_id"] == "000000"
    assert "front_rgb" in preview["previews"]

    terrain = services.export_orfd_beamng_terrain_draft(
        str(dataset_root),
        adapter="orfd",
        sequence_id="training/seq_0001",
        output_dir=tmp_path / "terrain",
        grid_size=16,
    )
    assert terrain["status"] == "draft_ready"
    assert terrain["beamng_import_ready"] is False
    assert (tmp_path / "terrain" / "heightmap.png").exists()
    assert (tmp_path / "terrain" / "terrain_mesh.obj").exists()


def test_save_manual_navigation_task_writes_valid_task(tmp_path) -> None:
    output = tmp_path / "manual_region.yaml"

    payload = services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            goal_radius=7.5,
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
            evaluation_drive_mode="manual",
        )
    )

    task = load_navigation_region_task(output)
    assert payload["task_path"] == str(output.resolve())
    assert task.task_id == "manual_region_test"
    assert task.contains_point((2.0, -170.0)) is True
    assert task.contains_point((6.0, -210.0)) is True
    assert task.beamng["evaluation_drive_mode"] == "manual"
    assert task.cost["out_of_region_weight"] == 250.0
    assert task.to_beamng_scenario(mode="evaluation")["metadata"]["beamng"]["drive_mode"] == "manual"


def test_save_manual_navigation_task_rejects_start_outside_region(tmp_path) -> None:
    try:
        services.save_manual_navigation_task(
            services.ManualNavigationTaskRequest(
                output_path=str(tmp_path / "bad.yaml"),
                task_id="bad",
                region_polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
                start_pos=(8.0, 1.0, 100.6),
                goal_pos=(1.0, 1.0),
            )
        )
    except ValueError as exc:
        assert "inside the selected region" in str(exc)
    else:
        raise AssertionError("Expected manual task validation failure.")


def test_navigation_task_analysis_checks_region_and_route(tmp_path) -> None:
    output = tmp_path / "manual_region.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
        )
    )

    payload = services.analyze_navigation_task(str(output))

    assert payload["start_in_region"] is True
    assert payload["goal_in_region"] is True
    assert payload["route_in_region"] is True
    assert payload["route_waypoint_count"] == 3
    assert payload["route_length_m"] > payload["straight_line_m"]


def test_preview_navigation_task_in_beamng_uses_manual_preview(tmp_path) -> None:
    output = tmp_path / "manual_region.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
        )
    )
    with patch("desktop_app.services.run_episode") as run_episode:
        run_episode.return_value.to_dict.return_value = {"episode_id": "preview", "metrics": {"drive_mode": "manual"}}

        payload = services.preview_navigation_task_in_beamng(str(output), hold_open_sec=1.0)

    scenario = run_episode.call_args.kwargs["scenario"]
    assert payload["analysis"]["start_in_region"] is True
    assert run_episode.call_args.kwargs["agent_name"] == "stop"
    assert run_episode.call_args.kwargs["close_backend"] is False
    assert scenario["metadata"]["beamng"]["drive_mode"] == "manual"
    assert scenario["metadata"]["beamng"]["preview_mode"] is True
    assert scenario["metadata"]["beamng"]["camera_mode"] == "topdown"
    assert scenario["metadata"]["beamng"]["camera_height_m"] == 90.0
    assert scenario["metadata"]["beamng"]["route"] == [[2.0, -170.0], [5.0, -190.0], [6.0, -210.0]]


def test_realtime_navigation_preview_session_reuses_backend(tmp_path, monkeypatch) -> None:
    output = tmp_path / "manual_region.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
        )
    )
    calls: list[str] = []

    class FakeBackend:
        def __init__(self, *, connection, vehicle_config=None):
            calls.append("init")
            self.metrics = {"level": "gridmap_v2", "route_waypoint_count": 0}

        def reset(self, scenario):
            calls.append("reset")
            self.metrics = {"level": scenario["metadata"]["beamng"]["level"], "route_waypoint_count": len(scenario["metadata"]["beamng"]["route"])}

        def update_navigation_preview(self, scenario):
            calls.append("update")
            self.metrics = {"level": scenario["metadata"]["beamng"]["level"], "route_waypoint_count": len(scenario["metadata"]["beamng"]["route"])}

        def get_metrics(self):
            return dict(self.metrics)

        def close(self):
            calls.append("close")

    monkeypatch.setattr(services, "BeamNGBackend", FakeBackend)
    session = services.BeamNGNavigationPreviewSession()

    first = session.update(str(output), camera_mode="topdown", camera_height_m=100.0)
    second = session.update(str(output), camera_mode="topdown", camera_height_m=120.0)
    session.close()

    assert calls == ["init", "reset", "update", "close"]
    assert first["preview"]["realtime"] is True
    assert second["metrics"]["route_waypoint_count"] == 3


def test_orfd_lewm_pipeline_uses_selected_sequence_and_options() -> None:
    with (
        patch("desktop_app.services.inspect_dataset", return_value={"selected_sequence": "training/seq_0001"}),
        patch("desktop_app.services.export_lewm_hdf5", return_value={"output_hdf5": "out.h5"}) as export,
        patch("desktop_app.services.train_lewm_cost_model", return_value={"output_dir": "model"}) as train,
        patch("desktop_app.services.run_episode_from_request", return_value={"metrics": {}, "episode_path": ""}) as run_episode,
    ):
        payload = services.run_orfd_lewm_pipeline(
            services.PipelineRequest(
                dataset_root="dataset",
                sequence_id="training/seq_0001",
                hdf5_path="custom.h5",
                model_dir="model",
                image_size=32,
                run_beamng=False,
            )
        )

    export.assert_called_once()
    assert export.call_args.kwargs["image_size"] == 32
    train.assert_called_once_with("out.h5", "model")
    assert run_episode.call_count == 1
    assert payload["beamng"] is None

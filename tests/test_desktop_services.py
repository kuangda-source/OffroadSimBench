from __future__ import annotations

import math
from unittest.mock import Mock, patch

from offroad_sim.datasets import create_mock_orfd_dataset

from desktop_app import services


def test_desktop_catalog_snapshot_exposes_runtime_choices() -> None:
    catalog = services.catalog_snapshot()

    assert any(item["name"] == "gym_heightmap" for item in catalog["backends"])
    assert any(item["name"] == "world_model" for item in catalog["agents"])
    assert any(item["name"] == "world_model_cem" for item in catalog["planners"])
    assert any(item["name"] == "tiny_learned" for item in catalog["world_models"])


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

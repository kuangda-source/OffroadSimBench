from __future__ import annotations

import math

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

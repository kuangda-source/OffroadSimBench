from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from desktop_app import services
from desktop_app.qt_main import MainWindow


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_visible_demo_request_keeps_dataset_model_and_backend_switchable() -> None:
    request = services.VisibleBeamNGDemoRequest(
        dataset_root="datasets/ORFD_Dataset_ICRA2022_ZIP",
        adapter="orfd",
        sequence_id="training/c2021_0228_1819",
        world_model_type="le_wm",
        world_model_path="outputs/models/lewm_orfd",
        planner="le_wm_cem",
        scenario="beamng_visible_autodrive",
        vehicle="configs/vehicles/ugv_medium.yaml",
    )

    payload = services.build_visible_beamng_demo_request(request)

    assert payload.agent == "route_world_model"
    assert payload.backend == "beamng"
    assert payload.world_model_type == "le_wm"
    assert payload.world_model_path == "outputs/models/lewm_orfd"


def test_visible_demo_service_runs_route_world_model_episode() -> None:
    request = services.VisibleBeamNGDemoRequest(world_model_type="simple_kinematic", planner="")
    with patch("desktop_app.services.run_episode") as run_episode:
        run_episode.return_value.to_dict.return_value = {"episode_id": "visible", "metrics": {"connected": True}}

        payload = services.run_visible_beamng_demo(request)

    assert payload["episode_id"] == "visible"
    assert run_episode.call_args.kwargs["backend_name"] == "beamng"
    assert run_episode.call_args.kwargs["agent_name"] == "route_world_model"
    assert run_episode.call_args.kwargs["vehicle"] == "configs/vehicles/ugv_medium.yaml"


def test_beamng_page_has_visible_demo_action() -> None:
    _ensure_app()
    window = MainWindow()

    texts = [button.text() for button in window.findChildren(QPushButton)]

    assert "启动 BeamNG 可视自动驾驶" in texts
    window.close()

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
    assert run_episode.call_args.kwargs["backend_options"]["connection"].gfx == "vk"


def test_visible_demo_defaults_are_human_visible() -> None:
    request = services.VisibleBeamNGDemoRequest(world_model_type="simple_kinematic", planner="")
    with patch("desktop_app.services.run_episode") as run_episode:
        run_episode.return_value.to_dict.return_value = {"episode_id": "visible", "metrics": {"connected": True}}

        services.run_visible_beamng_demo(request)

    kwargs = run_episode.call_args.kwargs
    assert kwargs["pre_run_hold_sec"] >= 5.0
    assert kwargs["step_delay_sec"] > 0.0
    assert kwargs["close_backend"] is False


def test_beamng_page_has_visible_demo_action() -> None:
    _ensure_app()
    window = MainWindow()

    texts = [button.text() for button in window.findChildren(QPushButton)]

    assert "BeamNG LE-WM 闭环训练评估" in texts
    assert "启动 BeamNG 可视自动驾驶" in texts
    assert "区域起终点 LE-WM 闭环" in texts
    assert "编辑区域/起终点" in texts
    window.close()


def test_gui_exposes_algorithm_choice() -> None:
    _ensure_app()
    window = MainWindow()

    values = [window.algorithm_combo.itemData(index) for index in range(window.algorithm_combo.count())]

    assert "local_lewm_cost" in values
    window.close()


def test_gui_visible_demo_uses_minimum_human_visible_steps(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    captured: dict[str, services.VisibleBeamNGDemoRequest] = {}

    monkeypatch.setattr(services, "run_visible_beamng_demo", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label: task())

    window.run_visible_beamng_demo()

    assert captured["request"].max_steps >= 600
    window.close()


def test_gui_closed_loop_uses_beamng_map_request(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    captured: dict[str, services.BeamNGMapLeWMClosedLoopRequest] = {}

    monkeypatch.setattr(services, "run_beamng_map_lewm_closed_loop", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label: task())

    window.run_beamng_lewm_closed_loop()

    assert captured["request"].collect_steps >= 160
    assert captured["request"].eval_steps >= 80
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_pipeline_finished_reads_closed_loop_evaluation() -> None:
    _ensure_app()
    window = MainWindow()

    window._pipeline_finished(
        {
            "hdf5_path": "map.h5",
            "model_dir": "model",
            "evaluation": {"metrics": {"steps": 12}, "episode_path": ""},
        }
    )

    assert window.metric_cards["steps"].value_label.text() == "12"
    window.close()


def test_gui_region_navigation_loop_uses_task_path(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    window.task_path_edit.setText("configs/tasks/beamng_region_nav_001.yaml")
    captured: dict[str, services.RegionNavigationClosedLoopRequest] = {}

    monkeypatch.setattr(services, "run_region_navigation_closed_loop", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label: task())

    window.run_region_navigation_loop()

    assert captured["request"].task_path == "configs/tasks/beamng_region_nav_001.yaml"
    assert captured["request"].collect_steps >= 160
    assert captured["request"].close_beamng is False
    window.close()

from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from desktop_app import services
from desktop_app.qt_main import MainWindow, NavigationTaskCanvas, NavigationTaskDialog


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
    assert "编辑/预览区域与起终点" in texts
    assert "编辑区域/起终点" not in texts
    assert "BeamNG 预览区域/起终点" not in texts
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


def test_gui_johnson_valley_demo_uses_agent_control_task(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 9
    captured: dict[str, services.RegionNavigationClosedLoopRequest] = {}

    monkeypatch.setattr(services, "run_region_navigation_closed_loop", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label: task())

    window.run_johnson_valley_demo_loop()

    assert captured["request"].task_path.endswith("beamng_johnson_valley_nav_001.yaml")
    assert captured["request"].planner == "le_wm_cem"
    assert captured["request"].collect_steps >= 240
    assert captured["request"].eval_steps >= 300
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_navigation_preview_uses_editor_callback(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    captured: dict[str, object] = {}

    def fake_preview(task_path: str, **kwargs):
        captured["task_path"] = task_path
        captured.update(kwargs)
        return {"preview": {"realtime": True}, "analysis": {"start_in_region": True}}

    monkeypatch.setattr(window.navigation_preview_session, "update", fake_preview)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label: callback(task()))

    window._preview_task_from_editor("configs/tasks/beamng_johnson_valley_nav_001.yaml", "topdown", 120.0)

    assert captured["task_path"] == "configs/tasks/beamng_johnson_valley_nav_001.yaml"
    assert captured["camera_mode"] == "topdown"
    assert captured["camera_height_m"] == 120.0
    assert "preview" in window.beamng_summary.toPlainText()
    window.close()


def test_gui_navigation_preview_coalesces_requests_while_loading(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    started: list[dict[str, object]] = []

    def fake_run_task(task, callback, label, **kwargs):
        started.append({"task": task, "callback": callback, "label": label, "kwargs": kwargs})

    monkeypatch.setattr(window, "_run_task", fake_run_task)
    monkeypatch.setattr(
        window.navigation_preview_session,
        "update",
        lambda task_path, **kwargs: {"task_path": task_path, "kwargs": kwargs, "analysis": {}},
    )

    window._preview_task_from_editor("configs/tasks/beamng_region_nav_001.yaml", "topdown", 90.0)
    window._preview_task_from_editor("configs/tasks/beamng_region_nav_001.yaml", "orbit", 150.0)

    assert len(started) == 1

    first_payload = started[0]["task"]()
    started[0]["callback"](first_payload)

    assert len(started) == 2
    second_payload = started[1]["task"]()
    assert second_payload["kwargs"]["camera_mode"] == "orbit"
    assert second_payload["kwargs"]["camera_height_m"] == 150.0
    window.close()


def test_navigation_task_dialog_previews_draft_from_same_editor(tmp_path) -> None:
    _ensure_app()
    preview_calls: list[dict[str, object]] = []
    task_path = tmp_path / "draft.yaml"
    dialog = NavigationTaskDialog(
        str(task_path),
        preview_callback=lambda task_path, camera_mode, camera_height_m: preview_calls.append(
            {"task_path": task_path, "camera_mode": camera_mode, "camera_height_m": camera_height_m}
        ),
    )

    dialog.task_id_edit.setText("draft")
    dialog.level_edit.setText("gridmap_v2")
    dialog.canvas.region = [(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)]
    dialog.canvas.start = (2.0, -170.0, 100.6)
    dialog.canvas.goal = (6.0, -210.0)
    dialog.canvas.route = [(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)]

    dialog.preview_task()

    assert dialog.result() == 0
    assert preview_calls == [
        {"task_path": str(task_path.resolve()), "camera_mode": "topdown", "camera_height_m": 150.0}
    ]
    assert task_path.exists()


def test_navigation_task_dialog_realtime_preview_uses_same_draft(tmp_path) -> None:
    _ensure_app()
    preview_calls: list[dict[str, object]] = []
    task_path = tmp_path / "draft.yaml"
    dialog = NavigationTaskDialog(
        str(task_path),
        preview_callback=lambda task_path, camera_mode, camera_height_m: preview_calls.append(
            {"task_path": task_path, "camera_mode": camera_mode, "camera_height_m": camera_height_m}
        ),
    )

    dialog.task_id_edit.setText("draft")
    dialog.level_edit.setText("gridmap_v2")
    dialog.canvas.region = [(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)]
    dialog.canvas.start = (2.0, -170.0, 100.6)
    dialog.canvas.goal = (6.0, -210.0)
    dialog.canvas.route = [(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)]
    assert dialog.realtime_preview_check.isChecked() is True
    dialog.preview_height_spin.setValue(120.0)

    dialog._run_realtime_preview()

    assert preview_calls == [
        {"task_path": str(task_path.resolve()), "camera_mode": "topdown", "camera_height_m": 120.0}
    ]
    assert task_path.exists()


def test_navigation_task_dialog_uses_beamng_pose_for_task_points(tmp_path) -> None:
    _ensure_app()
    task_path = tmp_path / "draft.yaml"
    dialog = NavigationTaskDialog(
        str(task_path),
        pose_callback=lambda: {"available": True, "x": 12.5, "y": -34.25, "z": 101.2, "yaw": 0.75},
    )
    dialog.canvas.clear_region()
    dialog.canvas.clear_route()

    dialog.refresh_beamng_pose()
    dialog._use_beamng_pose_as_region_point()
    dialog._use_beamng_pose_as_start()
    dialog._use_beamng_pose_as_goal()
    dialog._use_beamng_pose_as_route_point()

    assert dialog.current_beamng_pose["available"] is True
    assert "12.500" in dialog.beamng_pose_label.text()
    assert dialog.canvas.beamng_pose == (12.5, -34.25)
    assert dialog.canvas.region == [(12.5, -34.25)]
    assert dialog.canvas.start == (12.5, -34.25, 101.2)
    assert dialog.start_z_spin.value() == 101.2
    assert dialog.start_yaw_spin.value() == 0.75
    assert dialog.canvas.goal == (12.5, -34.25)
    assert dialog.canvas.route == [(12.5, -34.25), (12.5, -34.25), (12.5, -34.25)]


def test_navigation_task_dialog_applies_beamng_window_picks_by_current_mode(tmp_path) -> None:
    _ensure_app()
    task_path = tmp_path / "draft.yaml"
    picks = iter(
        [
            {"available": True, "sequence": 1, "x": 10.0, "y": 20.0, "z": 30.0},
            {"available": True, "sequence": 2, "x": 11.0, "y": 21.0, "z": 31.0, "yaw": 0.5},
            {"available": True, "sequence": 3, "x": 12.0, "y": 22.0, "z": 32.0},
            {"available": True, "sequence": 4, "x": 13.0, "y": 23.0, "z": 33.0},
        ]
    )
    dialog = NavigationTaskDialog(str(task_path), pick_callback=lambda: next(picks))
    dialog.canvas.clear_region()
    dialog.canvas.clear_route()

    dialog.canvas.set_mode("region")
    dialog._poll_beamng_pick()
    dialog.canvas.set_mode("start")
    dialog._poll_beamng_pick()
    dialog.canvas.set_mode("goal")
    dialog._poll_beamng_pick()
    dialog.canvas.set_mode("route")
    dialog._poll_beamng_pick()

    assert dialog.canvas.region == [(10.0, 20.0)]
    assert dialog.canvas.start == (11.0, 21.0, 31.0)
    assert dialog.start_z_spin.value() == 31.0
    assert dialog.start_yaw_spin.value() == 0.5
    assert dialog.canvas.goal == (12.0, 22.0)
    assert dialog.canvas.route == [(11.0, 21.0), (12.0, 22.0), (13.0, 23.0)]
    assert "sequence=4" in dialog.beamng_pick_label.text()


def test_navigation_task_dialog_mode_buttons_toggle_and_highlight(tmp_path) -> None:
    _ensure_app()
    dialog = NavigationTaskDialog(str(tmp_path / "draft.yaml"))

    assert dialog.canvas.mode == "region"
    assert dialog.mode_buttons["region"].isCheckable()
    assert dialog.mode_buttons["region"].isChecked()
    assert not dialog.mode_buttons["start"].isChecked()

    dialog.mode_buttons["region"].click()

    assert dialog.canvas.mode is None
    assert all(not button.isChecked() for button in dialog.mode_buttons.values())

    dialog.mode_buttons["goal"].click()

    assert dialog.canvas.mode == "goal"
    assert dialog.mode_buttons["goal"].isChecked()
    assert not dialog.mode_buttons["region"].isChecked()
    dialog.close()


def test_region_task_editor_opens_non_modal(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    calls = {"exec": 0}

    def fake_exec(self: NavigationTaskDialog) -> QDialog.DialogCode:
        calls["exec"] += 1
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(NavigationTaskDialog, "exec", fake_exec)

    window.open_region_task_editor()

    assert calls["exec"] == 0
    assert window.region_task_dialog is not None
    assert window.region_task_dialog.windowModality() == Qt.WindowModality.NonModal
    window.region_task_dialog.close()
    window.close()


def test_navigation_task_canvas_drag_moves_region_point_without_adding() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF) -> None:
            self._point = point

        def position(self) -> QPointF:
            return self._point

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    canvas.start = (1.0, 1.0, 0.0)
    canvas.goal = (9.0, 9.0)
    canvas.route = [(1.0, 1.0), (9.0, 9.0)]
    canvas._fit_bounds()
    canvas.set_mode("region")

    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((0.0, 0.0))))
    canvas.mouseMoveEvent(FakeMouseEvent(canvas._to_canvas((2.0, 3.0))))
    canvas.mouseReleaseEvent(FakeMouseEvent(canvas._to_canvas((2.0, 3.0))))

    assert len(canvas.region) == 4
    assert canvas.region[0] == (2.0, 3.0)


def test_navigation_task_canvas_right_click_deletes_active_region_point() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
            self._point = point
            self._button = button

        def position(self) -> QPointF:
            return self._point

        def button(self) -> Qt.MouseButton:
            return self._button

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    canvas.start = (1.0, 1.0, 0.0)
    canvas.goal = (9.0, 9.0)
    canvas.route = [(1.0, 1.0), (9.0, 9.0)]
    canvas._fit_bounds()
    canvas.set_mode("region")

    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((10.0, 0.0)), Qt.MouseButton.RightButton))

    assert canvas.region == [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_navigation_task_canvas_right_click_deletes_start_and_goal_points() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
            self._point = point
            self._button = button

        def position(self) -> QPointF:
            return self._point

        def button(self) -> Qt.MouseButton:
            return self._button

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    canvas.start = (1.0, 1.0, 0.0)
    canvas.goal = (9.0, 9.0)
    canvas.route = [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]
    canvas._fit_bounds()

    canvas.set_mode("start")
    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((1.0, 1.0)), Qt.MouseButton.RightButton))
    canvas.set_mode("goal")
    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((9.0, 9.0)), Qt.MouseButton.RightButton))

    assert canvas.start is None
    assert canvas.goal is None
    assert canvas.route == [(5.0, 5.0)]


def test_navigation_task_canvas_ignores_clicks_without_active_mode() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF) -> None:
            self._point = point

        def position(self) -> QPointF:
            return self._point

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = []
    canvas.route = []
    canvas.set_mode(None)

    canvas.mousePressEvent(FakeMouseEvent(QPointF(120.0, 120.0)))

    assert canvas.region == []
    assert canvas.route == []


def test_navigation_task_canvas_preserves_world_aspect_ratio() -> None:
    _ensure_app()
    canvas = NavigationTaskCanvas()
    canvas.resize(640, 360)
    canvas.bounds = (0.0, 100.0, 0.0, 100.0)

    origin = canvas._to_canvas((0.0, 0.0))
    east = canvas._to_canvas((100.0, 0.0))
    north = canvas._to_canvas((0.0, 100.0))
    roundtrip = canvas._from_canvas(canvas._to_canvas((25.0, 75.0)))

    east_distance = abs(east.x() - origin.x())
    north_distance = abs(north.y() - origin.y())
    assert abs(east_distance - north_distance) < 1e-6
    assert roundtrip == (25.0, 75.0)


def test_navigation_task_dialog_delays_invalid_region_warning_until_save(tmp_path, monkeypatch) -> None:
    _ensure_app()
    task_path = tmp_path / "invalid.yaml"
    dialog = NavigationTaskDialog(str(task_path), preview_callback=lambda *_args: None)
    dialog.canvas.region = [(0.0, 0.0), (1.0, 0.0)]
    warnings: list[str] = []
    monkeypatch.setattr("desktop_app.qt_main.QMessageBox.warning", lambda *_args: warnings.append(str(_args[-1])))

    dialog.preview_task()
    assert warnings == []

    dialog.save_task()
    assert warnings

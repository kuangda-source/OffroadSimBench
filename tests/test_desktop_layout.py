from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QWidget

from desktop_app.qt_main import MainWindow, _training_run_overview_text


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _margins_tuple(layout) -> tuple[int, int, int, int]:
    margins = layout.contentsMargins()
    return margins.left(), margins.top(), margins.right(), margins.bottom()


def test_desktop_pages_use_consistent_layout_rhythm() -> None:
    _ensure_app()
    window = MainWindow()

    for index in range(window.page_stack.count()):
        layout = window.page_stack.widget(index).layout()
        assert _margins_tuple(layout) == (0, 0, 0, 0)
        assert layout.spacing() == 16

    for group in window.findChildren(QGroupBox):
        layout = group.layout()
        assert _margins_tuple(layout) == (16, 20, 16, 16)
        assert layout.spacing() == 12

    window.close()


def test_desktop_main_window_fits_available_screen_and_scrolls_content() -> None:
    app = _ensure_app()
    window = MainWindow()
    screen = app.primaryScreen()
    available_height = screen.availableGeometry().height() if screen else 880
    scroll_areas = window.findChildren(QScrollArea)

    assert window.height() <= available_height
    assert any(area.widgetResizable() for area in scroll_areas)

    window.close()


def test_desktop_controls_keep_stable_visual_heights() -> None:
    _ensure_app()
    window = MainWindow()

    for button in window.nav_buttons:
        assert button.minimumHeight() == 40

    for button in window.findChildren(QPushButton):
        expected = 42 if button.objectName() == "primaryButton" else 36
        assert button.minimumHeight() >= expected

    for widget_type in (QLineEdit, QComboBox, QSpinBox):
        for control in window.findChildren(widget_type):
            assert control.minimumHeight() >= 36

    for card in window.metric_cards.values():
        assert card.minimumHeight() == 82

    window.close()


def test_dataset_training_page_exposes_external_trainer_controls() -> None:
    _ensure_app()
    window = MainWindow()

    assert window.trainer_entrypoint_edit.placeholderText()
    assert "{dataset_root}" in window.trainer_arguments_edit.toPlainText()
    assert "{output_dir}" in window.trainer_arguments_edit.toPlainText()
    assert window.save_trainer_button.text()

    window.close()


def test_dataset_training_tabs_have_visible_section_titles() -> None:
    _ensure_app()
    window = MainWindow()

    titles = {
        label.text()
        for label in window.page_stack.widget(1).findChildren(QLabel)
        if label.objectName() == "tabTitle"
    }

    assert "Dataset import" in titles
    assert "Model training" in titles
    assert "Training results" in titles
    assert "Processing and labels" in titles

    window.close()


def test_dataset_training_page_separates_dataset_trainer_config_and_results() -> None:
    _ensure_app()
    window = MainWindow()

    group_titles = {group.title() for group in window.page_stack.widget(1).findChildren(QGroupBox)}

    assert "Data source" in group_titles
    assert "Trainer / algorithm" in group_titles
    assert "Training config" in group_titles
    assert "Latest training result" in group_titles
    assert "Trained model registry" in group_titles

    window.close()


def test_beamng_run_page_uses_compact_config_and_action_toolbar() -> None:
    _ensure_app()
    window = MainWindow()
    beamng_page = window.page_stack.widget(2)
    group_titles = {group.title() for group in beamng_page.findChildren(QGroupBox)}
    action_toolbars = [widget for widget in beamng_page.findChildren(QWidget) if widget.objectName() == "beamngActionToolbar"]
    compact_fields = [widget for widget in beamng_page.findChildren(QWidget) if widget.objectName() == "compactField"]

    assert "Run configuration" in group_titles
    assert "Actions" in group_titles
    assert "BeamNG task and model" not in group_titles
    assert action_toolbars
    assert len(compact_fields) >= 3
    for field in compact_fields:
        assert field.maximumHeight() <= 64

    window.close()


def test_training_run_overview_surfaces_region_navigation_diagnostics() -> None:
    text = _training_run_overview_text(
        {
            "run_id": "demo",
            "preset_id": "region_self_supervised_world_model",
            "status": "completed",
            "artifact_path": "outputs/model",
            "summary": {
                "diagnostics": {
                    "status": "navigation_model_insufficient",
                    "message": "Model-controlled evaluation did not reach the goal.",
                    "next_actions": ["Collect wider coverage inside the region."],
                }
            },
            "metrics": {"goal_success": False, "min_goal_distance": 42.0},
        }
    )

    assert "diagnostic_status: navigation_model_insufficient" in text
    assert "diagnostic_next: Collect wider coverage inside the region." in text

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QWidget

from desktop_app.qt_main import MainWindow, _beamng_quality_report_text, _region_world_model_summary_text, _training_run_overview_text


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


def test_beamng_run_page_uses_compact_config_and_workflow_toolbars() -> None:
    _ensure_app()
    window = MainWindow()
    beamng_page = window.page_stack.widget(2)
    group_titles = {group.title() for group in beamng_page.findChildren(QGroupBox)}
    workflow_toolbars = [
        widget
        for widget in beamng_page.findChildren(QWidget)
        if widget.objectName().startswith("beamngWorkflow") and widget.objectName().endswith("Toolbar")
    ]
    compact_fields = [widget for widget in beamng_page.findChildren(QWidget) if widget.objectName() == "compactField"]

    assert "Run configuration" in group_titles
    assert "BeamNG task and model" not in group_titles
    assert len(workflow_toolbars) == 3
    assert len(compact_fields) >= 3
    for field in compact_fields:
        assert field.maximumHeight() <= 64

    window.close()


def test_beamng_page_separates_training_workflow_and_quality_report() -> None:
    _ensure_app()
    window = MainWindow()
    beamng_page = window.page_stack.widget(2)
    workflow_groups = {
        group.objectName()
        for group in beamng_page.findChildren(QGroupBox)
        if group.objectName().startswith("beamngWorkflow")
    }
    workflow_toolbars = {
        widget.objectName()
        for widget in beamng_page.findChildren(QWidget)
        if widget.objectName().startswith("beamngWorkflow") and widget.objectName().endswith("Toolbar")
    }

    assert workflow_groups == {
        "beamngWorkflowCollect",
        "beamngWorkflowTrain",
        "beamngWorkflowEvaluate",
    }
    assert workflow_toolbars == {
        "beamngWorkflowCollectToolbar",
        "beamngWorkflowTrainToolbar",
        "beamngWorkflowEvaluateToolbar",
    }
    assert any(group.objectName() == "beamngQualityReport" for group in beamng_page.findChildren(QGroupBox))
    assert hasattr(window, "beamng_quality_report")
    assert window.beamng_quality_report.placeholderText() == "Training quality report: NaN"

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


def test_training_run_overview_surfaces_validation_quality_metrics() -> None:
    text = _training_run_overview_text(
        {
            "run_id": "demo",
            "preset_id": "region_world_model_training",
            "status": "completed",
            "artifact_path": "outputs/model",
            "metrics": {
                "train_rmse": 0.42,
                "validation_rmse": 0.73,
                "validation_sample_count": 3,
                "segment_rmse": {"start": 0.1, "middle": 0.2, "goal": None},
            },
        }
    )

    assert "validation_rmse: 0.73" in text
    assert "validation_sample_count: 3" in text
    assert "segment_rmse.start: 0.1" in text
    assert "segment_rmse.middle: 0.2" in text


def test_region_world_model_summary_surfaces_baseline_comparison() -> None:
    text = _region_world_model_summary_text(
        {
            "acceptance": {
                "goal_success": False,
                "min_goal_distance": 52.0,
                "final_goal_distance": 80.0,
                "collision_count": 0,
                "reverse_count": 2,
                "stuck_recovery_count": 3,
            },
            "region_navigation": {"evaluation_agent": "world_model_direct", "route_free": True},
            "comparison": {
                "route_free_goal_success": False,
                "route_free_min_goal_distance": 52.0,
                "route_free_reverse_count": 2,
                "route_guided_goal_success": True,
                "route_guided_final_goal_distance": 3.0,
            },
            "trajectory_plot_path": "outputs/eval/region_world_model_trajectory.svg",
        }
    )

    assert "reverse_count: 2" in text
    assert "stuck_recovery_count: 3" in text
    assert "route_free_min_goal_distance: 52" in text
    assert "route_guided_goal_success: true" in text
    assert "trajectory_plot_path: outputs/eval/region_world_model_trajectory.svg" in text


def test_beamng_quality_report_surfaces_training_and_evaluation_evidence() -> None:
    text = _beamng_quality_report_text(
        {
            "preset_label": "Evaluate BeamNG model control",
            "status": "completed",
            "quality_gate": {
                "passed": True,
                "route_coverage_ratio": 0.8,
                "goal_zone_coverage": 0.5,
                "collection_min_goal_distance": 11.0,
                "unique_region_cells": 5,
            },
            "metrics": {
                "train_rmse": 0.12,
                "validation_rmse": 0.2,
                "segment_rmse": {"start": 0.1, "middle": 0.2, "goal": 0.3},
            },
            "comparison": {
                "route_free_goal_success": True,
                "route_free_min_goal_distance": 9.0,
                "route_guided_goal_success": True,
                "route_guided_final_goal_distance": 8.0,
            },
            "collection_manifest_path": "outputs/collection/region_training_collection.json",
            "model_dir": "outputs/model",
            "trajectory_plot_path": "outputs/eval/region_world_model_trajectory.svg",
        }
    )

    assert "quality_gate.passed: true" in text
    assert "quality_gate.route_coverage_ratio: 0.8" in text
    assert "train_rmse: 0.12" in text
    assert "validation_rmse: 0.2" in text
    assert "segment_rmse.goal: 0.3" in text
    assert "route_free_goal_success: true" in text
    assert "route_guided_final_goal_distance: 8" in text
    assert "collection_manifest_path: outputs/collection/region_training_collection.json" in text
    assert "trajectory_plot_path: outputs/eval/region_world_model_trajectory.svg" in text

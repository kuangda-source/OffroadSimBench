from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QLineEdit, QPushButton, QSpinBox

from desktop_app.qt_main import MainWindow


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

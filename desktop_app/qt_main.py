"""PySide6 desktop application for OffroadSimBench."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QObject, QRectF, QSize, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop_app import services


PAGE_SPACING = 16
CARD_SPACING = 12
FIELD_SPACING = 6
CARD_MARGINS = (16, 20, 16, 16)
CONTROL_HEIGHT = 36
BUTTON_HEIGHT = 36
PRIMARY_BUTTON_HEIGHT = 42
NAV_BUTTON_HEIGHT = 40
METRIC_CARD_HEIGHT = 82
PREVIEW_MIN_HEIGHT = 280
INITIAL_WINDOW_WIDTH = 1600
INITIAL_WINDOW_HEIGHT = 840
DATASET_SOURCE_MIN_WIDTH = 520


@dataclass
class GuiSettings:
    max_steps: int = 1000
    seed: int = 7
    planner_horizon: int = 6
    planner_samples: int = 32
    planner_iterations: int = 3
    image_size: int = 64
    preview_frame_index: int = 0
    terrain_frame_index: int = 0
    terrain_grid_size: int = 64
    terrain_size_m: int = 40
    record: bool = True
    record_arrays: bool = False
    load_assets: bool = False


class TaskWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    settled = Signal()

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        cancel_hook: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.cancel_hook = cancel_hook
        self._canceled = threading.Event()

    def cancel(self) -> None:
        if self._canceled.is_set():
            return
        self._canceled.set()
        if self.cancel_hook is not None:
            try:
                self.cancel_hook()
            except Exception:
                pass

    def is_canceled(self) -> bool:
        return self._canceled.is_set()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            if not self._canceled.is_set():
                self.finished.emit(result)
        except Exception as exc:
            if not self._canceled.is_set():
                self.failed.emit(str(exc))
        finally:
            self.settled.emit()


class StablePreviewLabel(QLabel):
    """A preview surface whose source image never changes layout geometry."""

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._placeholder = placeholder
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(PREVIEW_MIN_HEIGHT)
        self.setObjectName("previewPane")
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(360, PREVIEW_MIN_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(160, PREVIEW_MIN_HEIGHT)

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = QPixmap(pixmap)
        self.setText("")
        self._render_source()

    def clear_preview(self, placeholder: str | None = None) -> None:
        self._source_pixmap = QPixmap()
        super().setPixmap(QPixmap())
        self.setText(placeholder or self._placeholder)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render_source()

    def _render_source(self) -> None:
        if self._source_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        super().setPixmap(
            self._source_pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class AdvancedSettingsDialog(QDialog):
    def __init__(self, settings: GuiSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("高级参数")
        self.setMinimumWidth(420)
        self.controls: dict[str, QSpinBox | QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        for name, label, minimum, maximum in [
            ("max_steps", "最大步数", 1, 100000),
            ("seed", "随机种子", 0, 999999),
            ("planner_horizon", "规划 horizon", 1, 200),
            ("planner_samples", "规划 samples", 4, 5000),
            ("planner_iterations", "规划 iterations", 1, 100),
            ("image_size", "HDF5 图像尺寸", 16, 512),
            ("preview_frame_index", "预览帧索引", 0, 1000000),
            ("terrain_frame_index", "地形帧索引", 0, 1000000),
            ("terrain_grid_size", "地形网格", 16, 256),
            ("terrain_size_m", "地形尺寸 m", 5, 500),
        ]:
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(int(getattr(settings, name)))
            spin.setMinimumHeight(CONTROL_HEIGHT)
            self.controls[name] = spin
            form.addRow(label, spin)

        for name, label in [
            ("record", "记录 episode"),
            ("record_arrays", "记录数组"),
            ("load_assets", "回放时加载数据资产"),
        ]:
            checkbox = QCheckBox()
            checkbox.setChecked(bool(getattr(settings, name)))
            self.controls[name] = checkbox
            form.addRow(label, checkbox)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> GuiSettings:
        data: dict[str, Any] = {}
        for key, control in self.controls.items():
            if isinstance(control, QCheckBox):
                data[key] = control.isChecked()
            else:
                data[key] = control.value()
        return GuiSettings(**data)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = services.NAN_TEXT) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setMinimumHeight(METRIC_CARD_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: Any) -> None:
        self.value_label.setText(services.display_value(value))


class TrainingCurveWidget(QWidget):
    SERIES_COLORS = ("#007aff", "#ff9f0a", "#34c759", "#af52de", "#ff375f", "#5ac8fa")

    def __init__(self) -> None:
        super().__init__()
        self.history: dict[str, list[float]] = {}
        self.steps: dict[str, list[float]] = {}
        self.primary_metric = ""
        self.selected_metrics: list[str] = []
        self.diagnostics: dict[str, Any] = {}
        self.x_zoom = 1.0
        self.hover_metric = ""
        self.hover_index: int | None = None
        self._projected_points: dict[str, list[tuple[int, float, float, float, float]]] = {}
        self.setMouseTracking(True)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_history(
        self,
        history: dict[str, list[float]],
        steps: dict[str, list[float]] | None = None,
    ) -> None:
        normalized: dict[str, list[float]] = {}
        normalized_steps: dict[str, list[float]] = {}
        for key, values in history.items():
            finite_values: list[float] = []
            finite_steps: list[float] = []
            raw_steps = steps.get(str(key), []) if isinstance(steps, dict) else []
            for index, value in enumerate(values):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    finite_values.append(number)
                    try:
                        step = float(raw_steps[index]) if index < len(raw_steps) else float(index)
                    except (TypeError, ValueError):
                        step = float(index)
                    finite_steps.append(step if math.isfinite(step) else float(index))
            if finite_values:
                normalized[str(key)] = finite_values
                normalized_steps[str(key)] = finite_steps
        previous = self.primary_metric
        previous_lengths = {key: len(values) for key, values in self.history.items()}
        self.history = normalized
        self.steps = normalized_steps
        priority = [
            "loss",
            "train_loss",
            "val_loss",
            "validation_loss",
            "final_loss",
            "validation_rmse",
            "validation_mse",
            "train_rmse",
            "train_mse",
            "segment_rmse.goal",
            "segment_rmse.middle",
            "segment_rmse.start",
            "metadata.mean_goal_distance",
            "total_frames",
        ]
        primary = previous if previous in self.history else next(
            (key for key in priority if key in self.history), next(iter(self.history), "")
        )
        self.set_primary_metric(primary)
        if set(previous_lengths) != set(normalized) or any(
            len(normalized[key]) < previous_lengths.get(key, 0) for key in normalized
        ):
            self.x_zoom = 1.0
        self.hover_metric = ""
        self.hover_index = None
        self.update()

    def set_primary_metric(self, metric: str) -> None:
        self.primary_metric = metric if metric in self.history else next(iter(self.history), "")
        self.selected_metrics = self._related_metrics(self.primary_metric)
        self.update()

    def set_selected_metrics(self, metrics: list[str]) -> None:
        selected = [metric for metric in metrics if metric in self.history]
        self.selected_metrics = selected[:6]
        if self.selected_metrics:
            self.primary_metric = self.selected_metrics[0]
        self.update()

    def set_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        self.update()

    def metric_names(self) -> list[str]:
        return sorted(self.history)

    def reset_zoom(self) -> None:
        self.x_zoom = 1.0
        self.update()

    def _related_metrics(self, primary: str) -> list[str]:
        if not primary:
            return []
        lowered = primary.lower()
        if "loss" in lowered:
            candidates = [
                name
                for name in self.history
                if "loss" in name.lower() and not name.lower().startswith("resource.")
            ]
            ordered = [primary]
            ordered.extend(name for name in candidates if name != primary and name.lower().startswith("train"))
            ordered.extend(
                name
                for name in candidates
                if name not in ordered and (name.lower().startswith("val") or "validation" in name.lower())
            )
            ordered.extend(name for name in candidates if name not in ordered)
            return ordered[:4]
        return [primary]

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        rect = self.rect().adjusted(18, 34, -18, -28)
        painter.setPen(QPen(QColor("#d2d2d7"), 1))
        painter.drawRect(rect)
        for index in range(1, 4):
            y = rect.top() + rect.height() * index / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        visible_metrics = [name for name in self.selected_metrics if name in self.history]
        if not visible_metrics:
            painter.setPen(QPen(QColor("#6e6e73"), 1))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Training metrics: NaN")
            return

        max_points = max(len(self.history[name]) for name in visible_metrics)
        all_steps = [step for name in visible_metrics for step in self.steps.get(name, [])]
        minimum_step = min(all_steps)
        maximum_step = max(all_steps)
        full_span = max(maximum_step - minimum_step, 1.0)
        visible_start_step = maximum_step - full_span / self.x_zoom
        visible_values = [
            value
            for name in visible_metrics
            for step, value in zip(self.steps.get(name, []), self.history[name], strict=False)
            if step >= visible_start_step
        ]
        if not visible_values:
            visible_values = [value for name in visible_metrics for value in self.history[name]]
        low = min(visible_values)
        high = max(visible_values)
        if math.isclose(low, high):
            low -= 1.0
            high += 1.0

        def project(step: float, value: float) -> tuple[float, float]:
            x = rect.left() + ((step - visible_start_step) / max(maximum_step - visible_start_step, 1e-12)) * rect.width()
            y = rect.bottom() - ((value - low) / (high - low)) * rect.height()
            return x, y

        self._projected_points = {}
        legend_slot_width = rect.width() / max(len(visible_metrics), 1)
        for series_index, metric in enumerate(visible_metrics):
            color = QColor(self.SERIES_COLORS[series_index % len(self.SERIES_COLORS)])
            values = self.history[metric]
            metric_steps = self.steps.get(metric, list(range(len(values))))
            projected = [
                (index, step, value, *project(step, value))
                for index, (step, value) in enumerate(zip(metric_steps, values, strict=False))
                if step >= visible_start_step
            ]
            self._projected_points[metric] = projected
            painter.setPen(QPen(color, 2))
            for start, end in zip(projected, projected[1:], strict=False):
                painter.drawLine(int(start[3]), int(start[4]), int(end[3]), int(end[4]))
            if len(projected) == 1:
                _, _, _, x, y = projected[0]
                painter.setBrush(color)
                painter.drawEllipse(QRectF(x - 4, y - 4, 8, 8))
            legend_x = rect.left() + series_index * legend_slot_width
            painter.setPen(QPen(color, 2))
            painter.drawLine(legend_x, 16, legend_x + 14, 16)
            painter.setPen(QPen(QColor("#1d1d1f"), 1))
            label = f"{metric}: {services.display_value(values[-1])}"
            label = painter.fontMetrics().elidedText(
                label,
                Qt.TextElideMode.ElideRight,
                max(20, int(legend_slot_width - 24)),
            )
            painter.drawText(legend_x + 18, 20, label)

        warning_rows = self.diagnostics.get("warnings") if isinstance(self.diagnostics.get("warnings"), list) else []
        painter.setPen(QPen(QColor("#ff3b30"), 2))
        painter.setBrush(QColor("#ff3b30"))
        for warning in warning_rows:
            if not isinstance(warning, dict):
                continue
            metric = str(warning.get("metric") or "")
            point_index = warning.get("point_index")
            if not isinstance(point_index, int):
                continue
            point = next(
                (row for row in self._projected_points.get(metric, []) if row[0] == point_index),
                None,
            )
            if point is not None:
                painter.drawEllipse(QRectF(point[3] - 4, point[4] - 4, 8, 8))

        painter.setPen(QPen(QColor("#6e6e73"), 1))
        painter.drawText(
            rect.left(),
            self.height() - 8,
            f"points={max_points}  step={visible_start_step:.4g}-{maximum_step:.4g}  zoom={self.x_zoom:.1f}x",
        )
        if self.hover_metric and self.hover_index is not None:
            point = next(
                (row for row in self._projected_points.get(self.hover_metric, []) if row[0] == self.hover_index),
                None,
            )
            if point is not None:
                painter.setPen(QPen(QColor("#1d1d1f"), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(point[3]), rect.top(), int(point[3]), rect.bottom())

    def mouseMoveEvent(self, event: Any) -> None:
        position = event.position()
        nearest: tuple[float, str, int, float, float] | None = None
        for metric, points in self._projected_points.items():
            for index, step, value, x, y in points:
                distance = (position.x() - x) ** 2 + (position.y() - y) ** 2
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, metric, index, step, value)
        if nearest is not None and nearest[0] <= 18.0**2:
            _, metric, index, step, value = nearest
            self.hover_metric = metric
            self.hover_index = index
            self.setToolTip(f"{metric}\nstep: {step:.8g}\npoint: {index + 1}\nvalue: {value:.8g}")
        else:
            self.hover_metric = ""
            self.hover_index = None
            self.setToolTip("")
        self.update()

    def leaveEvent(self, event: Any) -> None:
        del event
        self.hover_metric = ""
        self.hover_index = None
        self.update()

    def wheelEvent(self, event: Any) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.x_zoom = max(1.0, min(20.0, self.x_zoom * factor))
        event.accept()
        self.update()


class TrajectoryCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.trace: list[dict[str, Any]] = []
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_trace(self, trace: list[dict[str, Any]]) -> None:
        self.trace = trace
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(14, 14, -14, -14)
        painter.fillRect(self.rect(), QColor("#101820"))
        painter.setPen(QPen(QColor("#263644"), 1))
        for index in range(1, 6):
            x = rect.left() + rect.width() * index / 6
            y = rect.top() + rect.height() * index / 6
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        points = [(row.get("x"), row.get("y")) for row in self.trace]
        points = [(float(x), float(y)) for x, y in points if _is_finite(x) and _is_finite(y)]
        if len(points) < 2:
            painter.setPen(QPen(QColor("#7e8b96"), 1))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "NaN / 暂无轨迹")
            return

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        margin = 1.0
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        if math.isclose(min_x, max_x):
            max_x += 1.0
        if math.isclose(min_y, max_y):
            max_y += 1.0

        def project(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            px = rect.left() + (x - min_x) / (max_x - min_x) * rect.width()
            py = rect.bottom() - (y - min_y) / (max_y - min_y) * rect.height()
            return px, py

        projected = [project(point) for point in points]
        painter.setPen(QPen(QColor("#43d9ad"), 3))
        for start, end in zip(projected, projected[1:], strict=False):
            painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

        start = projected[0]
        end = projected[-1]
        painter.setBrush(QColor("#f6c85f"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(start[0] - 5, start[1] - 5, 10, 10))
        painter.setBrush(QColor("#ff6b6b"))
        painter.drawEllipse(QRectF(end[0] - 6, end[1] - 6, 12, 12))

        goal = _first_goal(self.trace)
        if goal is not None:
            gx, gy = project(goal)
            painter.setPen(QPen(QColor("#8bd3ff"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(gx - 7, gy - 7, 14, 14))


class NavigationTaskCanvas(QWidget):
    changed = Signal()
    mode_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.mode: str | None = "region"
        self.region: list[tuple[float, float]] = [(0.0, -160.0), (30.0, -160.0), (30.0, -230.0), (0.0, -230.0)]
        self.start: tuple[float, float, float] | None = (1.0, -170.0, 100.6)
        self.goal: tuple[float, float] | None = (6.0, -215.0)
        self.route: list[tuple[float, float]] = [(1.0, -170.0), (6.0, -215.0)]
        self.beamng_pose: tuple[float, float] | None = None
        self._drag_target: tuple[str, int] | None = None
        self._drag_pick_radius_px = 12.0
        self.bounds = (-8.0, 36.0, -240.0, -150.0)
        self.setMinimumHeight(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_mode(self, mode: str | None) -> None:
        if mode not in {None, "region", "start", "goal", "route"}:
            raise ValueError(f"Unsupported navigation edit mode: {mode}")
        if self.mode == mode:
            return
        self.mode = mode
        self.update()
        self.mode_changed.emit(mode)

    def clear_region(self) -> None:
        self.region = []
        self.update()
        self.changed.emit()

    def clear_route(self) -> None:
        self.route = []
        self.update()
        self.changed.emit()

    def add_region_point(self, point: tuple[float, float]) -> None:
        self.region.append((float(point[0]), float(point[1])))
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def set_start_pose(self, point: tuple[float, float], z: float | None = None) -> None:
        current_z = self.start[2] if self.start is not None else 100.6
        self.start = (float(point[0]), float(point[1]), current_z if z is None else float(z))
        if not self.route:
            self.route = [(self.start[0], self.start[1])]
        else:
            self.route[0] = (self.start[0], self.start[1])
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def set_goal_point(self, point: tuple[float, float]) -> None:
        self.goal = (float(point[0]), float(point[1]))
        if len(self.route) < 2:
            self.route.append(self.goal)
        else:
            self.route[-1] = self.goal
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def add_route_point(self, point: tuple[float, float]) -> None:
        self.route.append((float(point[0]), float(point[1])))
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def set_beamng_pose_marker(self, point: tuple[float, float] | None) -> None:
        self.beamng_pose = None if point is None else (float(point[0]), float(point[1]))
        self._fit_bounds()
        self.update()

    def load_task(self, task: Any) -> None:
        self.region = [tuple(point) for point in task.region_polygon]
        self.start = tuple(task.start_pos)
        self.goal = tuple(task.goal_pos)
        self.route = [tuple(point) for point in task.expert_route]
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d151d"))
        plot = self._plot_rect()
        painter.setPen(QPen(QColor("#263847"), 1))
        painter.drawRect(plot)
        for index in range(1, 6):
            x = plot.left() + plot.width() * index / 6
            y = plot.top() + plot.height() * index / 6
            painter.drawLine(int(x), int(plot.top()), int(x), int(plot.bottom()))
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))

        if len(self.region) >= 2:
            projected = [self._to_canvas(point) for point in self.region]
            painter.setPen(QPen(QColor("#7fb7ff"), 2))
            for start, end in zip(projected, projected[1:], strict=False):
                painter.drawLine(start, end)
            if len(projected) >= 3:
                painter.drawLine(projected[-1], projected[0])
        for point in self.region:
            self._draw_point(painter, point, QColor("#7fb7ff"), 5)

        if len(self.route) >= 2:
            painter.setPen(QPen(QColor("#43d9ad"), 3))
            projected_route = [self._to_canvas(point) for point in self.route]
            for start, end in zip(projected_route, projected_route[1:], strict=False):
                painter.drawLine(start, end)
        for point in self.route:
            self._draw_point(painter, point, QColor("#43d9ad"), 4)

        if self.start is not None:
            self._draw_point(painter, (self.start[0], self.start[1]), QColor("#f6c85f"), 7)
        if self.goal is not None:
            self._draw_point(painter, self.goal, QColor("#ff6b6b"), 8)
        if self.beamng_pose is not None:
            projected_pose = self._to_canvas(self.beamng_pose)
            painter.setPen(QPen(QColor("#d77bff"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(projected_pose.x() - 9, projected_pose.y() - 9, 18, 18))
        painter.setPen(QPen(QColor("#8da1af"), 1))
        painter.setFont(QFont("Segoe UI", 10))
        mode_label = self.mode or "none"
        mode_hint = "click to set points" if self.mode else "select a mode button"
        painter.drawText(18, 22, f"mode: {mode_label} | {mode_hint}")

    def mousePressEvent(self, event: Any) -> None:
        canvas_point = event.position() if hasattr(event, "position") else QPointF(event.x(), event.y())
        if self.mode is None:
            self._drag_target = None
            return
        if hasattr(event, "button") and event.button() == Qt.MouseButton.RightButton:
            self._delete_nearest_target(canvas_point)
            return
        point = self._from_canvas(canvas_point)
        self._drag_target = self._nearest_drag_target(canvas_point)
        if self._drag_target is not None:
            return
        if self.mode == "region":
            self.region.append(point)
        elif self.mode == "start":
            current_z = self.start[2] if self.start is not None else 100.6
            self.start = (point[0], point[1], current_z)
            if not self.route:
                self.route = [(point[0], point[1])]
            else:
                self.route[0] = (point[0], point[1])
        elif self.mode == "goal":
            self.goal = point
            if len(self.route) < 2:
                self.route.append(point)
            else:
                self.route[-1] = point
        elif self.mode == "route":
            self.route.append(point)
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_target is None:
            return
        point = self._from_canvas(event.position() if hasattr(event, "position") else QPointF(event.x(), event.y()))
        self._move_drag_target(point)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drag_target is not None:
            point = self._from_canvas(event.position() if hasattr(event, "position") else QPointF(event.x(), event.y()))
            self._move_drag_target(point)
        self._drag_target = None

    def _plot_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(32.0, 28.0, -20.0, -28.0)

    def _to_canvas(self, point: tuple[float, float]) -> QPointF:
        plot = self._plot_rect()
        min_x, max_x, min_y, max_y = self._view_bounds(plot)
        x = plot.left() + (point[0] - min_x) / max(max_x - min_x, 1e-6) * plot.width()
        y = plot.bottom() - (point[1] - min_y) / max(max_y - min_y, 1e-6) * plot.height()
        return QPointF(x, y)

    def _from_canvas(self, point: QPointF) -> tuple[float, float]:
        plot = self._plot_rect()
        min_x, max_x, min_y, max_y = self._view_bounds(plot)
        x = min_x + (point.x() - plot.left()) / max(plot.width(), 1e-6) * (max_x - min_x)
        y = min_y + (plot.bottom() - point.y()) / max(plot.height(), 1e-6) * (max_y - min_y)
        return (round(x, 3), round(y, 3))

    def _view_bounds(self, plot: QRectF | None = None) -> tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = self.bounds
        world_width = max(max_x - min_x, 1e-6)
        world_height = max(max_y - min_y, 1e-6)
        plot = plot or self._plot_rect()
        plot_width = max(plot.width(), 1e-6)
        plot_height = max(plot.height(), 1e-6)
        plot_aspect = plot_width / plot_height
        world_aspect = world_width / world_height
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        if world_aspect > plot_aspect:
            view_width = world_width
            view_height = world_width / plot_aspect
        else:
            view_height = world_height
            view_width = world_height * plot_aspect
        return (
            center_x - view_width / 2.0,
            center_x + view_width / 2.0,
            center_y - view_height / 2.0,
            center_y + view_height / 2.0,
        )

    def _draw_point(self, painter: QPainter, point: tuple[float, float], color: QColor, radius: int) -> None:
        projected = self._to_canvas(point)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(projected.x() - radius, projected.y() - radius, radius * 2, radius * 2))

    def _nearest_drag_target(self, canvas_point: QPointF) -> tuple[str, int] | None:
        handles: list[tuple[str, int, tuple[float, float]]] = []
        if self.mode == "region":
            handles.extend(("region", index, point) for index, point in enumerate(self.region))
        elif self.mode == "route":
            handles.extend(("route", index, point) for index, point in enumerate(self.route))
        elif self.mode == "start" and self.start is not None:
            handles.append(("start", 0, (self.start[0], self.start[1])))
        elif self.mode == "goal" and self.goal is not None:
            handles.append(("goal", 0, self.goal))
        best: tuple[str, int] | None = None
        best_distance = self._drag_pick_radius_px
        for kind, index, point in handles:
            projected = self._to_canvas(point)
            distance = math.hypot(projected.x() - canvas_point.x(), projected.y() - canvas_point.y())
            if distance <= best_distance:
                best = (kind, index)
                best_distance = distance
        return best

    def _move_drag_target(self, point: tuple[float, float]) -> None:
        if self._drag_target is None:
            return
        kind, index = self._drag_target
        if kind == "region" and 0 <= index < len(self.region):
            self.region[index] = point
        elif kind == "route" and 0 <= index < len(self.route):
            self.route[index] = point
        elif kind == "start":
            current_z = self.start[2] if self.start is not None else 100.6
            self.start = (point[0], point[1], current_z)
            if self.route:
                self.route[0] = point
        elif kind == "goal":
            self.goal = point
            if self.route:
                self.route[-1] = point
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def _delete_nearest_target(self, canvas_point: QPointF) -> None:
        target = self._nearest_drag_target(canvas_point)
        if target is None:
            return
        kind, index = target
        if kind == "region" and 0 <= index < len(self.region):
            self.region.pop(index)
        elif kind == "route" and 0 <= index < len(self.route):
            self.route.pop(index)
        elif kind == "start":
            self.start = None
            if self.route:
                self.route.pop(0)
        elif kind == "goal":
            self.goal = None
            if self.route:
                self.route.pop(-1)
        self._drag_target = None
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def _fit_bounds(self) -> None:
        points = [*self.region, *self.route]
        if self.start is not None:
            points.append((self.start[0], self.start[1]))
        if self.goal is not None:
            points.append(self.goal)
        if self.beamng_pose is not None:
            points.append(self.beamng_pose)
        if not points:
            return
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        margin = 8.0
        self.bounds = (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


class NavigationTaskDialog(QDialog):
    def __init__(
        self,
        task_path: str,
        parent: QWidget | None = None,
        preview_callback: Callable[[str, str, float], None] | None = None,
        pose_callback: Callable[[], dict[str, Any]] | None = None,
        pick_callback: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑/预览区域与起终点")
        self.resize(760, 680)
        self.saved_task_path = ""
        self.preview_callback = preview_callback
        self.pose_callback = pose_callback
        self.pick_callback = pick_callback
        self.current_beamng_pose: dict[str, Any] = {"available": False}
        self.last_beamng_pick_sequence: Any = None
        self.canvas = NavigationTaskCanvas()
        self.mode_buttons: dict[str, QPushButton] = {}
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(650)
        self.preview_timer.timeout.connect(self._run_realtime_preview)
        self.picker_timer = QTimer(self)
        self.picker_timer.setInterval(50)
        self.picker_timer.timeout.connect(self._poll_beamng_pick)

        self.output_path_edit = QLineEdit(task_path or r"configs\tasks\manual_region_nav.yaml")
        self.task_id_edit = QLineEdit("manual_region_nav")
        self.level_edit = QLineEdit("gridmap_v2")
        self.start_z_spin = self._double_spin(100.6, 0.0, 1000.0)
        self.start_yaw_spin = self._double_spin(-1.57, -6.29, 6.29)
        self.goal_radius_spin = self._double_spin(8.0, 0.5, 100.0)
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 100000)
        self.max_steps_spin.setValue(300)
        self.evaluation_drive_combo = QComboBox()
        self.evaluation_drive_combo.addItem("agent/model control", "manual")
        self.evaluation_drive_combo.addItem("BeamNG ai_line", "ai_line")
        self.preview_camera_combo = QComboBox()
        self.preview_camera_combo.addItem("俯视高视角", "topdown")
        self.preview_camera_combo.addItem("环绕车辆", "orbit")
        self.preview_camera_combo.addItem("自由跟随", "free")
        self.preview_height_spin = self._double_spin(150.0, 10.0, 500.0)
        self.realtime_preview_check = QCheckBox("实时预览")
        self.realtime_preview_check.setChecked(True)
        self.beamng_pick_check = QCheckBox("BeamNG 窗口点击拾点")
        self.beamng_pick_check.setEnabled(self.pick_callback is not None)
        self.beamng_pick_check.setChecked(self.pick_callback is not None)
        self.beamng_pose_label = QLabel("BeamNG 当前位置：未读取")
        self.beamng_pose_label.setObjectName("mutedText")
        self.beamng_pick_label = QLabel("BeamNG 点击拾点：未启用")
        self.beamng_pick_label.setObjectName("mutedText")

        self._load_existing_task(task_path)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.addRow("Task file", self.output_path_edit)
        form.addRow("Task id", self.task_id_edit)
        form.addRow("BeamNG level", self.level_edit)
        form.addRow("Start Z", self.start_z_spin)
        form.addRow("Start yaw", self.start_yaw_spin)
        form.addRow("Goal radius", self.goal_radius_spin)
        form.addRow("Max steps", self.max_steps_spin)
        form.addRow("Evaluation drive", self.evaluation_drive_combo)
        form.addRow("Preview camera", self.preview_camera_combo)
        form.addRow("Preview height", self.preview_height_spin)
        form.addRow("Realtime preview", self.realtime_preview_check)
        form.addRow("BeamNG click pick", self.beamng_pick_check)
        layout.addLayout(form)

        pose_row = QHBoxLayout()
        pose_row.addWidget(self.beamng_pose_label, 1)
        refresh_pose = QPushButton("读取 BeamNG 位置")
        refresh_pose.clicked.connect(self.refresh_beamng_pose)
        use_region = QPushButton("作为区域点")
        use_region.clicked.connect(self._use_beamng_pose_as_region_point)
        use_start = QPushButton("作为起点")
        use_start.clicked.connect(self._use_beamng_pose_as_start)
        use_goal = QPushButton("作为终点")
        use_goal.clicked.connect(self._use_beamng_pose_as_goal)
        use_route = QPushButton("加入路线")
        use_route.clicked.connect(self._use_beamng_pose_as_route_point)
        for button in (refresh_pose, use_region, use_start, use_goal, use_route):
            pose_row.addWidget(button)
        layout.addLayout(pose_row)
        layout.addWidget(self.beamng_pick_label)

        mode_row = QHBoxLayout()
        for label, mode in [
            ("选区域点", "region"),
            ("选起点", "start"),
            ("选终点", "goal"),
            ("加专家路径点", "route"),
        ]:
            button = QPushButton(label)
            button.setObjectName("modeButton")
            button.setCheckable(True)
            button.setChecked(self.canvas.mode == mode)
            button.clicked.connect(lambda checked=False, value=mode: self._toggle_canvas_mode(value))
            self.mode_buttons[mode] = button
            mode_row.addWidget(button)
        clear_region = QPushButton("清空区域")
        clear_region.clicked.connect(self.canvas.clear_region)
        clear_route = QPushButton("清空路径")
        clear_route.clicked.connect(self.canvas.clear_route)
        preview_button = QPushButton("保存并刷新 BeamNG 预览")
        preview_button.clicked.connect(self.preview_task)
        mode_row.addWidget(clear_region)
        mode_row.addWidget(clear_route)
        mode_row.addWidget(preview_button)
        layout.addLayout(mode_row)
        layout.addWidget(self.canvas, 1)
        self.canvas.mode_changed.connect(self._sync_mode_buttons)

        hint = QLabel("在这个窗口里编辑区域、起点、终点和专家路径；点击预览会保存当前草稿并刷新 BeamNG 画面，默认使用俯视高视角观察是否可通行。")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save_task)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._connect_realtime_preview_signals()
        self.schedule_realtime_preview()
        self._sync_beamng_pick_timer()

    def _toggle_canvas_mode(self, mode: str) -> None:
        self.canvas.set_mode(None if self.canvas.mode == mode else mode)
        self._sync_mode_buttons()

    def _sync_mode_buttons(self, *_args: Any) -> None:
        for mode, button in self.mode_buttons.items():
            button.setChecked(self.canvas.mode == mode)

    def save_task(self) -> None:
        self._save_task(close_after_save=True, show_errors=True)

    def preview_task(self) -> None:
        if not self._save_task(close_after_save=False, show_errors=False):
            return
        if self.preview_callback is not None and self.saved_task_path:
            self.preview_callback(
                self.saved_task_path,
                str(self.preview_camera_combo.currentData() or "topdown"),
                float(self.preview_height_spin.value()),
            )

    def schedule_realtime_preview(self, *_args: Any) -> None:
        if self.realtime_preview_check.isChecked():
            self.preview_timer.start()

    def _run_realtime_preview(self) -> None:
        if self.realtime_preview_check.isChecked():
            self.preview_task()

    def _connect_realtime_preview_signals(self) -> None:
        self.realtime_preview_check.toggled.connect(self.schedule_realtime_preview)
        self.beamng_pick_check.toggled.connect(self._sync_beamng_pick_timer)
        self.canvas.changed.connect(self.schedule_realtime_preview)
        for edit in (self.output_path_edit, self.task_id_edit, self.level_edit):
            edit.textChanged.connect(self.schedule_realtime_preview)
        for spin in (
            self.start_z_spin,
            self.start_yaw_spin,
            self.goal_radius_spin,
            self.max_steps_spin,
            self.preview_height_spin,
        ):
            spin.valueChanged.connect(self.schedule_realtime_preview)
        self.evaluation_drive_combo.currentIndexChanged.connect(self.schedule_realtime_preview)
        self.preview_camera_combo.currentIndexChanged.connect(self.schedule_realtime_preview)

    def _sync_beamng_pick_timer(self) -> None:
        if self.pick_callback is not None and self.beamng_pick_check.isChecked():
            self.beamng_pick_label.setText("BeamNG 点击拾点：等待窗口点击")
            self.picker_timer.start()
            return
        self.picker_timer.stop()
        self.beamng_pick_label.setText("BeamNG 点击拾点：未启用")

    def _poll_beamng_pick(self) -> None:
        if self.pick_callback is None or not self.beamng_pick_check.isChecked():
            return
        try:
            pick = self.pick_callback()
        except Exception as exc:
            self.beamng_pick_label.setText(f"BeamNG 点击拾点：不可用 ({exc})")
            return
        if not pick.get("available"):
            message = str(pick.get("message") or "等待窗口点击")
            self.beamng_pick_label.setText(f"BeamNG 点击拾点：{message}")
            return
        sequence = pick.get("sequence")
        if sequence is not None and sequence == self.last_beamng_pick_sequence:
            return
        self.last_beamng_pick_sequence = sequence
        try:
            x = float(pick["x"])
            y = float(pick["y"])
        except (KeyError, TypeError, ValueError):
            self.beamng_pick_label.setText("BeamNG 点击拾点：坐标无效")
            return
        payload = {"x": x, "y": y}
        for key in ("z", "yaw"):
            if self._finite_pose_value(pick.get(key)):
                payload[key] = float(pick[key])
        self.canvas.set_beamng_pose_marker((x, y))
        self._apply_beamng_pick(payload)
        z = payload.get("z", math.nan)
        self.beamng_pick_label.setText(
            f"BeamNG 点击拾点：sequence={sequence} mode={self.canvas.mode} x={x:.3f} y={y:.3f} z={z:.3f}"
        )

    def _apply_beamng_pick(self, pick: dict[str, float]) -> None:
        point = (float(pick["x"]), float(pick["y"]))
        if self.canvas.mode == "region":
            self.canvas.add_region_point(point)
        elif self.canvas.mode == "start":
            self.canvas.set_start_pose(point, pick.get("z"))
            if self._finite_pose_value(pick.get("z")):
                self.start_z_spin.setValue(float(pick["z"]))
            if self._finite_pose_value(pick.get("yaw")):
                self.start_yaw_spin.setValue(float(pick["yaw"]))
        elif self.canvas.mode == "goal":
            self.canvas.set_goal_point(point)
        elif self.canvas.mode == "route":
            self.canvas.add_route_point(point)

    def refresh_beamng_pose(self) -> dict[str, Any]:
        if self.pose_callback is None:
            pose = {"available": False, "message": "BeamNG preview session is not connected."}
        else:
            try:
                pose = self.pose_callback()
            except Exception as exc:
                pose = {"available": False, "message": str(exc)}
        self.current_beamng_pose = dict(pose)
        self._update_beamng_pose_label()
        if self.current_beamng_pose.get("available"):
            try:
                self.canvas.set_beamng_pose_marker((float(self.current_beamng_pose["x"]), float(self.current_beamng_pose["y"])))
            except (KeyError, TypeError, ValueError):
                self.canvas.set_beamng_pose_marker(None)
        else:
            self.canvas.set_beamng_pose_marker(None)
        return self.current_beamng_pose

    def _use_beamng_pose_as_region_point(self) -> None:
        pose = self._active_beamng_pose()
        if pose is None:
            return
        self.canvas.add_region_point((pose["x"], pose["y"]))

    def _use_beamng_pose_as_start(self) -> None:
        pose = self._active_beamng_pose()
        if pose is None:
            return
        self.canvas.set_start_pose((pose["x"], pose["y"]), pose.get("z"))
        if self._finite_pose_value(pose.get("z")):
            self.start_z_spin.setValue(float(pose["z"]))
        if self._finite_pose_value(pose.get("yaw")):
            self.start_yaw_spin.setValue(float(pose["yaw"]))

    def _use_beamng_pose_as_goal(self) -> None:
        pose = self._active_beamng_pose()
        if pose is None:
            return
        self.canvas.set_goal_point((pose["x"], pose["y"]))

    def _use_beamng_pose_as_route_point(self) -> None:
        pose = self._active_beamng_pose()
        if pose is None:
            return
        self.canvas.add_route_point((pose["x"], pose["y"]))

    def _active_beamng_pose(self) -> dict[str, float] | None:
        pose = self.current_beamng_pose
        if not pose.get("available"):
            pose = self.refresh_beamng_pose()
        if not pose.get("available"):
            return None
        try:
            x = float(pose["x"])
            y = float(pose["y"])
        except (KeyError, TypeError, ValueError):
            return None
        result: dict[str, float] = {"x": x, "y": y}
        for key in ("z", "yaw"):
            if self._finite_pose_value(pose.get(key)):
                result[key] = float(pose[key])
        return result

    def _update_beamng_pose_label(self) -> None:
        pose = self.current_beamng_pose
        if not pose.get("available"):
            message = str(pose.get("message") or "unavailable")
            self.beamng_pose_label.setText(f"BeamNG 当前位置：不可用 ({message})")
            return
        x = float(pose.get("x", math.nan))
        y = float(pose.get("y", math.nan))
        z = float(pose.get("z", math.nan))
        yaw = float(pose.get("yaw", math.nan))
        level = pose.get("level", self.level_edit.text().strip() or "unknown")
        self.beamng_pose_label.setText(
            f"BeamNG 当前位置：level={level}  x={x:.3f}  y={y:.3f}  z={z:.3f}  yaw={yaw:.3f}"
        )

    def _finite_pose_value(self, value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _save_task(self, *, close_after_save: bool, show_errors: bool = True) -> bool:
        try:
            if self.canvas.start is None:
                raise ValueError("Start point is not set.")
            if self.canvas.goal is None:
                raise ValueError("Goal point is not set.")
            start = (self.canvas.start[0], self.canvas.start[1], float(self.start_z_spin.value()))
            payload = services.save_manual_navigation_task(
                services.ManualNavigationTaskRequest(
                    output_path=self.output_path_edit.text().strip(),
                    task_id=self.task_id_edit.text().strip(),
                    level=self.level_edit.text().strip() or "gridmap_v2",
                    map_id=f"{self.level_edit.text().strip() or 'gridmap_v2'}_manual",
                    region_polygon=self.canvas.region,
                    start_pos=start,
                    start_yaw=float(self.start_yaw_spin.value()),
                    goal_pos=self.canvas.goal,
                    goal_radius=float(self.goal_radius_spin.value()),
                    expert_route=self.canvas.route,
                    max_steps=int(self.max_steps_spin.value()),
                    evaluation_drive_mode=self.evaluation_drive_combo.currentData() or "manual",
                )
            )
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "任务保存失败", str(exc))
            return False
        self.saved_task_path = str(payload["task_path"])
        if close_after_save:
            self.accept()
        return True

    def _load_existing_task(self, task_path: str) -> None:
        path = Path(task_path) if task_path else Path()
        if task_path and not path.is_absolute():
            path = services.ROOT / path
        if not path.exists():
            return
        try:
            task = services.load_navigation_region_task(path)
        except Exception:
            return
        self.output_path_edit.setText(str(path))
        self.task_id_edit.setText(task.task_id)
        self.level_edit.setText(task.level)
        self.start_z_spin.setValue(float(task.start_pos[2]))
        self.start_yaw_spin.setValue(float(task.start_yaw))
        self.goal_radius_spin.setValue(float(task.goal_radius))
        self.max_steps_spin.setValue(int(task.max_steps))
        mode = str(task.beamng.get("evaluation_drive_mode", "manual"))
        index = self.evaluation_drive_combo.findData(mode)
        if index >= 0:
            self.evaluation_drive_combo.setCurrentIndex(index)
        self.canvas.load_task(task)

    def _double_spin(self, value: float, minimum: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        return spin


class MainWindow(QMainWindow):
    PAGE_TITLES = [
        ("总览", "通过引导式 demo 快速检查配置并运行可视演示。"),
        ("数据集与训练", "导入/预览数据集，训练算法模型，并检查训练或推理结果。"),
        ("BeamNG 仿真", "编辑地图任务，选择模型配置，运行自动驾驶并评估。"),
        ("实验记录", "浏览 episode、轨迹、指标和日志。"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OffroadSimBench Desktop")
        self.settings = GuiSettings()
        self.catalog: dict[str, list[dict[str, Any]]] = {}
        self.threads: list[threading.Thread] = []
        self.workers: list[TaskWorker] = []
        self.detached_task_names: list[str] = []
        self.metric_cards: dict[str, MetricCard] = {}
        self.nav_buttons: list[QPushButton] = []
        self.dataset_info: dict[str, Any] | None = None
        self.navigation_preview_session = services.BeamNGNavigationPreviewSession()
        self.region_task_dialog: NavigationTaskDialog | None = None
        self._navigation_preview_busy = False
        self._navigation_preview_pending: tuple[str, str, float] | None = None
        self._dataset_preview_busy = False
        self.dataset_preview_session = services.DatasetPreviewSession(max_cached_frames=24)
        self._busy_depth = 0
        self.training_job_queue = services.TrainingJobQueue(max_parallel=1)
        self._current_training_job_id = ""
        self._handled_training_job_ids: set[str] = set()
        self._filtered_training_runs: list[dict[str, Any]] = []
        self.latest_experiment_comparison: dict[str, Any] = {}
        self.training_job_timer = QTimer(self)
        self.training_job_timer.setInterval(250)
        self.training_job_timer.timeout.connect(self._refresh_training_jobs)

        self._init_shared_controls()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_main_area(), 1)
        self.refresh_catalogs()
        self.select_page(0)
        self._resize_to_available_screen()

    def _init_shared_controls(self) -> None:
        self.backend_combo = self._combo()
        self.scenario_combo = self._combo()
        self.agent_combo = self._combo()
        self.world_model_combo = self._combo()
        self.planner_combo = self._combo()
        self.algorithm_combo = self._combo()
        self.demo_config_combo = self._combo()
        self.demo_config_combo.currentIndexChanged.connect(lambda _: self._refresh_demo_status())
        self.training_config_combo = self._combo()
        self.training_preset_combo = self._combo()
        self.training_artifact_combo = self._combo()
        self.training_artifact_combo.currentIndexChanged.connect(lambda _: self._update_training_artifact_details())
        self.inference_params_edit = QTextEdit()
        self.inference_params_edit.setFixedHeight(68)
        self.inference_params_edit.setPlaceholderText('{"max_samples": 8, "split_name": "test"}')
        self._inference_params_artifact_path = ""
        self.training_preset_summary = QTextEdit()
        self.training_preset_summary.setReadOnly(True)
        self.training_preset_summary.setFixedHeight(126)
        self.training_preset_summary.setPlaceholderText("Training config: NaN")
        self.world_model_config_combo = self._combo()
        self.world_model_config_edit_combo = self._combo()
        self.beamng_model_config_combo = self._combo()
        self.beamng_training_model_type_combo = self._combo()
        self.beamng_training_model_type_combo.addItem("MLP dynamics", "mlp_dynamics")
        self.beamng_training_model_type_combo.addItem("Tiny learned", "tiny_learned")
        self.trainer_params_edit = QTextEdit()
        self.trainer_params_edit.setPlaceholderText('{"epochs": 10, "batch_size": 16}')
        self.trainer_params_edit.setFixedHeight(96)
        self._trainer_params_autofill = ""
        self.trainer_parameter_form = QWidget()
        self.trainer_parameter_form_layout = QFormLayout(self.trainer_parameter_form)
        self.trainer_parameter_form_layout.setContentsMargins(0, 0, 0, 0)
        self.trainer_parameter_form_layout.setHorizontalSpacing(12)
        self.trainer_parameter_form_layout.setVerticalSpacing(8)
        self.trainer_parameter_controls: dict[str, QWidget] = {}
        self.trainer_parameter_specs: dict[str, dict[str, Any]] = {}
        self.trainer_entrypoint_edit = QLineEdit()
        self.trainer_entrypoint_edit.setPlaceholderText(r"D:\models\my_algorithm\train.py")
        self.trainer_arguments_edit = QTextEdit()
        self.trainer_arguments_edit.setPlainText(json.dumps(["{dataset_root}", "--output", "{output_dir}"], indent=2))
        self.trainer_arguments_edit.setFixedHeight(86)
        self.trainer_schema_edit = QTextEdit()
        self.trainer_schema_edit.setPlaceholderText('{"epochs": {"type": "int", "default": 10}}')
        self.trainer_schema_edit.setFixedHeight(86)
        self.training_config_name_edit = QLineEdit()
        self.training_config_name_edit.setPlaceholderText("Custom training config")
        self.training_output_edit = QLineEdit()
        self.training_output_edit.setPlaceholderText(r"outputs\models\custom_training_run")
        self.model_config_name_edit = QLineEdit()
        self.model_config_name_edit.setPlaceholderText("Johnson Valley MLP support-route validated")

        self.dataset_catalog_combo = self._combo()
        self.dataset_manifest_name_edit = QLineEdit()
        self.dataset_manifest_name_edit.setPlaceholderText("Custom driving dataset")
        self.dataset_manifest_sequences_edit = QTextEdit()
        self.dataset_manifest_sequences_edit.setPlaceholderText(
            '[{"id": "clip_001", "root": ".", "assets": {"front_rgb": "images/*.png"}}]'
        )
        self.dataset_manifest_sequences_edit.setFixedHeight(92)
        self.dataset_manifest_sequences_edit.setVisible(False)
        self.dataset_sequence_table = QTableWidget(0, 5)
        self.dataset_sequence_table.setHorizontalHeaderLabels(
            ["序列 ID", "相对目录", "位姿 CSV", "动作 CSV", "资产映射 JSON"]
        )
        self.dataset_sequence_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dataset_sequence_table.verticalHeader().setVisible(False)
        self.dataset_sequence_table.setMinimumHeight(150)
        self.dataset_sequence_table.setAlternatingRowColors(True)
        self.dataset_sequence_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dataset_sequence_table.itemSelectionChanged.connect(self._sync_sequence_from_table_selection)
        self._dataset_sequence_view_mode = "mapping"
        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.setPlaceholderText(r"datasets\ORFD_Dataset_ICRA2022_ZIP")
        self.dataset_split_path_edit = QLineEdit()
        self.dataset_split_path_edit.setPlaceholderText(r"outputs\dataset_splits\dataset_split.json")
        self.sequence_combo = self._combo(editable=True)
        self.sequence_combo.currentTextChanged.connect(self._sync_sequence_table_selection)
        self.adapter_edit = QLineEdit("orfd")
        self.stablewm_hdf5_edit = QLineEdit()
        self.stablewm_hdf5_edit.setPlaceholderText(r"outputs\stablewm\orfd_gui.h5")
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(str(services.DEFAULT_LEWM_CHECKPOINT_PATH))
        self.task_path_edit = QLineEdit(str(services.DEFAULT_NAVIGATION_TASK_PATH))
        self.region_collection_manifest_edit = QLineEdit()
        self.region_collection_manifest_edit.setPlaceholderText(r"outputs\beamng_region_training_data\...\region_training_collection.json")
        self.home_task_combo = self._combo(editable=True)
        self.beamng_task_combo = self._combo(editable=True)
        self.home_model_combo = self._combo(editable=True)
        self.home_task_combo.currentTextChanged.connect(self._sync_home_task_to_edit)
        self.beamng_task_combo.currentTextChanged.connect(self._sync_beamng_task_to_edit)
        self.home_model_combo.currentTextChanged.connect(self._sync_home_model_to_edit)
        self.world_model_config_combo.currentIndexChanged.connect(self._sync_home_world_model_config)
        self.world_model_config_edit_combo.currentIndexChanged.connect(self._sync_edit_world_model_config)
        self.beamng_model_config_combo.currentIndexChanged.connect(self._sync_beamng_world_model_config)
        self.training_config_combo.currentIndexChanged.connect(lambda _: self._sync_training_config_selection())
        self.training_preset_combo.currentIndexChanged.connect(lambda _: self._sync_training_preset_selection(force=True))
        self.dataset_catalog_combo.currentIndexChanged.connect(lambda _: self._sync_dataset_manifest_selection())
        self.dataset_root_edit.textChanged.connect(self._dataset_root_changed)
        for edit in (
            self.dataset_root_edit,
            self.adapter_edit,
            self.stablewm_hdf5_edit,
            self.model_path_edit,
            self.training_output_edit,
            self.training_config_name_edit,
            self.dataset_manifest_name_edit,
            self.trainer_entrypoint_edit,
            self.model_config_name_edit,
            self.task_path_edit,
            self.region_collection_manifest_edit,
        ):
            self._configure_control(edit)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(232)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(6)

        title = QLabel("OffroadSimBench")
        title.setObjectName("appTitle")
        subtitle = QLabel("本地越野仿真实验台")
        subtitle.setObjectName("mutedText")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(16)

        for index, (label, _) in enumerate(self.PAGE_TITLES):
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            self._configure_button(button, height=NAV_BUTTON_HEIGHT)
            button.clicked.connect(lambda checked=False, page=index: self.select_page(page))
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)
        self.runtime_label = QLabel("BeamNG: NaN")
        self.runtime_label.setObjectName("mutedText")
        layout.addWidget(self.runtime_label)
        return sidebar

    def _build_main_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(24, 20, 24, 22)
        layout.setSpacing(PAGE_SPACING)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(4)
        self.page_title = QLabel()
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel()
        self.page_subtitle.setObjectName("mutedText")
        self.page_subtitle.setWordWrap(True)
        title_box.addWidget(self.page_title)
        title_box.addWidget(self.page_subtitle)
        header.addLayout(title_box, 1)
        self.busy_label = QLabel("Ready")
        self.busy_label.setObjectName("busyLabel")
        self.busy_bar = QProgressBar()
        self.busy_bar.setObjectName("busyBar")
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedWidth(150)
        self.busy_bar.setMinimumHeight(8)
        self.busy_label.setVisible(False)
        self.busy_bar.setVisible(False)
        header.addWidget(self.busy_label)
        header.addWidget(self.busy_bar)

        advanced_button = QPushButton("高级参数")
        self._configure_button(advanced_button)
        advanced_button.clicked.connect(self.open_advanced_settings)
        self.refresh_button = QPushButton("刷新状态")
        self._configure_button(self.refresh_button)
        self.refresh_button.clicked.connect(self.refresh_catalogs)
        header.addWidget(advanced_button)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._build_overview_page())
        self.page_stack.addWidget(self._build_dataset_training_page())
        self.page_stack.addWidget(self._build_beamng_simulation_page())
        self.page_stack.addWidget(self._build_records_page())
        layout.addWidget(self.page_stack, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(area)
        return scroll

    def _resize_to_available_screen(self) -> None:
        target_width = INITIAL_WINDOW_WIDTH
        target_height = INITIAL_WINDOW_HEIGHT
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = min(target_width, max(480, available.width() - 40), available.width())
            target_height = min(target_height, max(360, available.height() - 80), available.height())
        self.resize(int(target_width), int(target_height))

    def _build_overview_page(self) -> QWidget:
        page, layout = self._page()

        body = self._row_layout()
        launcher_box, launcher_layout = self._new_group("Guided demo launcher")
        guide = QLabel(
            "选择一个标准 demo 配置，点击开始运行，然后在结果区查看验收指标。复杂的数据集、训练和任务编辑放到对应工作台。"
        )
        guide.setObjectName("mutedText")
        guide.setWordWrap(True)
        launcher_layout.addWidget(guide)
        launcher_layout.addWidget(self._field("Demo config", self.demo_config_combo))

        run_button = QPushButton("Start demo")
        self._configure_button(run_button, primary=True)
        run_button.clicked.connect(self.run_guided_demo)
        launcher_layout.addWidget(run_button)
        body.addWidget(launcher_box, 2)

        result_box, result_layout = self._new_group("Demo result")
        self.demo_result_summary = QTextEdit()
        self.demo_result_summary.setReadOnly(True)
        self.demo_result_summary.setPlaceholderText("Demo result: NaN")
        result_layout.addWidget(self.demo_result_summary, 1)
        body.addWidget(result_box, 1)
        layout.addLayout(body)

        metrics_box, metrics_layout = self._new_group("Demo metrics")
        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setHorizontalSpacing(CARD_SPACING)
        metrics.setVerticalSpacing(CARD_SPACING)
        for index, key in enumerate(["steps", "done", "best_cost", "final_speed", "max_risk", "reward"]):
            card = MetricCard(key.replace("_", " ").title())
            self.metric_cards[key] = card
            metrics.addWidget(card, index // 3, index % 3)
        metrics_layout.addLayout(metrics)
        layout.addWidget(metrics_box)
        layout.addStretch(1)
        return page

    def _build_dataset_training_page(self) -> QWidget:
        page, layout = self._page()
        tabs = QTabWidget()
        self.data_training_tabs = tabs

        data_tab = QWidget()
        data_root = QVBoxLayout(data_tab)
        data_root.setContentsMargins(0, 0, 0, 0)
        data_root.setSpacing(CARD_SPACING)
        data_root.addWidget(self._tab_header("数据集", "导入数据源，检查序列和数据质量，并同步预览不同模态。"))
        data_layout = self._row_layout()
        dataset_browse = QPushButton("选择文件夹")
        self._configure_button(dataset_browse)
        dataset_browse.clicked.connect(lambda: self._browse_dir(self.dataset_root_edit))
        self.dataset_import_button = QPushButton("导入数据源配置")
        self.dataset_import_button.setToolTip("导入已有的 dataset_manifest.yaml；直接选择普通数据集文件夹时无需使用。")
        self._configure_button(self.dataset_import_button)
        self.dataset_import_button.clicked.connect(self.import_dataset_manifest)
        self.dataset_save_button = QPushButton("保存为通用数据源")
        self.dataset_save_button.setToolTip("保存当前序列映射，供未内置适配器的数据集重复使用。")
        self._configure_button(self.dataset_save_button)
        self.dataset_save_button.clicked.connect(self.save_dataset_manifest_from_gui)
        self.dataset_detect_button = QPushButton("识别格式与序列")
        self.dataset_detect_button.setToolTip("自动选择已注册的数据集适配器，并读取该适配器发现的全部序列。")
        self._configure_button(self.dataset_detect_button)
        self.dataset_detect_button.clicked.connect(self.suggest_dataset_manifest_sequences_from_gui)
        self.dataset_sequence_add_button = QPushButton("添加序列")
        self.dataset_sequence_add_button.setToolTip("为没有内置适配器的数据集添加一条自定义序列映射。")
        self._configure_button(self.dataset_sequence_add_button)
        self.dataset_sequence_add_button.clicked.connect(self._add_dataset_sequence_row)
        self.dataset_sequence_remove_button = QPushButton("删除所选序列")
        self._configure_button(self.dataset_sequence_remove_button)
        self.dataset_sequence_remove_button.clicked.connect(self._remove_dataset_sequence_rows)
        source_toolbar = self._action_toolbar(
            [self.dataset_detect_button, self.dataset_import_button],
            object_name="datasetSourceToolbar",
        )
        manifest_toolbar = self._action_toolbar(
            [
                self.dataset_sequence_add_button,
                self.dataset_sequence_remove_button,
                self.dataset_save_button,
            ],
            object_name="datasetManifestToolbar",
        )
        self.dataset_catalog_combo.setToolTip("选择之前保存或导入的数据源配置；临时浏览文件夹可保持“临时数据集”。")
        self.dataset_sequence_table.setToolTip("仅在自定义数据集没有内置适配器时编辑序列目录和资产映射。")
        self.adapter_edit.setToolTip("识别后自动填写；也可手动指定已注册的适配器名称。")
        self._set_custom_dataset_controls(True)
        controls = self._group(
            "数据源",
            [
                self._field("已保存的数据源", self.dataset_catalog_combo),
                self._field("数据集根目录", self._with_button(self.dataset_root_edit, dataset_browse)),
                source_toolbar,
                self._field("保存名称（自定义数据源）", self.dataset_manifest_name_edit),
                self._field("序列列表 / 自定义映射", self.dataset_sequence_table),
                manifest_toolbar,
                self._field("当前序列", self.sequence_combo),
                self._field("识别到的适配器", self.adapter_edit),
                self._action_button("检查数据集", self.inspect_dataset),
                self._action_button("预览数据帧", self.preview_dataset, primary=True),
            ],
        )
        controls.setMinimumWidth(DATASET_SOURCE_MIN_WIDTH)
        self.dataset_source_panel = controls
        data_layout.addWidget(controls, 1)
        dataset_workspace = QTabWidget()
        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(CARD_SPACING)
        preview_controls = QWidget()
        preview_controls.setObjectName("datasetPreviewToolbar")
        preview_controls_layout = QHBoxLayout(preview_controls)
        preview_controls_layout.setContentsMargins(0, 0, 0, 0)
        preview_controls_layout.setSpacing(8)
        self.dataset_previous_button = QPushButton("上一帧")
        self._configure_button(self.dataset_previous_button)
        self.dataset_previous_button.clicked.connect(lambda: self._step_dataset_preview(-1))
        self.dataset_play_button = QPushButton("播放")
        self.dataset_play_button.setCheckable(True)
        self._configure_button(self.dataset_play_button)
        self.dataset_play_button.toggled.connect(self._toggle_dataset_playback)
        self.dataset_next_button = QPushButton("下一帧")
        self._configure_button(self.dataset_next_button)
        self.dataset_next_button.clicked.connect(lambda: self._step_dataset_preview(1))
        self.dataset_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.dataset_frame_slider.setRange(0, 0)
        self.dataset_frame_slider.sliderReleased.connect(self.preview_dataset)
        self.dataset_frame_label = QLabel("帧 0 / 0")
        self.dataset_frame_label.setObjectName("mutedText")
        preview_controls_layout.addWidget(self.dataset_previous_button)
        preview_controls_layout.addWidget(self.dataset_play_button)
        preview_controls_layout.addWidget(self.dataset_next_button)
        preview_controls_layout.addWidget(self.dataset_frame_slider, 1)
        preview_controls_layout.addWidget(self.dataset_frame_label)
        self.dataset_preview_timer = QTimer(self)
        self.dataset_preview_timer.setInterval(250)
        self.dataset_preview_timer.timeout.connect(self._advance_dataset_preview)
        preview_layout.addWidget(preview_controls)
        image_grid = QGridLayout()
        image_grid.setContentsMargins(0, 0, 0, 0)
        image_grid.setHorizontalSpacing(CARD_SPACING)
        image_grid.setVerticalSpacing(CARD_SPACING)
        self.rgb_preview = self._preview_label("RGB: NaN")
        self.depth_preview = self._preview_label("Depth/Label: NaN")
        self.lidar_preview = self._preview_label("LiDAR: NaN")
        image_grid.addWidget(self._preview_panel("RGB 预览", self.rgb_preview), 0, 0)
        image_grid.addWidget(self._preview_panel("深度 / 标签预览", self.depth_preview), 0, 1)
        image_grid.addWidget(self._preview_panel("LiDAR 预览", self.lidar_preview), 1, 0, 1, 2)
        preview_layout.addLayout(image_grid, 2)
        self.dataset_summary = QTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setPlaceholderText("数据集检查结果：NaN")
        preview_layout.addWidget(self._section_label("帧元数据"))
        preview_layout.addWidget(self.dataset_summary, 1)
        dataset_workspace.addTab(preview_page, "同步预览")

        detail_page = QWidget()
        detail_layout = QVBoxLayout(detail_page)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(CARD_SPACING)
        self.dataset_detail_summary = QTextEdit()
        self.dataset_detail_summary.setReadOnly(True)
        self.dataset_detail_summary.setMaximumHeight(180)
        self.dataset_detail_summary.setPlaceholderText("数据集详情：NaN")
        self.dataset_sequence_detail_table = QTableWidget(0, 6)
        self.dataset_sequence_detail_table.setHorizontalHeaderLabels(
            ["序列", "样本数", "模态", "开始时间", "结束时间", "间隔问题"]
        )
        self.dataset_sequence_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dataset_sequence_detail_table.verticalHeader().setVisible(False)
        self.dataset_sequence_detail_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        detail_layout.addWidget(self.dataset_detail_summary)
        detail_layout.addWidget(self.dataset_sequence_detail_table, 1)
        dataset_workspace.addTab(detail_page, "数据详情")

        quality_page = QWidget()
        quality_layout = QVBoxLayout(quality_page)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(CARD_SPACING)
        split_browse = QPushButton("选择")
        self._configure_button(split_browse)
        split_browse.clicked.connect(
            lambda: self._browse_file(self.dataset_split_path_edit, "选择数据划分文件")
        )
        self.dataset_train_ratio = QDoubleSpinBox()
        self.dataset_validation_ratio = QDoubleSpinBox()
        self.dataset_test_ratio = QDoubleSpinBox()
        for control, value in (
            (self.dataset_train_ratio, 0.7),
            (self.dataset_validation_ratio, 0.15),
            (self.dataset_test_ratio, 0.15),
        ):
            control.setRange(0.0, 1.0)
            control.setDecimals(2)
            control.setSingleStep(0.05)
            control.setValue(value)
            self._configure_control(control)
        split_fields = self._action_toolbar(
            [
                self._compact_field("训练", self.dataset_train_ratio),
                self._compact_field("验证", self.dataset_validation_ratio),
                self._compact_field("测试", self.dataset_test_ratio),
            ],
            object_name="datasetSplitToolbar",
        )
        quality_layout.addWidget(split_fields)
        quality_layout.addWidget(
            self._field("数据划分文件", self._with_button(self.dataset_split_path_edit, split_browse))
        )
        quality_layout.addWidget(
            self._action_toolbar(
                [
                    self._action_button("生成质量报告", self.run_dataset_quality_analysis, primary=True),
                    self._action_button("生成数据划分", self.create_dataset_split),
                ],
                object_name="datasetQualityToolbar",
            )
        )
        self.dataset_quality_summary = QTextEdit()
        self.dataset_quality_summary.setReadOnly(True)
        self.dataset_quality_summary.setMaximumHeight(190)
        self.dataset_quality_summary.setPlaceholderText("质量报告：NaN")
        self.dataset_issue_table = QTableWidget(0, 5)
        self.dataset_issue_table.setHorizontalHeaderLabels(["级别", "类型", "序列", "帧", "说明"])
        self.dataset_issue_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dataset_issue_table.verticalHeader().setVisible(False)
        self.dataset_issue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dataset_split_summary = QTextEdit()
        self.dataset_split_summary.setReadOnly(True)
        self.dataset_split_summary.setMaximumHeight(160)
        self.dataset_split_summary.setPlaceholderText("数据划分：NaN")
        quality_layout.addWidget(self._section_label("质量检查"))
        quality_layout.addWidget(self.dataset_quality_summary)
        quality_layout.addWidget(self.dataset_issue_table, 2)
        quality_layout.addWidget(self._section_label("训练 / 验证 / 测试划分"))
        quality_layout.addWidget(self.dataset_split_summary, 1)
        dataset_workspace.addTab(quality_page, "质量与划分")
        data_layout.addWidget(dataset_workspace, 2)
        data_root.addLayout(data_layout, 1)
        tabs.addTab(data_tab, "数据集")

        training_tab = QWidget()
        training_root = QVBoxLayout(training_tab)
        training_root.setContentsMargins(0, 0, 0, 0)
        training_root.setSpacing(CARD_SPACING)
        training_root.addWidget(self._tab_header("模型训练", "选择可复用训练配置，或注册本地训练脚本和参数。"))
        training_layout = self._row_layout()
        model_browse = QPushButton("选择")
        self._configure_button(model_browse)
        model_browse.clicked.connect(lambda: self._browse_path_combo(self.home_model_combo))
        model_import = QPushButton("导入模型 / Checkpoint")
        self._configure_button(model_import)
        model_import.clicked.connect(self.import_world_model_config)
        model_dir_import = QPushButton("导入模型目录")
        self._configure_button(model_dir_import)
        model_dir_import.clicked.connect(self.import_world_model_directory_config)
        trainer_import = QPushButton("导入训练器清单")
        self._configure_button(trainer_import)
        trainer_import.clicked.connect(self.import_trainer_manifest)
        trainer_entry_browse = QPushButton("选择")
        self._configure_button(trainer_entry_browse)
        trainer_entry_browse.clicked.connect(lambda: self._browse_file(self.trainer_entrypoint_edit, "Select trainer entrypoint"))
        self.save_trainer_button = QPushButton("从脚本保存训练器")
        self._configure_button(self.save_trainer_button)
        self.save_trainer_button.clicked.connect(self.save_trainer_manifest_from_gui)
        self.save_script_config_button = QPushButton("保存脚本训练配置")
        self._configure_button(self.save_script_config_button)
        self.save_script_config_button.clicked.connect(self.save_script_training_config_from_gui)
        training_config_import = QPushButton("导入训练配置")
        self._configure_button(training_config_import)
        training_config_import.clicked.connect(self.import_training_config)
        training_controls = self._group(
            "训练配置",
            [
                self._field("训练配置", self.training_config_combo),
                training_config_import,
                self._field("配置名称", self.training_config_name_edit),
                self._field("训练预设", self.training_preset_combo),
                self._section_label("配置摘要"),
                self.training_preset_summary,
                self._field("模型参数", self.trainer_parameter_form),
                self._field("高级 JSON 覆盖", self.trainer_params_edit),
                self._field("输出目录", self.training_output_edit),
                self._action_button("验证配置", self.validate_training_config),
                self._action_button("保存训练配置", self.save_training_config),
                self._action_button("开始训练 / 导出", self.run_training_preset, primary=True),
            ],
        )
        trainer_box = self._group(
            "训练器 / 算法",
            [
                trainer_import,
                self._field("训练入口", self._with_button(self.trainer_entrypoint_edit, trainer_entry_browse)),
                self._field("命令参数", self.trainer_arguments_edit),
                self._field("参数 Schema", self.trainer_schema_edit),
                self.save_trainer_button,
                self.save_script_config_button,
            ],
        )
        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(CARD_SPACING)
        left_column.addWidget(training_controls, 3)
        left_column.addWidget(trainer_box, 1)
        training_layout.addLayout(left_column, 1)
        output_box, output_layout = self._new_group("最近训练结果")
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setPlaceholderText("模型训练/推理结果：NaN")
        self.latest_training_curve = TrainingCurveWidget()
        self.training_job_status_label = QLabel("训练任务：NaN")
        self.training_job_status_label.setObjectName("mutedText")
        self.training_job_status_label.setWordWrap(True)
        self.training_job_progress = QProgressBar()
        self.training_job_progress.setRange(0, 100)
        self.training_job_progress.setValue(0)
        self.training_job_progress.setFormat("等待任务")
        self.cancel_training_button = QPushButton("取消任务")
        self._configure_button(self.cancel_training_button)
        self.cancel_training_button.setEnabled(False)
        self.cancel_training_button.clicked.connect(self.cancel_current_training_job)
        job_status_row = QWidget()
        job_status_layout = QHBoxLayout(job_status_row)
        job_status_layout.setContentsMargins(0, 0, 0, 0)
        job_status_layout.setSpacing(8)
        job_status_layout.addWidget(self.training_job_progress, 1)
        job_status_layout.addWidget(self.cancel_training_button)
        output_layout.addWidget(self.training_job_status_label)
        output_layout.addWidget(job_status_row)
        output_layout.addWidget(self._section_label("最近指标曲线"))
        latest_metric_toolbar = QWidget()
        latest_metric_toolbar_layout = QHBoxLayout(latest_metric_toolbar)
        latest_metric_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        latest_metric_toolbar_layout.setSpacing(8)
        latest_metric_toolbar_layout.addWidget(QLabel("指标"))
        self.latest_metric_combo = QComboBox()
        self.latest_metric_combo.setMinimumHeight(CONTROL_HEIGHT)
        self.latest_metric_combo.currentTextChanged.connect(
            lambda metric: self._select_training_metric(self.latest_training_curve, metric)
        )
        latest_metric_toolbar_layout.addWidget(self.latest_metric_combo, 1)
        latest_reset_button = QPushButton("重置缩放")
        self._configure_button(latest_reset_button)
        latest_reset_button.clicked.connect(self.latest_training_curve.reset_zoom)
        latest_metric_toolbar_layout.addWidget(latest_reset_button)
        output_layout.addWidget(latest_metric_toolbar)
        self.latest_metric_summary = QLabel("指标曲线：NaN")
        self.latest_metric_summary.setObjectName("mutedText")
        self.latest_metric_summary.setWordWrap(True)
        output_layout.addWidget(self.latest_metric_summary)
        self.latest_metric_warning_label = QLabel("训练诊断：NaN")
        self.latest_metric_warning_label.setObjectName("mutedText")
        self.latest_metric_warning_label.setWordWrap(True)
        output_layout.addWidget(self.latest_metric_warning_label)
        output_layout.addWidget(self.latest_training_curve)
        self.latest_training_log = QTextEdit()
        self.latest_training_log.setReadOnly(True)
        self.latest_training_log.setMaximumHeight(100)
        self.latest_training_log.setPlaceholderText("Training logs: NaN")
        output_layout.addWidget(self._section_label("训练日志"))
        output_layout.addWidget(self.latest_training_log)
        output_layout.addWidget(self._section_label("训练输出"))
        output_layout.addWidget(self.model_summary, 1)
        registry_box = self._group(
            "已训练模型",
            [
                self._field("世界模型配置", self.world_model_config_edit_combo),
                self._field("配置名称", self.model_config_name_edit),
                self._field("模型路径", self._with_button(self.home_model_combo, model_browse)),
                model_import,
                model_dir_import,
                self._field("算法", self.algorithm_combo),
                self._field("世界模型", self.world_model_combo),
                self._action_button("注册最近训练产物", self.register_latest_training_artifact_model),
                self._action_button("保存世界模型配置", self.save_world_model_config, primary=True),
            ],
        )
        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(CARD_SPACING)
        right_column.addWidget(output_box, 3)
        right_column.addWidget(registry_box, 1)
        training_layout.addLayout(right_column, 2)
        training_root.addLayout(training_layout, 1)
        tabs.addTab(training_tab, "模型训练")

        runs_tab = QWidget()
        runs_root = QVBoxLayout(runs_tab)
        runs_root.setContentsMargins(0, 0, 0, 0)
        runs_root.setSpacing(CARD_SPACING)
        runs_root.addWidget(self._tab_header("训练结果", "查看训练记录、指标曲线、产物、日志和已注册模型。"))
        runs_layout = self._row_layout()
        run_list_box, run_list_layout = self._new_group("训练记录")
        self.training_job_table = QTableWidget(0, 4)
        self.training_job_table.setHorizontalHeaderLabels(["任务", "状态", "进度", "输出目录"])
        self.training_job_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.training_job_table.verticalHeader().setVisible(False)
        self.training_job_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.training_job_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.training_job_table.setMaximumHeight(190)
        self.training_job_table.cellClicked.connect(self._select_training_job_row)
        run_list_layout.addWidget(self._section_label("任务队列"))
        run_list_layout.addWidget(self.training_job_table)
        run_list_layout.addWidget(self._section_label("训练记录"))
        filter_grid = QGridLayout()
        filter_grid.setContentsMargins(0, 0, 0, 0)
        filter_grid.setHorizontalSpacing(8)
        filter_grid.setVerticalSpacing(8)
        self.experiment_query_edit = QLineEdit()
        self.experiment_query_edit.setPlaceholderText("搜索模型、数据集或 run id")
        self.experiment_status_filter = QComboBox()
        self.experiment_preset_filter = QComboBox()
        self.experiment_dataset_filter = QComboBox()
        for combo in (
            self.experiment_status_filter,
            self.experiment_preset_filter,
            self.experiment_dataset_filter,
        ):
            combo.setMinimumHeight(CONTROL_HEIGHT)
        self.experiment_date_from_edit = QLineEdit()
        self.experiment_date_from_edit.setPlaceholderText("起始日期 YYYY-MM-DD")
        self.experiment_date_to_edit = QLineEdit()
        self.experiment_date_to_edit.setPlaceholderText("结束日期 YYYY-MM-DD")
        for edit in (
            self.experiment_query_edit,
            self.experiment_date_from_edit,
            self.experiment_date_to_edit,
        ):
            edit.setMinimumHeight(CONTROL_HEIGHT)
        filter_grid.addWidget(self.experiment_query_edit, 0, 0, 1, 2)
        filter_grid.addWidget(self.experiment_status_filter, 1, 0)
        filter_grid.addWidget(self.experiment_preset_filter, 1, 1)
        filter_grid.addWidget(self.experiment_dataset_filter, 2, 0, 1, 2)
        filter_grid.addWidget(self.experiment_date_from_edit, 3, 0)
        filter_grid.addWidget(self.experiment_date_to_edit, 3, 1)
        run_list_layout.addLayout(filter_grid)
        self.training_run_list = QListWidget()
        self.training_run_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.training_run_list.itemClicked.connect(self._load_selected_training_run)
        run_list_layout.addWidget(self.training_run_list, 1)
        run_action_row = QWidget()
        run_action_layout = QHBoxLayout(run_action_row)
        run_action_layout.setContentsMargins(0, 0, 0, 0)
        run_action_layout.setSpacing(8)
        compare_runs_button = QPushButton("对比所选")
        self._configure_button(compare_runs_button)
        compare_runs_button.clicked.connect(self.compare_selected_training_runs)
        rerun_button = QPushButton("重新运行")
        self._configure_button(rerun_button)
        rerun_button.clicked.connect(self.rerun_selected_training_run)
        clone_run_button = QPushButton("复制配置")
        self._configure_button(clone_run_button)
        clone_run_button.clicked.connect(self.clone_selected_training_run)
        cleanup_runs_button = QPushButton("清理无效")
        self._configure_button(cleanup_runs_button)
        cleanup_runs_button.clicked.connect(self.cleanup_selected_training_runs)
        run_action_layout.addWidget(compare_runs_button)
        run_action_layout.addWidget(rerun_button)
        run_action_layout.addWidget(clone_run_button)
        run_action_layout.addWidget(cleanup_runs_button)
        run_list_layout.addWidget(run_action_row)
        for control in (
            self.experiment_query_edit,
            self.experiment_date_from_edit,
            self.experiment_date_to_edit,
        ):
            control.textChanged.connect(self._fill_training_run_list)
        for control in (
            self.experiment_status_filter,
            self.experiment_preset_filter,
            self.experiment_dataset_filter,
        ):
            control.currentIndexChanged.connect(self._fill_training_run_list)
        runs_layout.addWidget(run_list_box, 1)
        run_summary_box, run_summary_layout = self._new_group("运行详情")
        self.training_run_overview = QTextEdit()
        self.training_run_overview.setReadOnly(True)
        self.training_run_overview.setMaximumHeight(150)
        self.training_run_overview.setPlaceholderText("Training run summary: NaN")
        run_summary_layout.addWidget(self._section_label("运行摘要"))
        run_summary_layout.addWidget(self.training_run_overview)
        inference_row = QWidget()
        inference_row_layout = QHBoxLayout(inference_row)
        inference_row_layout.setContentsMargins(0, 0, 0, 0)
        inference_row_layout.setSpacing(8)
        inference_row_layout.addWidget(self.training_artifact_combo, 1)
        inference_button = QPushButton("运行推理")
        self._configure_button(inference_button)
        inference_button.clicked.connect(self.run_selected_artifact_inference)
        inference_row_layout.addWidget(inference_button)
        self.favorite_artifact_button = QPushButton("收藏")
        self._configure_button(self.favorite_artifact_button)
        self.favorite_artifact_button.clicked.connect(self.toggle_selected_artifact_favorite)
        inference_row_layout.addWidget(self.favorite_artifact_button)
        delete_artifact_button = QPushButton("删除产物")
        self._configure_button(delete_artifact_button)
        delete_artifact_button.clicked.connect(self.delete_selected_training_artifact)
        inference_row_layout.addWidget(delete_artifact_button)
        run_summary_layout.addWidget(self._section_label("Checkpoint 推理"))
        run_summary_layout.addWidget(inference_row)
        run_summary_layout.addWidget(self._field("推理参数", self.inference_params_edit))
        self.training_artifact_detail_label = QLabel("Checkpoint：NaN")
        self.training_artifact_detail_label.setObjectName("mutedText")
        self.training_artifact_detail_label.setWordWrap(True)
        run_summary_layout.addWidget(self.training_artifact_detail_label)
        self.inference_metric_summary = QLabel("推理指标：NaN")
        self.inference_metric_summary.setObjectName("mutedText")
        self.inference_metric_summary.setWordWrap(True)
        run_summary_layout.addWidget(self.inference_metric_summary)
        self.inference_prediction_table = QTableWidget(0, 6)
        self.inference_prediction_table.setHorizontalHeaderLabels(
            ["样本", "预测 X", "预测 Y", "真值 X", "真值 Y", "位置误差"]
        )
        self.inference_prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.inference_prediction_table.verticalHeader().setVisible(False)
        self.inference_prediction_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.inference_prediction_table.setMaximumHeight(180)
        run_summary_layout.addWidget(self.inference_prediction_table)
        self.inference_preview = self._preview_label("推理预览：NaN")
        self.inference_preview.setMaximumHeight(240)
        run_summary_layout.addWidget(self.inference_preview)
        self.training_curve = TrainingCurveWidget()
        run_summary_layout.addWidget(self._section_label("指标曲线"))
        run_metric_toolbar = QWidget()
        run_metric_toolbar_layout = QHBoxLayout(run_metric_toolbar)
        run_metric_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        run_metric_toolbar_layout.setSpacing(8)
        run_metric_toolbar_layout.addWidget(QLabel("指标"))
        self.training_metric_combo = QComboBox()
        self.training_metric_combo.setMinimumHeight(CONTROL_HEIGHT)
        self.training_metric_combo.currentTextChanged.connect(
            lambda metric: self._select_training_metric(self.training_curve, metric)
        )
        run_metric_toolbar_layout.addWidget(self.training_metric_combo, 1)
        export_metrics_button = QPushButton("导出指标与曲线")
        self._configure_button(export_metrics_button)
        export_metrics_button.clicked.connect(self.export_selected_training_metrics)
        run_metric_toolbar_layout.addWidget(export_metrics_button)
        run_summary_layout.addWidget(run_metric_toolbar)
        self.training_run_metric_summary = QLabel("指标曲线：NaN")
        self.training_run_metric_summary.setObjectName("mutedText")
        self.training_run_metric_summary.setWordWrap(True)
        run_summary_layout.addWidget(self.training_run_metric_summary)
        self.training_metric_warning_label = QLabel("训练诊断：NaN")
        self.training_metric_warning_label.setObjectName("mutedText")
        self.training_metric_warning_label.setWordWrap(True)
        run_summary_layout.addWidget(self.training_metric_warning_label)
        run_summary_layout.addWidget(self.training_curve)
        self.training_run_summary = QTextEdit()
        self.training_run_summary.setReadOnly(True)
        self.training_run_summary.setPlaceholderText("Training run: NaN")
        run_summary_layout.addWidget(self._section_label("原始 training_run.json"))
        run_summary_layout.addWidget(self.training_run_summary, 1)

        comparison_page = QWidget()
        comparison_layout = QVBoxLayout(comparison_page)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(10)
        comparison_toolbar = QWidget()
        comparison_toolbar_layout = QHBoxLayout(comparison_toolbar)
        comparison_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        comparison_toolbar_layout.setSpacing(8)
        comparison_toolbar_layout.addWidget(QLabel("排名指标"))
        self.experiment_metric_combo = QComboBox()
        self.experiment_metric_combo.setMinimumHeight(CONTROL_HEIGHT)
        comparison_toolbar_layout.addWidget(self.experiment_metric_combo, 1)
        self.experiment_direction_combo = QComboBox()
        self.experiment_direction_combo.setMinimumHeight(CONTROL_HEIGHT)
        self.experiment_direction_combo.addItem("自动方向", "auto")
        self.experiment_direction_combo.addItem("越小越好", "min")
        self.experiment_direction_combo.addItem("越大越好", "max")
        comparison_toolbar_layout.addWidget(self.experiment_direction_combo)
        rerank_button = QPushButton("重新排名")
        self._configure_button(rerank_button)
        rerank_button.clicked.connect(self.compare_selected_training_runs)
        comparison_toolbar_layout.addWidget(rerank_button)
        comparison_layout.addWidget(comparison_toolbar)
        self.experiment_comparison_summary = QLabel("实验对比：请选择至少两次训练。")
        self.experiment_comparison_summary.setObjectName("mutedText")
        self.experiment_comparison_summary.setWordWrap(True)
        comparison_layout.addWidget(self.experiment_comparison_summary)
        self.experiment_comparison_table = QTableWidget(0, 5)
        self.experiment_comparison_table.setHorizontalHeaderLabels(["排名", "实验", "指标值", "状态", "参数"])
        self.experiment_comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.experiment_comparison_table.verticalHeader().setVisible(False)
        self.experiment_comparison_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.experiment_comparison_table.setMaximumHeight(220)
        comparison_layout.addWidget(self.experiment_comparison_table)
        self.experiment_comparison_curve = TrainingCurveWidget()
        comparison_layout.addWidget(self.experiment_comparison_curve)
        comparison_actions = QWidget()
        comparison_actions_layout = QHBoxLayout(comparison_actions)
        comparison_actions_layout.setContentsMargins(0, 0, 0, 0)
        comparison_actions_layout.setSpacing(8)
        mark_best_button = QPushButton("标记最佳")
        self._configure_button(mark_best_button)
        mark_best_button.clicked.connect(self.mark_best_experiment)
        export_report_button = QPushButton("导出 Markdown / HTML")
        self._configure_button(export_report_button)
        export_report_button.clicked.connect(self.export_experiment_comparison_report)
        comparison_actions_layout.addWidget(mark_best_button)
        comparison_actions_layout.addWidget(export_report_button)
        comparison_actions_layout.addStretch(1)
        comparison_layout.addWidget(comparison_actions)
        comparison_layout.addStretch(1)

        self.training_result_tabs = QTabWidget()
        self.training_result_tabs.addTab(run_summary_box, "运行详情")
        self.training_result_tabs.addTab(comparison_page, "实验对比")
        runs_layout.addWidget(self.training_result_tabs, 2)
        runs_root.addLayout(runs_layout, 1)
        tabs.addTab(runs_tab, "训练结果")

        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.setSpacing(PAGE_SPACING)
        processing_layout.addWidget(self._tab_header("处理与标注", "集中放置分割、Mask、标签检查和未来的数据集转换工具。"))
        processing_hint = QLabel(
            "图像分割、标签检查、terrain mask 和数据集到 BeamNG 地图转换会放在这里；未实现项保持 NaN/未完成。"
        )
        processing_hint.setObjectName("mutedText")
        processing_hint.setWordWrap(True)
        processing_layout.addWidget(self._group("数据处理与标注", [processing_hint]))
        processing_layout.addStretch(1)
        tabs.addTab(processing_tab, "处理与标注")

        layout.addWidget(tabs, 1)
        return page

    def _build_beamng_simulation_page(self) -> QWidget:
        page, layout = self._page()
        tabs = QTabWidget()

        setup_tab = QWidget()
        setup_layout = QHBoxLayout(setup_tab)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(PAGE_SPACING)
        config_box = self._group(
            "Run configuration",
            [
                self._compact_field("Region task", self.beamng_task_combo),
                self._compact_field("World model config", self.beamng_model_config_combo),
                self._compact_field("Resolved task path", self.task_path_edit),
            ],
        )
        collect_box, collect_layout = self._new_group("采集训练数据")
        collect_box.setObjectName("beamngWorkflowCollect")
        collect_hint = QLabel("按当前区域任务沿专家路线分段采集，并生成 coverage/goal-zone 质量门槛报告。")
        collect_hint.setObjectName("mutedText")
        collect_hint.setWordWrap(True)
        collect_layout.addWidget(collect_hint)
        collect_layout.addWidget(
            self._action_toolbar(
                [
                    self._action_button("编辑/预览区域", self.open_region_task_editor),
                    self._action_button("采集训练数据", self.collect_region_training_data),
                ],
                object_name="beamngWorkflowCollectToolbar",
            )
        )
        train_box, train_layout = self._new_group("训练模型")
        train_box.setObjectName("beamngWorkflowTrain")
        train_hint = QLabel("从 collection manifest 训练 world model；采集质量不达标时会拒绝训练。")
        train_hint.setObjectName("mutedText")
        train_hint.setWordWrap(True)
        train_layout.addWidget(train_hint)
        train_layout.addWidget(self._compact_field("Training model", self.beamng_training_model_type_combo))
        train_layout.addWidget(self._compact_field("Collection manifest", self.region_collection_manifest_edit))
        train_layout.addWidget(
            self._action_toolbar(
                [
                    self._action_button("训练模型", self.train_region_world_model_from_collection),
                    self._action_button("区域自监督训练", self.train_region_self_supervised_world_model),
                ],
                object_name="beamngWorkflowTrainToolbar",
            )
        )
        evaluate_box, evaluate_layout = self._new_group("评估模型控车")
        evaluate_box.setObjectName("beamngWorkflowEvaluate")
        evaluate_hint = QLabel("使用当前任务和模型配置运行 route-free，并同时生成 route-guided baseline 对比。")
        evaluate_hint.setObjectName("mutedText")
        evaluate_hint.setWordWrap(True)
        evaluate_layout.addWidget(evaluate_hint)
        evaluate_layout.addWidget(
            self._action_toolbar(
                [
                    self._action_button("开始评估", self.run_region_navigation_loop, primary=True),
                    self._action_button("检查 BeamNG", self.check_beamng),
                ],
                object_name="beamngWorkflowEvaluateToolbar",
            )
        )
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(PAGE_SPACING)
        left_layout.addWidget(config_box)
        left_layout.addWidget(collect_box)
        left_layout.addWidget(train_box)
        left_layout.addWidget(evaluate_box)
        left_layout.addStretch(1)
        setup_layout.addWidget(left_column, 1)
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(PAGE_SPACING)
        quality_box, quality_layout = self._new_group("训练质量报告")
        quality_box.setObjectName("beamngQualityReport")
        self.beamng_quality_report = QTextEdit()
        self.beamng_quality_report.setReadOnly(True)
        self.beamng_quality_report.setPlaceholderText("Training quality report: NaN")
        self.beamng_quality_report.setMinimumHeight(150)
        quality_layout.addWidget(self._section_label("Coverage / evaluation metrics"))
        quality_layout.addWidget(self.beamng_quality_report, 1)
        self.beamng_quality_curve = TrainingCurveWidget()
        self.beamng_quality_curve.setMinimumHeight(160)
        quality_layout.addWidget(self._section_label("Latest metric curve"))
        quality_layout.addWidget(self.beamng_quality_curve)
        self.beamng_trajectory_plot = self._preview_label("Trajectory plot: NaN")
        self.beamng_trajectory_plot.setMinimumHeight(180)
        quality_layout.addWidget(self._section_label("Trajectory plot"))
        quality_layout.addWidget(self.beamng_trajectory_plot, 1)
        summary_box, summary_layout = self._new_group("Simulation status")
        self.beamng_summary = QTextEdit()
        self.beamng_summary.setReadOnly(True)
        self.beamng_summary.setPlaceholderText("BeamNG 状态、任务分析与运行结果：NaN")
        summary_layout.addWidget(self.beamng_summary, 1)
        right_layout.addWidget(quality_box, 2)
        right_layout.addWidget(summary_box, 1)
        setup_layout.addWidget(right_column, 2)
        tabs.addTab(setup_tab, "运行配置")

        map_tab = QWidget()
        map_layout = QHBoxLayout(map_tab)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(PAGE_SPACING)
        map_controls, map_controls_layout = self._new_group("Map and terrain tools")
        map_controls_layout.addWidget(
            self._action_toolbar(
                [
                    self._action_button("编辑/预览区域", self.open_region_task_editor),
                    self._action_button("导出地形草案", self.export_beamng_terrain_draft),
                ],
                object_name="beamngMapActionToolbar",
            )
        )
        map_controls_layout.addStretch(1)
        map_layout.addWidget(map_controls, 1)
        preview_box, preview_layout = self._new_group("Terrain draft preview")
        self.terrain_preview = self._preview_label("Terrain: NaN")
        preview_layout.addWidget(self.terrain_preview, 1)
        map_layout.addWidget(preview_box, 2)
        tabs.addTab(map_tab, "地图/区域")

        eval_tab = QWidget()
        eval_layout = QVBoxLayout(eval_tab)
        eval_layout.setContentsMargins(0, 0, 0, 0)
        eval_layout.setSpacing(PAGE_SPACING)
        self.planner_summary = QTextEdit()
        self.planner_summary.setReadOnly(True)
        self.planner_summary.setMaximumHeight(220)
        hint = QLabel("规划器和 CEM 参数在高级参数中调整；运行后指标会写入总览和实验记录。")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        advanced_button = QPushButton("打开高级参数")
        self._configure_button(advanced_button)
        advanced_button.clicked.connect(self.open_advanced_settings)
        eval_layout.addWidget(self._group("Evaluation and planner settings", [hint, self.planner_summary, advanced_button]))
        eval_layout.addStretch(1)
        tabs.addTab(eval_tab, "评估")

        layout.addWidget(tabs, 1)
        return page

    def _build_records_page(self) -> QWidget:
        page, layout = self._page()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        trajectory_box, trajectory_layout = self._new_group("轨迹预览")
        self.trajectory = TrajectoryCanvas()
        trajectory_layout.addWidget(self.trajectory, 1)
        splitter.addWidget(trajectory_box)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        log_box, log_layout = self._new_group("运行日志")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("运行日志")
        log_layout.addWidget(self.log_view, 1)
        episode_box, episode_layout = self._new_group("Episode")
        self.episode_list = QListWidget()
        self.episode_list.itemClicked.connect(self.load_selected_episode)
        episode_layout.addWidget(self.episode_list, 1)
        bottom.addWidget(log_box)
        bottom.addWidget(episode_box)
        bottom.setStretchFactor(0, 2)
        bottom.setStretchFactor(1, 1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return page

    def select_page(self, index: int) -> None:
        self.page_stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        title, subtitle = self.PAGE_TITLES[index]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)
        if index == 2:
            self._refresh_planner_summary()

    def refresh_catalogs(self) -> None:
        self.catalog = services.catalog_snapshot()
        self._fill_combo(self.backend_combo, self.catalog["backends"], "name", default="gym_heightmap")
        self._fill_combo(self.scenario_combo, self.catalog["scenarios"], "id", default="forest_trail_001")
        self._fill_combo(self.agent_combo, self.catalog["agents"], "name", default="rule_based")
        self._fill_combo(self.algorithm_combo, self.catalog["algorithms"], "name", default="stablewm_lewm")
        self._fill_combo(self.world_model_combo, self.catalog["world_models"], "name", default="le_wm")
        self._fill_combo(self.planner_combo, [{"name": ""}] + self.catalog["planners"], "name", default="navigation_mpc")
        self._fill_demo_config_combo()
        self._fill_path_combo(
            self.home_task_combo,
            self.catalog.get("navigation_tasks", []),
            default_path=str(services.DEFAULT_NAVIGATION_TASK_PATH),
        )
        self._fill_path_combo(
            self.beamng_task_combo,
            self.catalog.get("navigation_tasks", []),
            default_path=str(services.DEFAULT_NAVIGATION_TASK_PATH),
        )
        self._fill_path_combo(
            self.home_model_combo,
            self.catalog.get("model_checkpoints", []),
            default_path=str(services.DEFAULT_LEWM_CHECKPOINT_PATH),
        )
        self._fill_world_model_config_combo(
            self.world_model_config_combo,
            self.catalog.get("demo_ready_world_model_configs", []),
            default_id=services.DEFAULT_WORLD_MODEL_CONFIG_ID,
        )
        self._fill_world_model_config_combo(
            self.world_model_config_edit_combo,
            self.catalog.get("world_model_configs", []),
            default_id=services.DEFAULT_WORLD_MODEL_CONFIG_ID,
        )
        self._fill_world_model_config_combo(
            self.beamng_model_config_combo,
            self.catalog.get("world_model_configs", []),
            default_id=services.DEFAULT_WORLD_MODEL_CONFIG_ID,
        )
        self._fill_dataset_manifest_combo()
        self._fill_training_preset_combo()
        self._fill_training_config_combo()
        self._fill_training_run_list()
        self._fill_training_job_table(self.catalog.get("training_jobs", []))
        self._fill_training_artifact_combo()
        self._fill_episode_list()
        self._refresh_planner_summary()
        beamng = _find_named(self.catalog["backends"], "beamng")
        self.runtime_label.setText(f"BeamNG: {services.display_value(beamng.get('available') if beamng else None)}")
        if hasattr(self, "demo_result_summary"):
            self._refresh_demo_status()
        self.log("状态已刷新")

    def open_advanced_settings(self) -> None:
        dialog = AdvancedSettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.values()
            self._refresh_planner_summary()
            self.log(f"高级参数已更新: {_compact_json(asdict(self.settings))}")

    def open_region_task_editor(self) -> None:
        if self.region_task_dialog is not None:
            self.region_task_dialog.show()
            self.region_task_dialog.raise_()
            self.region_task_dialog.activateWindow()
            return
        dialog = NavigationTaskDialog(
            self.task_path_edit.text().strip(),
            None,
            preview_callback=self._preview_task_from_editor,
            pose_callback=self._read_navigation_preview_pose,
            pick_callback=self._consume_navigation_preview_pick,
        )
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.accepted.connect(lambda dialog=dialog: self._region_task_editor_saved(dialog))
        dialog.destroyed.connect(lambda *_args: self._clear_region_task_editor(dialog))
        self.region_task_dialog = dialog
        dialog.show()

    def _region_task_editor_saved(self, dialog: NavigationTaskDialog) -> None:
        if not dialog.saved_task_path:
            return
        self.task_path_edit.setText(dialog.saved_task_path)
        self.home_task_combo.setCurrentText(dialog.saved_task_path)
        self.beamng_task_combo.setCurrentText(dialog.saved_task_path)
        self.beamng_summary.setText(_compact_json({"status": "task_saved", "task_path": dialog.saved_task_path}))
        self.log(f"区域任务已保存: {dialog.saved_task_path}")

    def _clear_region_task_editor(self, dialog: NavigationTaskDialog) -> None:
        if self.region_task_dialog is dialog:
            self.region_task_dialog = None
            self._navigation_preview_pending = None

    def _preview_task_from_editor(self, task_path: str, camera_mode: str, camera_height_m: float) -> None:
        self.task_path_edit.setText(task_path)
        self.home_task_combo.setCurrentText(task_path)
        self.beamng_task_combo.setCurrentText(task_path)
        request = (task_path, camera_mode, camera_height_m)
        if self._navigation_preview_busy:
            self._navigation_preview_pending = request
            self.beamng_summary.setText(
                _compact_json(
                    {
                        "status": "preview_queued",
                        "task_path": task_path,
                        "camera_mode": camera_mode,
                        "camera_height_m": camera_height_m,
                    }
                )
            )
            return
        self._navigation_preview_busy = True
        self.log(f"BeamNG 预览区域/起终点: {task_path}, camera={camera_mode}, height={camera_height_m:.1f}m")
        self._run_task(
            lambda: self.navigation_preview_session.update(
                task_path,
                camera_mode=camera_mode,
                camera_height_m=camera_height_m,
            ),
            self._navigation_preview_finished,
            "navigation realtime preview failed",
            task_label="刷新 BeamNG 预览",
            cancel_hook=self.navigation_preview_session.close,
        )

    def _finish_navigation_preview_task(self) -> None:
        self._navigation_preview_busy = False
        pending = self._navigation_preview_pending
        self._navigation_preview_pending = None
        if pending is not None:
            self._preview_task_from_editor(*pending)

    def _read_navigation_preview_pose(self) -> dict[str, Any]:
        return self.navigation_preview_session.current_pose()

    def _consume_navigation_preview_pick(self) -> dict[str, Any]:
        return self.navigation_preview_session.consume_picker_pick()

    def closeEvent(self, event: Any) -> None:
        if hasattr(self, "dataset_preview_timer"):
            self.dataset_preview_timer.stop()
        if hasattr(self, "training_job_timer"):
            self.training_job_timer.stop()
        if hasattr(self, "training_job_queue"):
            self.training_job_queue.close(cancel_running=True)
        for worker in list(self.workers):
            worker.cancel()
        if self.region_task_dialog is not None:
            self.region_task_dialog.close()
            self.region_task_dialog = None
        self.navigation_preview_session.close()
        deadline = time.monotonic() + 1.5
        for thread in list(self.threads):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        alive_pairs = [
            (thread, worker)
            for thread, worker in zip(self.threads, self.workers, strict=False)
            if thread.is_alive()
        ]
        self.threads = [thread for thread, _ in alive_pairs]
        self.workers = [worker for _, worker in alive_pairs]
        self.detached_task_names = [thread.name for thread, _ in alive_pairs]
        if self.detached_task_names:
            self.log("后台任务仍在安全退出：" + ", ".join(self.detached_task_names))
        self._busy_depth = 0
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        super().closeEvent(event)

    def run_guided_demo(self) -> None:
        demo = self._combo_config_row(self.demo_config_combo)
        agent = "model_mpc"
        planner = "navigation_mpc"
        if demo:
            task_path = str(demo.get("task_path") or "")
            if task_path:
                self._select_path_combo_value(self.home_task_combo, task_path)
                self.task_path_edit.setText(task_path)
            config_id = str(demo.get("world_model_config_id") or services.DEFAULT_WORLD_MODEL_CONFIG_ID)
            self._select_world_model_config(config_id)
            agent = str(demo.get("evaluation_agent") or agent)
            planner = str(demo.get("planner") or planner)
        self._select_combo_value(self.backend_combo, "beamng")
        self._select_combo_value(self.agent_combo, agent)
        self._select_combo_value(self.planner_combo, planner)
        self._refresh_demo_status()
        self.run_home_region_model_test()

    def run_home_start(self) -> None:
        backend = self.backend_combo.currentData() or self.backend_combo.currentText()
        if str(backend) == "beamng":
            self.run_home_region_model_test()
            return
        row = self._combo_config_row(self.world_model_config_combo)
        if row:
            self._apply_world_model_config(row, sync_editor=True)
        self.run_episode()

    def run_home_region_model_test(self) -> None:
        task_path = self._path_combo_value(self.home_task_combo).strip()
        config = self._combo_config_row(self.world_model_config_combo)
        if config:
            self._apply_world_model_config(config, sync_editor=True)
        algorithm = str(config.get("algorithm") or self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "stablewm_lewm")
        world_model = str(config.get("world_model") or self.world_model_combo.currentData() or self.world_model_combo.currentText() or "le_wm")
        model_path = str(config.get("model_path") or self._path_combo_value(self.home_model_combo)).strip()
        validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
        use_experience_corridor = bool(validation.get("experience_corridor"))
        use_model_support_subgoals = bool(validation.get("model_support_subgoals"))
        use_model_support_field_subgoals = bool(validation.get("model_support_field_subgoals"))
        use_model_support_graph_subgoals = bool(validation.get("model_support_graph_subgoals"))
        if not task_path:
            self.log("开始测试需要先选择 BeamNG region task。")
            return
        if algorithm == "world_model_direct" or world_model == "tiny_learned":
            if not model_path:
                self.log("direct world-model evaluation requires a model path.")
                return
            self.task_path_edit.setText(task_path)
            self.model_path_edit.setText(model_path)
            self._select_combo_value(self.agent_combo, "world_model_direct")
            self._select_combo_value(self.world_model_combo, world_model)
            self._select_combo_value(self.planner_combo, "navigation_mpc")
            request = services.RegionWorldModelEvaluationRequest(
                task_path=task_path,
                world_model_type=world_model,
                world_model_path=model_path,
                eval_steps=max(int(self.settings.max_steps), 1200),
                seed=self.settings.seed,
                planner="navigation_mpc",
                planner_horizon=self.settings.planner_horizon,
                planner_samples=self.settings.planner_samples,
                planner_iterations=self.settings.planner_iterations,
                planner_goal_weight=_validation_float(validation, "planner_goal_weight"),
                planner_progress_weight=_validation_float(validation, "planner_progress_weight"),
                planner_risk_weight=_validation_float(validation, "planner_risk_weight"),
                planner_heading_weight=_validation_float(validation, "planner_heading_weight"),
                include_route_guided_baseline=True,
                evaluation_allow_reverse_recovery=bool(validation.get("evaluation_allow_reverse_recovery")),
                evaluation_reverse_recovery_after_steps=_validation_int(validation, "evaluation_reverse_recovery_after_steps", 96),
                evaluation_local_subgoal_distance_m=_validation_float(validation, "evaluation_local_subgoal_distance_m", 12.0) or 12.0,
                use_experience_corridor=use_experience_corridor,
                evaluation_use_model_support_subgoals=use_model_support_subgoals,
                evaluation_use_model_support_field_subgoals=use_model_support_field_subgoals,
                evaluation_use_model_support_graph_subgoals=use_model_support_graph_subgoals,
                close_beamng=False,
                step_delay_sec=0.02,
                post_run_hold_sec=20.0,
            )
            self.log(f"direct world-model evaluation: task={task_path}, world_model={world_model}, model={model_path}")
            self._run_task(
                lambda: services.run_region_world_model_evaluation(request),
                self._pipeline_finished,
                "home direct world model test failed",
                task_label="direct world-model evaluation",
            )
            return
        if algorithm == "stablewm_lewm" and not model_path:
            self.log("开始测试需要先选择模型 checkpoint。")
            return
        self.task_path_edit.setText(task_path)
        self.model_path_edit.setText(model_path)
        self._select_combo_value(self.agent_combo, "model_mpc")
        self._select_combo_value(self.algorithm_combo, algorithm)
        self._select_combo_value(self.world_model_combo, world_model)
        self._select_combo_value(self.planner_combo, "navigation_mpc")
        request = services.RegionNavigationClosedLoopRequest(
            task_path=task_path,
            algorithm=algorithm,
            algorithm_model_path=model_path if algorithm == "stablewm_lewm" else "",
            collect_steps=max(int(self.settings.max_steps), 1000),
            eval_steps=max(int(self.settings.max_steps), 1200),
            seed=self.settings.seed,
            planner="navigation_mpc",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            evaluation_agent="model_mpc",
            close_beamng=False,
            step_delay_sec=0.02,
            post_run_hold_sec=20.0,
        )
        self.log(f"开始测试：task={task_path}, model={model_path}")
        self._run_task(
            lambda: services.run_region_navigation_closed_loop(request),
            self._pipeline_finished,
            "home region model test failed",
            task_label="开始测试",
        )

    def run_episode(self) -> None:
        request = self._current_request()
        self.log(f"开始运行：backend={request.backend}, agent={request.agent}, planner={request.planner or 'none'}")
        self._run_task(lambda: services.run_episode_from_request(request), self._episode_finished, "episode failed", task_label="运行 episode")

    def inspect_dataset(self) -> None:
        self.log("检查数据集...")
        self._run_task(
            lambda: services.inspect_dataset(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
            ),
            self._dataset_inspected,
            "数据集检查失败",
            task_label="检查数据集",
        )

    def _dataset_root_changed(self, _text: str) -> None:
        self.dataset_info = None
        if hasattr(self, "dataset_preview_timer"):
            self.dataset_preview_timer.stop()
        if hasattr(self, "dataset_play_button"):
            self.dataset_play_button.setChecked(False)
        self.sequence_combo.clear()
        self.adapter_edit.clear()
        if hasattr(self, "dataset_save_button"):
            self._set_custom_dataset_controls(True)
            self._set_dataset_sequence_rows([])

    def preview_dataset(self) -> None:
        if self._dataset_preview_busy:
            return
        self._dataset_preview_busy = True
        frame_index = self.dataset_frame_slider.value() if hasattr(self, "dataset_frame_slider") else self.settings.preview_frame_index
        self.settings.preview_frame_index = frame_index
        self.log(f"生成数据集帧预览：{frame_index}")
        self._run_task(
            lambda: self.dataset_preview_session.preview(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                frame_index=frame_index,
            ),
            self._preview_ready,
            "数据集预览失败",
            task_label="预览数据集",
        )

    def run_dataset_quality_analysis(self) -> None:
        self.log("开始数据集质量检查...")
        self._run_task(
            lambda: services.analyze_dataset_quality(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
            ),
            self._dataset_quality_ready,
            "数据集质量检查失败",
            task_label="生成数据质量报告",
        )

    def create_dataset_split(self) -> None:
        ratios = (
            self.dataset_train_ratio.value(),
            self.dataset_validation_ratio.value(),
            self.dataset_test_ratio.value(),
        )
        self.log(f"生成数据划分：train={ratios[0]:.2f}, validation={ratios[1]:.2f}, test={ratios[2]:.2f}")
        self._run_task(
            lambda: services.create_dataset_split_definition(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                train_ratio=ratios[0],
                validation_ratio=ratios[1],
                test_ratio=ratios[2],
                seed=self.settings.seed,
                output_path=self.dataset_split_path_edit.text().strip() or None,
            ),
            self._dataset_split_ready,
            "数据划分失败",
            task_label="生成数据划分",
        )

    def _toggle_dataset_playback(self, active: bool) -> None:
        self.dataset_play_button.setText("暂停" if active else "播放")
        if active:
            self.dataset_preview_timer.start()
            self.preview_dataset()
        else:
            self.dataset_preview_timer.stop()

    def _step_dataset_preview(self, offset: int) -> None:
        maximum = self.dataset_frame_slider.maximum()
        value = min(max(0, self.dataset_frame_slider.value() + int(offset)), maximum)
        self.dataset_frame_slider.setValue(value)
        self.preview_dataset()

    def _advance_dataset_preview(self) -> None:
        if self._dataset_preview_busy:
            return
        if self.dataset_frame_slider.value() >= self.dataset_frame_slider.maximum():
            self.dataset_play_button.setChecked(False)
            return
        self.dataset_frame_slider.setValue(self.dataset_frame_slider.value() + 1)
        self.preview_dataset()

    def run_training_preset(self) -> None:
        data = self.training_preset_combo.currentData()
        preset = dict(data) if isinstance(data, dict) else {}
        preset_id = str(preset.get("id") or "")
        if not preset_id:
            self.model_summary.setText(_compact_json({"status": services.NAN_TEXT, "message": "No training preset selected."}))
            return
        if preset.get("available") is False:
            payload = {"status": preset.get("status", services.UNFINISHED_TEXT), "training_preset": preset}
            self.model_summary.setText(_compact_json(payload))
            self.log(f"Training preset is not available yet: {preset.get('label', preset_id)}")
            return
        try:
            row = self._current_training_config_row()
        except ValueError as exc:
            self.model_summary.setText(_compact_json({"status": "invalid_parameters", "message": str(exc)}))
            self.log(f"Training parameters are invalid: {exc}")
            return
        manifest_path = str(preset.get("manifest_path") or "").strip()
        trainer_root = str(Path(manifest_path).parent) if manifest_path else None
        self.log(f"开始训练配置：{row.get('label', row.get('id', services.NAN_TEXT))}")
        if manifest_path:
            try:
                job = services.queue_training_config_job(
                    self.training_job_queue,
                    row,
                    trainer_root=trainer_root,
                )
            except Exception as exc:
                self.model_summary.setText(_compact_json({"status": "queue_failed", "message": str(exc)}))
                self.log(f"训练任务入队失败: {exc}")
                return
            self._current_training_job_id = job.job_id
            self._handled_training_job_ids.discard(job.job_id)
            self.training_job_timer.start()
            self.model_summary.setText(_compact_json(job.snapshot()))
            self.log(f"训练任务已入队: {job.job_id}")
            self._refresh_training_jobs()
            return
        self._run_task(
            lambda: services.run_training_config_job(row, trainer_root=trainer_root),
            self._training_config_finished,
            "训练配置运行失败",
            task_label=f"训练 {preset.get('label', preset_id)}",
        )

    def validate_training_config(self) -> None:
        try:
            row = self._current_training_config_row()
            report = services.validate_training_config_setup(row)
        except Exception as exc:
            report = {"ready": False, "status": "invalid", "issues": [str(exc)]}
        self.model_summary.setText(_compact_json(report))
        self._set_training_run_views(report)
        status = report.get("status", services.NAN_TEXT) if isinstance(report, dict) else services.NAN_TEXT
        self.log(f"Training config validation: {status}")

    def import_training_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import training config",
            str(services.CONFIG_ROOT),
            "Training config (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        try:
            row = services.import_training_config(path)
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "import_failed", "message": str(exc)}))
            self.log(f"Training config import failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "imported", "training_config": row}))
        self.log(f"Training config imported: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_training_config(str(row.get("id") or ""))

    def import_dataset_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import dataset manifest",
            str(services.CONFIG_ROOT / "datasets"),
            "Dataset manifest (dataset_manifest.yaml dataset_manifest.yml);;YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            row = services.import_dataset_manifest(path)
        except Exception as exc:
            self.dataset_summary.setText(_compact_json({"status": "import_failed", "message": str(exc)}))
            self.log(f"Dataset manifest import failed: {exc}")
            return
        self.dataset_summary.setText(_compact_json({"status": "imported", "dataset": row}))
        self.log(f"Dataset manifest imported: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_dataset_manifest(str(row.get("id") or ""))

    def save_dataset_manifest_from_gui(self) -> None:
        dataset_root = self.dataset_root_edit.text().strip()
        dataset_id = self.dataset_manifest_name_edit.text().strip() or Path(dataset_root).name or "Custom Dataset"
        if not dataset_root:
            self.dataset_summary.setText(_compact_json({"status": "invalid_dataset", "message": "Dataset root is required."}))
            return
        try:
            sequences = self._dataset_sequences_from_text()
            row = services.save_dataset_manifest(
                dataset_id=dataset_id,
                display_name=dataset_id,
                dataset_root=dataset_root,
                sequences=sequences,
            )
        except Exception as exc:
            self.dataset_summary.setText(_compact_json({"status": "save_failed", "message": str(exc)}))
            self.log(f"Dataset manifest save failed: {exc}")
            return
        self.dataset_summary.setText(_compact_json({"status": "saved", "dataset": row}))
        self.log(f"Dataset manifest saved: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_dataset_manifest(str(row.get("id") or ""))

    def suggest_dataset_manifest_sequences_from_gui(self) -> None:
        dataset_root = self.dataset_root_edit.text().strip()
        if not dataset_root:
            self.dataset_summary.setText(_compact_json({"status": "invalid_dataset", "message": "Dataset root is required."}))
            return
        try:
            result = services.detect_dataset_sequences(dataset_root, self.adapter_edit.text().strip())
        except Exception as exc:
            self.dataset_summary.setText(_compact_json({"status": "detection_failed", "message": str(exc)}))
            self.log(f"Dataset format detection failed: {exc}")
            return
        definitions = result.get("sequence_definitions") if isinstance(result.get("sequence_definitions"), list) else []
        sequence_ids = [str(item) for item in result.get("sequences", [])] if isinstance(result.get("sequences"), list) else []
        if not self.dataset_manifest_name_edit.text().strip():
            self.dataset_manifest_name_edit.setText(Path(dataset_root).name or "Custom Dataset")
        adapter = str(result.get("adapter") or "")
        self.adapter_edit.setText(adapter)
        self.sequence_combo.clear()
        for sequence_id in sequence_ids:
            self.sequence_combo.addItem(sequence_id)
        if adapter == "manifest_dataset":
            self._set_custom_dataset_controls(True)
            self._set_dataset_sequence_rows([dict(row) for row in definitions if isinstance(row, dict)])
        else:
            self._set_detected_dataset_sequence_rows(sequence_ids, adapter)
        self.dataset_summary.setText(_compact_json(result))
        self.log(
            f"Dataset detected: adapter={result.get('adapter', services.NAN_TEXT)}, "
            f"sequences={len(sequence_ids)}"
        )

    def import_trainer_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import trainer manifest",
            str(services.CONFIG_ROOT / "trainers"),
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            row = services.import_trainer_manifest(path)
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "import_failed", "message": str(exc)}))
            self.log(f"Trainer manifest import failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "imported", "trainer": row}))
        self.log(f"Trainer manifest imported: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_training_preset(str(row.get("id") or ""))

    def save_trainer_manifest_from_gui(self) -> None:
        entrypoint = self.trainer_entrypoint_edit.text().strip()
        if not entrypoint:
            self.model_summary.setText(_compact_json({"status": "invalid_trainer", "message": "Trainer entrypoint is required."}))
            return
        try:
            arguments = self._trainer_arguments_from_text()
            schema = self._trainer_schema_from_text()
            row = services.save_trainer_manifest(
                trainer_id=Path(entrypoint).stem,
                label=Path(entrypoint).stem.replace("_", " ").title(),
                entrypoint=entrypoint,
                runtime="python",
                arguments=arguments,
                parameters=schema,
                input_spec={"dataset_format": "any_registered_adapter"},
                outputs={"artifact_type": "checkpoint"},
            )
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "save_failed", "message": str(exc)}))
            self.log(f"Trainer manifest save failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "saved", "trainer": row}))
        self.log(f"Trainer manifest saved: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_training_preset(str(row.get("id") or ""))

    def save_script_training_config_from_gui(self) -> None:
        try:
            bundle = self._save_script_training_config_bundle()
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "save_failed", "message": str(exc)}))
            self.log(f"Script training config save failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "saved", **bundle}))
        self.log(f"Script training config saved: {bundle['training_config'].get('label', services.NAN_TEXT)}")
        self.refresh_catalogs()
        self._select_training_config(str(bundle["training_config"].get("id") or ""))
        self._select_training_preset(str(bundle["trainer"].get("id") or ""))

    def run_script_training_config(self) -> None:
        try:
            bundle = self._save_script_training_config_bundle()
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "save_failed", "message": str(exc)}))
            self.log(f"Script training config save failed: {exc}")
            return
        trainer = bundle["trainer"]
        config = bundle["training_config"]
        manifest_path = str(trainer.get("manifest_path") or "").strip()
        trainer_root = str(Path(manifest_path).parent.resolve()) if manifest_path else None
        self.model_summary.setText(_compact_json({"status": "queued", **bundle}))
        self.log(f"Starting script training config: {config.get('label', config.get('id', services.NAN_TEXT))}")
        try:
            job = services.queue_training_config_job(
                self.training_job_queue,
                config,
                trainer_root=trainer_root,
            )
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "queue_failed", "message": str(exc)}))
            self.log(f"Script training config queue failed: {exc}")
            return
        self._current_training_job_id = job.job_id
        self._handled_training_job_ids.discard(job.job_id)
        self.training_job_timer.start()
        self._refresh_training_jobs()

    def _save_script_training_config_bundle(self) -> dict[str, Any]:
        entrypoint = self.trainer_entrypoint_edit.text().strip()
        if not entrypoint:
            raise ValueError("Trainer entrypoint is required.")
        label = self.training_config_name_edit.text().strip() or Path(entrypoint).stem.replace("_", " ").title()
        parameters = self._trainer_parameters_from_text()
        schema = self._trainer_schema_from_text()
        arguments = self._trainer_arguments_for_script_config()
        return services.save_script_training_config(
            label=label,
            trainer_entrypoint=entrypoint,
            dataset_root=self.dataset_root_edit.text().strip(),
            adapter=self.adapter_edit.text().strip(),
            sequence_id=self.sequence_combo.currentText().strip(),
            split_path=self.dataset_split_path_edit.text().strip(),
            output_path=self.training_output_edit.text().strip(),
            parameters=parameters,
            parameter_schema=schema or None,
            arguments=arguments,
        )

    def import_world_model_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import model/checkpoint",
            self.model_path_edit.text().strip() or str(services.ROOT),
            "World model files (*.ckpt *.json *.npz);;All files (*)",
        )
        if not path:
            return
        self._import_world_model_path(path)

    def import_world_model_directory_config(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Import model folder",
            self.model_path_edit.text().strip() or str(services.ROOT),
        )
        if not path:
            return
        self._import_world_model_path(path)

    def _import_world_model_path(self, path: str) -> None:
        try:
            row = services.import_world_model_config(path)
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "import_failed", "message": str(exc)}))
            self.log(f"World model import failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "imported", "world_model_config": row}))
        self.log(f"World model imported: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_world_model_config(str(row.get("id") or ""))

    def run_manifest_trainer(self, preset: dict[str, Any]) -> None:
        try:
            parameters = self._trainer_parameters_from_text()
        except ValueError as exc:
            payload = {"status": "invalid_parameters", "message": str(exc)}
            self.model_summary.setText(_compact_json(payload))
            self.log(f"Training parameters are invalid: {exc}")
            return
        output = self._current_training_output_path(str(preset.get("id") or ""))
        self.log(f"Starting trainer manifest: {preset.get('label', preset.get('id', services.NAN_TEXT))}")
        self._run_task(
            lambda: services.run_trainer_manifest_job(
                str(preset["manifest_path"]),
                dataset_root=self.dataset_root_edit.text().strip(),
                output_dir=output,
                parameters=parameters,
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
            ),
            self._training_finished,
            "trainer manifest failed",
            task_label=f"Train {preset.get('label', preset.get('id', 'model'))}",
        )

    def save_world_model_config(self) -> None:
        label = self.model_config_name_edit.text().strip()
        selected = self._combo_config_row(self.world_model_config_edit_combo)
        if not label:
            label = str(selected.get("label") or selected.get("id") or "World Model Config")
        model_path = self._path_combo_value(self.home_model_combo).strip() or self.model_path_edit.text().strip()
        try:
            row = services.save_world_model_config(
                config_id=label,
                label=label,
                algorithm=self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "stablewm_lewm",
                world_model=self.world_model_combo.currentData() or self.world_model_combo.currentText() or "le_wm",
                model_path=model_path,
            )
        except ValueError as exc:
            self.log(f"World model config save failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "saved", "world_model_config": row}))
        self.log(f"World model config saved: {row['label']}")
        self.refresh_catalogs()
        self._select_world_model_config(str(row["id"]))

    def register_latest_training_artifact_model(self) -> None:
        run = getattr(self, "latest_training_run_record", {})
        path = str(run.get("path") or run.get("training_run_path") or "").strip()
        if not path:
            payload = {"status": "register_failed", "message": "No completed training run is selected."}
            self.model_summary.setText(_compact_json(payload))
            self.log("World model config registration failed: no selected training run.")
            return
        label = self.model_config_name_edit.text().strip()
        if not label:
            label = str(run.get("preset_label") or run.get("run_id") or "Training Model")
        try:
            row = services.register_training_run_artifact_as_world_model_config(path, label=label)
        except Exception as exc:
            payload = {"status": "register_failed", "message": str(exc)}
            self.model_summary.setText(_compact_json(payload))
            self.log(f"World model config registration failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "registered", "world_model_config": row}))
        self.log(f"World model config registered: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self.refresh_catalogs()
        self._select_world_model_config(str(row.get("id") or ""))

    def save_training_config(self) -> None:
        label = self.training_config_name_edit.text().strip()
        selected = self.training_config_combo.currentData()
        if not label and isinstance(selected, dict):
            label = str(selected.get("label") or selected.get("id") or "Training Config")
        if not label:
            label = "Training Config"
        preset = self.training_preset_combo.currentData()
        preset_id = str(preset.get("id") if isinstance(preset, dict) else self.training_preset_combo.currentText())
        try:
            parameters = self._trainer_parameters_from_text()
        except ValueError as exc:
            self.model_summary.setText(_compact_json({"status": "invalid_parameters", "message": str(exc)}))
            self.log(f"Training config save failed: {exc}")
            return
        output_path = self._current_training_output_path(preset_id)
        try:
            row = services.save_training_config(
                config_id=label,
                label=label,
                training_preset_id=preset_id,
                dataset_root=self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                split_path=self.dataset_split_path_edit.text().strip(),
                output_path=output_path,
                parameters=parameters,
            )
        except Exception as exc:
            self.model_summary.setText(_compact_json({"status": "save_failed", "message": str(exc)}))
            self.log(f"Training config save failed: {exc}")
            return
        self.model_summary.setText(_compact_json({"status": "saved", "training_config": row}))
        self.log(f"Training config saved: {row['label']}")
        self.refresh_catalogs()
        self._select_training_config(str(row["id"]))

    def _current_training_config_row(self) -> dict[str, Any]:
        label = self.training_config_name_edit.text().strip() or "Manual training config"
        preset = self.training_preset_combo.currentData()
        preset_id = str(preset.get("id") if isinstance(preset, dict) else self.training_preset_combo.currentText())
        return {
            "id": label,
            "label": label,
            "training_preset_id": preset_id,
            "dataset_root": self.dataset_root_edit.text().strip(),
            "adapter": self.adapter_edit.text().strip(),
            "sequence_id": self.sequence_combo.currentText().strip(),
            "split_path": self.dataset_split_path_edit.text().strip(),
            "output_path": self._current_training_output_path(preset_id),
            "parameters": self._trainer_parameters_from_text(),
        }

    def train_tiny_model(self) -> None:
        root = self.dataset_root_edit.text().strip()
        if not root:
            self.log("训练 tiny world model 需要 dataset root")
            return
        try:
            parameters = self._trainer_parameters_from_text()
            ridge = float(parameters.get("ridge", 1e-4))
        except (TypeError, ValueError) as exc:
            self.model_summary.setText(_compact_json({"status": "invalid_parameters", "message": str(exc)}))
            self.log(f"Training parameters are invalid: {exc}")
            return
        output = self._current_training_output_path("tiny_world_model") or str(services.ROOT / "outputs" / "models" / "gui_tiny_world_model")
        self.log(f"训练 tiny world model -> {output}")
        self._run_task(
            lambda: services.train_tiny_world_model(
                root,
                output,
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                ridge=ridge,
            ),
            self._training_finished,
            "training failed",
            task_label="训练 tiny world model",
        )

    def export_stablewm_hdf5(self) -> None:
        root = self.dataset_root_edit.text().strip()
        try:
            parameters = self._trainer_parameters_from_text()
            image_size = int(parameters.get("image_size", self.settings.image_size))
        except (TypeError, ValueError) as exc:
            self.dataset_summary.setText(_compact_json({"status": "invalid_parameters", "message": str(exc)}))
            self.log(f"StableWM export parameters are invalid: {exc}")
            return
        output = self._current_training_output_path("stablewm_hdf5") or str(services.ROOT / "outputs" / "stablewm" / "gui_export.h5")
        self.stablewm_hdf5_edit.setText(output)
        self.log(f"导出 StableWM HDF5 -> {output}")
        self._run_task(
            lambda: services.export_lewm_hdf5(
                root,
                output,
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                image_size=image_size,
            ),
            self._hdf5_exported,
            "stablewm export failed",
            task_label="导出 StableWM HDF5",
        )

    def train_lewm_cost_model(self) -> None:
        try:
            parameters = self._trainer_parameters_from_text()
        except ValueError as exc:
            self.model_summary.setText(_compact_json({"status": "invalid_parameters", "message": str(exc)}))
            self.log(f"LE-WM cost parameters are invalid: {exc}")
            return
        hdf5_path = str(parameters.get("input_hdf5") or self.stablewm_hdf5_edit.text().strip())
        output = self._current_training_output_path("lewm_cost_model") or str(services.ROOT / "outputs" / "models" / "gui_lewm_cost")
        self.log(f"训练 LE-WM cost model -> {output}")
        self._run_task(
            lambda: services.train_lewm_cost_model(hdf5_path, output),
            self._training_finished,
            "lewm training failed",
            task_label="训练 LE-WM cost model",
        )

    def run_orfd_lewm_pipeline(self) -> None:
        request = services.PipelineRequest(
            dataset_root=self.dataset_root_edit.text().strip(),
            adapter=self.adapter_edit.text().strip() or "orfd",
            sequence_id=self.sequence_combo.currentText().strip(),
            hdf5_path=self.stablewm_hdf5_edit.text().strip(),
            model_dir=self.model_path_edit.text().strip(),
            image_size=self.settings.image_size,
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            max_steps=self.settings.max_steps,
            seed=self.settings.seed,
            run_beamng=False,
            beamng_scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText(),
        )
        self.log("启动数据集训练流程：ORFD -> HDF5 -> LE-WM cost -> dataset replay")
        self._run_task(lambda: services.run_orfd_lewm_pipeline(request), self._pipeline_finished, "pipeline failed", task_label="运行数据集训练流程")

    def export_beamng_terrain_draft(self) -> None:
        self.log("导出 ORFD 局部 BeamNG 地形草案...")
        self._run_task(
            lambda: services.export_orfd_beamng_terrain_draft(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                frame_index=self.settings.terrain_frame_index,
                grid_size=self.settings.terrain_grid_size,
                terrain_size_m=float(self.settings.terrain_size_m),
            ),
            self._terrain_exported,
            "terrain export failed",
            task_label="导出 BeamNG 地形草案",
        )

    def check_beamng(self) -> None:
        status = services.beamng_status()
        self.beamng_summary.setText(_compact_json(status))
        self.log(f"BeamNG: {status.get('message', services.NAN_TEXT)}")
        self.refresh_catalogs()

    def run_visible_beamng_demo(self) -> None:
        request = services.VisibleBeamNGDemoRequest(
            dataset_root=self.dataset_root_edit.text().strip(),
            adapter=self.adapter_edit.text().strip() or "orfd",
            sequence_id=self.sequence_combo.currentText().strip(),
            world_model_type=self.world_model_combo.currentData() or self.world_model_combo.currentText() or "simple_kinematic",
            world_model_path=self.model_path_edit.text().strip(),
            planner=self.planner_combo.currentData() or self.planner_combo.currentText() or "",
            scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText() or "beamng_visible_autodrive",
            max_steps=max(int(self.settings.max_steps), 600),
            seed=self.settings.seed,
            record=self.settings.record,
        )
        self.log("启动 BeamNG 可视自动驾驶...")
        self._run_task(lambda: services.run_visible_beamng_demo(request), self._visible_demo_finished, "visible BeamNG demo failed", task_label="运行 BeamNG 演示")

    def run_beamng_lewm_closed_loop(self) -> None:
        request = services.BeamNGMapLeWMClosedLoopRequest(
            algorithm=self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "local_lewm_cost",
            scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText() or "beamng_visible_autodrive",
            collect_steps=max(int(self.settings.max_steps), 160),
            eval_steps=max(int(self.settings.max_steps), 80),
            seed=self.settings.seed,
            planner=self.planner_combo.currentData() or self.planner_combo.currentText() or "le_wm_cem",
            close_beamng=False,
        )
        self.log("BeamNG LE-WM 闭环：采集 -> HDF5 -> 训练 -> 评估")
        self._run_task(
            lambda: services.run_beamng_map_lewm_closed_loop(request),
            self._pipeline_finished,
            "beamng lewm closed loop failed",
            task_label="运行 BeamNG LE-WM 闭环",
        )

    def run_region_navigation_loop(self) -> None:
        config = self._combo_config_row(self.beamng_model_config_combo)
        if config:
            self._apply_world_model_config(config, sync_editor=True)
        task_path = self.task_path_edit.text().strip() or str(services.DEFAULT_NAVIGATION_TASK_PATH)
        algorithm = str(config.get("algorithm") or self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "local_lewm_cost")
        world_model = str(config.get("world_model") or self.world_model_combo.currentData() or self.world_model_combo.currentText() or "le_wm")
        model_path = str(config.get("model_path") or self.model_path_edit.text().strip()).strip()
        validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
        use_experience_corridor = bool(validation.get("experience_corridor"))
        use_model_support_subgoals = bool(validation.get("model_support_subgoals"))
        use_model_support_field_subgoals = bool(validation.get("model_support_field_subgoals"))
        use_model_support_graph_subgoals = bool(validation.get("model_support_graph_subgoals"))
        if algorithm == "world_model_direct" or world_model == "tiny_learned":
            if not model_path:
                self.log("direct world-model evaluation requires a model path.")
                return
            self.task_path_edit.setText(task_path)
            self.model_path_edit.setText(model_path)
            self._select_combo_value(self.agent_combo, "world_model_direct")
            self._select_combo_value(self.world_model_combo, world_model)
            self._select_combo_value(self.planner_combo, "navigation_mpc")
            request = services.RegionWorldModelEvaluationRequest(
                task_path=task_path,
                world_model_type=world_model,
                world_model_path=model_path,
                eval_steps=max(int(self.settings.max_steps), 1200),
                seed=self.settings.seed,
                planner="navigation_mpc",
                planner_horizon=self.settings.planner_horizon,
                planner_samples=self.settings.planner_samples,
                planner_iterations=self.settings.planner_iterations,
                planner_goal_weight=_validation_float(validation, "planner_goal_weight"),
                planner_progress_weight=_validation_float(validation, "planner_progress_weight"),
                planner_risk_weight=_validation_float(validation, "planner_risk_weight"),
                planner_heading_weight=_validation_float(validation, "planner_heading_weight"),
                include_route_guided_baseline=True,
                evaluation_allow_reverse_recovery=bool(validation.get("evaluation_allow_reverse_recovery")),
                evaluation_reverse_recovery_after_steps=_validation_int(validation, "evaluation_reverse_recovery_after_steps", 96),
                evaluation_local_subgoal_distance_m=_validation_float(validation, "evaluation_local_subgoal_distance_m", 12.0) or 12.0,
                use_experience_corridor=use_experience_corridor,
                evaluation_use_model_support_subgoals=use_model_support_subgoals,
                evaluation_use_model_support_field_subgoals=use_model_support_field_subgoals,
                evaluation_use_model_support_graph_subgoals=use_model_support_graph_subgoals,
                close_beamng=False,
                step_delay_sec=0.02,
                post_run_hold_sec=20.0,
            )
            self.log("区域导航评估：按当前 direct world model 配置运行 BeamNG start/goal 控车")
            self._run_task(
                lambda: services.run_region_world_model_evaluation(request),
                self._pipeline_finished,
                "direct world-model evaluation failed",
                task_label="direct world-model evaluation",
            )
            return
        request = services.RegionNavigationClosedLoopRequest(
            task_path=task_path,
            algorithm=algorithm,
            algorithm_model_path=model_path if algorithm == "stablewm_lewm" else "",
            collect_steps=max(int(self.settings.max_steps), 1000),
            eval_steps=max(int(self.settings.max_steps), 1200),
            seed=self.settings.seed,
            planner=self.planner_combo.currentData() or self.planner_combo.currentText() or "navigation_mpc",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            evaluation_agent="model_mpc",
            close_beamng=False,
            step_delay_sec=0.02,
            post_run_hold_sec=20.0,
        )
        self.log("区域导航闭环：按当前 task/model/planner 运行 BeamNG start/goal 评估")
        self._run_task(
            lambda: services.run_region_navigation_closed_loop(request),
            self._pipeline_finished,
            "region navigation loop failed",
            task_label="运行区域导航",
        )

    def collect_region_training_data(self) -> None:
        task_path = self.task_path_edit.text().strip() or self._path_combo_value(self.beamng_task_combo).strip()
        if not task_path:
            self.log("采集训练数据需要先选择 BeamNG region task。")
            return
        request = services.RegionTrainingDataCollectionRequest(
            task_path=task_path,
            collect_steps=max(int(self.settings.max_steps), 1500),
            collect_rollouts=6,
            min_collection_goal_progress_ratio=0.35,
            collection_strategy="route_aware",
            collection_route_target_interval=1,
            collection_route_lateral_m=2.5,
            collection_multi_start=True,
            collection_multi_start_lateral_m=1.5,
            min_route_coverage_ratio=0.5,
            min_goal_zone_coverage=0.2,
            max_collection_min_goal_distance_m=70.0,
            min_unique_region_cells=3,
            collection_coverage_grid_size=6,
            collection_coverage_target_interval=1,
            collection_max_target_steps=30,
            seed=self.settings.seed,
            close_beamng=False,
            step_delay_sec=0.02,
            post_run_hold_sec=2.0,
        )
        self.log("开始采集 BeamNG 区域训练数据：region_explorer -> recorded episodes")
        self._run_task(
            lambda: services.collect_region_training_data(request),
            self._region_training_data_collected,
            "region training data collection failed",
            task_label="采集 BeamNG 训练数据",
        )

    def train_region_world_model_from_collection(self) -> None:
        manifest_path = self.region_collection_manifest_edit.text().strip()
        if not manifest_path:
            self.log("训练模型需要先采集训练数据，或手动填写 collection manifest。")
            return
        request = services.RegionWorldModelTrainingRequest(
            collection_manifest_path=manifest_path,
            world_model_type=self.beamng_training_model_type_combo.currentData()
            or self.beamng_training_model_type_combo.currentText()
            or "mlp_dynamics",
            register_world_model_config=True,
        )
        self.log("开始从 BeamNG collection manifest 训练 region world model")
        self._run_task(
            lambda: services.train_region_world_model_from_collection(request),
            self._region_world_model_training_finished,
            "region world model training failed",
            task_label="训练 BeamNG world model",
        )

    def train_region_self_supervised_world_model(self) -> None:
        task_path = self.task_path_edit.text().strip() or self._path_combo_value(self.beamng_task_combo).strip()
        if not task_path:
            self.log("区域自监督训练需要先选择 BeamNG region task。")
            return
        request = services.RegionSelfSupervisedWorldModelRequest(
            task_path=task_path,
            world_model_type=self.beamng_training_model_type_combo.currentData()
            or self.beamng_training_model_type_combo.currentText()
            or "mlp_dynamics",
            collect_steps=max(int(self.settings.max_steps), 1500),
            collect_rollouts=6,
            min_collection_goal_progress_ratio=0.35,
            collection_strategy="route_aware",
            collection_route_target_interval=1,
            collection_route_lateral_m=2.5,
            collection_multi_start=True,
            collection_multi_start_lateral_m=1.5,
            min_route_coverage_ratio=0.5,
            min_goal_zone_coverage=0.2,
            max_collection_min_goal_distance_m=70.0,
            min_unique_region_cells=3,
            collection_coverage_grid_size=6,
            collection_coverage_target_interval=1,
            collection_max_target_steps=30,
            eval_steps=max(int(self.settings.max_steps), 1200),
            seed=self.settings.seed,
            planner=self.planner_combo.currentData() or self.planner_combo.currentText() or "navigation_mpc",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            evaluation_agent="world_model_direct",
            evaluation_route_mode="route_free",
            use_experience_corridor=False,
            evaluation_use_model_support_subgoals=True,
            evaluation_use_model_support_field_subgoals=False,
            evaluation_use_model_support_graph_subgoals=True,
            close_beamng=False,
            step_delay_sec=0.02,
            post_run_hold_sec=20.0,
            register_world_model_config=True,
        )
        self.log("区域自监督训练：探索采集 -> tiny world model -> 无路线模型控车评估")
        self._run_task(
            lambda: services.run_region_self_supervised_world_model(request),
            self._pipeline_finished,
            "region self-supervised world model failed",
            task_label="区域自监督训练 world model",
        )

    def load_selected_episode(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        trace = services.load_episode_trace(path)
        self.trajectory.set_trace(trace)
        self.log(f"已加载 episode: {item.text()}")

    def _episode_finished(self, payload: dict[str, Any]) -> None:
        self.log(f"episode 完成: {payload.get('episode_id', services.NAN_TEXT)}")
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        self._update_metrics(metrics)
        path = payload.get("episode_path")
        self.trajectory.set_trace(services.load_episode_trace(path) if path else [])
        self.refresh_catalogs()

    def _pipeline_finished(self, payload: dict[str, Any]) -> None:
        self.stablewm_hdf5_edit.setText(str(payload.get("hdf5_path", "")))
        self.model_path_edit.setText(str(payload.get("model_dir", "")))
        if payload.get("model_dir"):
            self.home_model_combo.setCurrentText(str(payload.get("model_dir", "")))
        saved_config_id = self._register_pipeline_world_model_config(payload)
        replay = payload.get("dataset_replay") if isinstance(payload.get("dataset_replay"), dict) else {}
        beamng = payload.get("beamng") if isinstance(payload.get("beamng"), dict) else None
        evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else None
        metrics_source = beamng or evaluation or replay
        if isinstance(metrics_source, dict):
            metrics = metrics_source.get("metrics", {}) if isinstance(metrics_source.get("metrics"), dict) else {}
            self._update_metrics(metrics)
            path = metrics_source.get("episode_path")
            self.trajectory.set_trace(services.load_episode_trace(path) if path else [])
        self.model_summary.setText(_compact_json(payload))
        self._set_training_run_views(payload)
        self._set_beamng_quality_views(payload)
        if hasattr(self, "demo_result_summary"):
            self.demo_result_summary.setText(_compact_json(_demo_result_payload(payload)))
        region_summary = _region_world_model_summary_text(payload)
        if region_summary and hasattr(self, "beamng_summary"):
            self.beamng_summary.setText(region_summary)
        self.log("流程完成")
        self.refresh_catalogs()
        if saved_config_id:
            self._select_world_model_config(saved_config_id)

    def _region_training_data_collected(self, payload: dict[str, Any]) -> None:
        manifest_path = str(payload.get("collection_manifest_path") or "").strip()
        if manifest_path:
            self.region_collection_manifest_edit.setText(manifest_path)
        view_payload = dict(payload)
        view_payload.setdefault("preset_id", "beamng_region_training_data")
        view_payload.setdefault("preset_label", "Collect BeamNG region training data")
        view_payload.setdefault("artifact_type", "beamng_collection")
        view_payload.setdefault("artifact_path", manifest_path)
        metrics = view_payload.get("metrics") if isinstance(view_payload.get("metrics"), dict) else {}
        view_payload.setdefault("history", {key: [value] for key, value in metrics.items() if isinstance(value, (int, float))})
        self.beamng_summary.setText(_compact_json(view_payload))
        self.model_summary.setText(_compact_json(view_payload))
        self._set_training_run_views(view_payload)
        self._set_beamng_quality_views(view_payload)
        self.log(f"BeamNG 训练数据采集完成: {manifest_path or services.NAN_TEXT}")
        self.refresh_catalogs()

    def _region_world_model_training_finished(self, payload: dict[str, Any]) -> None:
        model_dir = str(payload.get("model_dir") or "").strip()
        if model_dir:
            self.model_path_edit.setText(model_dir)
            self.home_model_combo.setCurrentText(model_dir)
        view_payload = dict(payload)
        view_payload.setdefault("preset_id", "region_world_model_training")
        view_payload.setdefault("preset_label", "Train BeamNG region world model")
        view_payload.setdefault("artifact_type", "world_model")
        view_payload.setdefault("artifact_path", model_dir)
        training = view_payload.get("training") if isinstance(view_payload.get("training"), dict) else {}
        metrics = training.get("metrics") if isinstance(training.get("metrics"), dict) else {}
        view_payload.setdefault("metrics", metrics)
        view_payload.setdefault("history", {key: [value] for key, value in metrics.items() if isinstance(value, (int, float))})
        self.model_summary.setText(_compact_json(view_payload))
        self.beamng_summary.setText(_compact_json(view_payload))
        self._set_training_run_views(view_payload)
        self._set_beamng_quality_views(view_payload)
        self.log(f"BeamNG world model 训练完成: {model_dir or services.NAN_TEXT}")
        self.refresh_catalogs()
        config = payload.get("world_model_config") if isinstance(payload.get("world_model_config"), dict) else {}
        config_id = str(config.get("id") or "").strip()
        if config_id:
            self._select_world_model_config(config_id)

    def _register_pipeline_world_model_config(self, payload: dict[str, Any]) -> str:
        existing = payload.get("world_model_config") if isinstance(payload.get("world_model_config"), dict) else {}
        if existing:
            return str(existing.get("id") or "")
        if str(payload.get("status") or "") != "completed":
            return ""
        model_dir = str(payload.get("model_dir") or "").strip()
        training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
        world_model = str(training.get("model_type") or "").strip()
        if not model_dir or world_model != "tiny_learned" or str(training.get("status") or "completed") != "completed":
            return ""
        acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
        if not bool(acceptance.get("goal_success")):
            return ""
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        task_id = str(task.get("task_id") or "self_supervised_world_model")
        row = services.save_world_model_config(
            config_id=f"{task_id}_self_supervised_world_model",
            label=f"{task_id} self-supervised world model",
            algorithm="world_model_direct",
            world_model=world_model,
            model_path=model_dir,
            source_training_run_path=str(payload.get("training_run_path") or ""),
            validation={
                "goal_success": bool(acceptance.get("goal_success")),
                "goal_reached": bool(acceptance.get("goal_reached")),
                "min_goal_distance": acceptance.get("min_goal_distance"),
                "final_goal_distance": acceptance.get("final_goal_distance"),
                "collision_count": acceptance.get("collision_count"),
            },
        )
        return str(row.get("id") or "")

    def _dataset_inspected(self, payload: dict[str, Any]) -> None:
        self.dataset_info = payload
        adapter = str(payload.get("adapter") or "")
        self.adapter_edit.setText(adapter)
        self.dataset_summary.setText(_compact_json(payload))
        self.dataset_detail_summary.setText(_dataset_detail_text(payload))
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        self.dataset_quality_summary.setText(_dataset_quality_text(quality))
        self._populate_dataset_analysis_tables(quality)
        self.sequence_combo.clear()
        sequence_ids = [str(item) for item in payload.get("sequences", [])]
        for sequence_id in sequence_ids:
            self.sequence_combo.addItem(sequence_id)
        if adapter == "manifest_dataset":
            self._set_custom_dataset_controls(True)
        else:
            self._set_detected_dataset_sequence_rows(sequence_ids, adapter)
        selected = str(payload.get("selected_sequence", ""))
        index = self.sequence_combo.findText(selected)
        if index >= 0:
            self.sequence_combo.setCurrentIndex(index)
        frame_count = int(payload.get("frame_count") or 0)
        self.dataset_frame_slider.setRange(0, max(0, frame_count - 1))
        self.dataset_frame_slider.setValue(min(self.settings.preview_frame_index, max(0, frame_count - 1)))
        self.dataset_frame_label.setText(f"帧 {self.dataset_frame_slider.value() + 1 if frame_count else 0} / {frame_count}")
        self.log(f"数据集 OK: {payload.get('dataset_id')} / frames={frame_count}")

    def _preview_ready(self, payload: dict[str, Any]) -> None:
        self._dataset_preview_busy = False
        previews = payload.get("previews", {}) if isinstance(payload.get("previews"), dict) else {}
        self._set_preview(self.rgb_preview, previews.get("front_rgb"), "RGB: NaN")
        self._set_preview(self.depth_preview, previews.get("depth") or previews.get("label"), "Depth/Label: NaN")
        self._set_preview(self.lidar_preview, previews.get("lidar_points"), "LiDAR: NaN")
        frame_count = int(payload.get("frame_count") or 0)
        frame_index = int(payload.get("frame_index") or 0)
        self.dataset_frame_slider.setRange(0, max(0, frame_count - 1))
        self.dataset_frame_slider.setValue(frame_index)
        self.dataset_frame_label.setText(f"帧 {frame_index + 1 if frame_count else 0} / {frame_count}")
        self.dataset_summary.setText(_compact_json(payload))
        if not self.dataset_play_button.isChecked():
            self.log(f"预览完成: frame={payload.get('frame_id', services.NAN_TEXT)}")

    def _dataset_quality_ready(self, payload: dict[str, Any]) -> None:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        self.dataset_quality_summary.setText(
            _dataset_quality_text(
                analysis,
                report_json=str(payload.get("report_json_path") or ""),
                report_markdown=str(payload.get("report_markdown_path") or ""),
            )
        )
        self.dataset_detail_summary.setText(_dataset_detail_text({"details": analysis, "quality": analysis}))
        self._populate_dataset_analysis_tables(analysis)
        self.log(f"质量报告完成：status={payload.get('status', services.NAN_TEXT)}")

    def _dataset_split_ready(self, payload: dict[str, Any]) -> None:
        self.dataset_split_path_edit.setText(str(payload.get("path") or ""))
        self.dataset_split_summary.setText(_dataset_split_text(payload))
        self.log(f"数据划分完成：{payload.get('path', services.NAN_TEXT)}")

    def _populate_dataset_analysis_tables(self, analysis: dict[str, Any]) -> None:
        self.dataset_sequence_detail_table.setRowCount(0)
        sequences = analysis.get("sequences") if isinstance(analysis.get("sequences"), list) else []
        for sequence in sequences:
            row = self.dataset_sequence_detail_table.rowCount()
            self.dataset_sequence_detail_table.insertRow(row)
            values = [
                sequence.get("sequence_id", ""),
                sequence.get("frame_count", 0),
                ", ".join(sequence.get("modalities", [])),
                sequence.get("time_start", services.NAN_TEXT),
                sequence.get("time_end", services.NAN_TEXT),
                int(sequence.get("frame_id_gap_count", 0)) + int(sequence.get("timestamp_issue_count", 0)),
            ]
            for column, value in enumerate(values):
                self.dataset_sequence_detail_table.setItem(row, column, QTableWidgetItem(str(value)))

        self.dataset_issue_table.setRowCount(0)
        issues = analysis.get("issues") if isinstance(analysis.get("issues"), list) else []
        for issue in issues:
            row = self.dataset_issue_table.rowCount()
            self.dataset_issue_table.insertRow(row)
            values = [
                issue.get("severity", ""),
                issue.get("code", ""),
                issue.get("sequence_id", ""),
                issue.get("frame_id", ""),
                issue.get("message", ""),
            ]
            for column, value in enumerate(values):
                self.dataset_issue_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _terrain_exported(self, payload: dict[str, Any]) -> None:
        self._set_preview(self.terrain_preview, payload.get("preview"), "Terrain: NaN")
        self.beamng_summary.setText(_compact_json(payload))
        self.log(f"地形草案已导出: {payload.get('manifest', services.NAN_TEXT)}")

    def _training_config_finished(self, payload: dict[str, Any]) -> None:
        if isinstance(payload, dict) and payload.get("output_hdf5"):
            self._hdf5_exported(payload)
            return
        self._training_finished(payload)

    def _training_finished(self, payload: dict[str, Any]) -> None:
        self.model_path_edit.setText(str(payload.get("output_dir", "")))
        if payload.get("output_dir"):
            self.home_model_combo.setCurrentText(str(payload.get("output_dir", "")))
            self.training_output_edit.setText(str(payload.get("output_dir", "")))
        self.model_summary.setText(_compact_json(payload))
        self._set_training_run_views(payload)
        self.log(f"模型训练完成: {payload.get('model_path', payload.get('checkpoint_path', services.NAN_TEXT))}")
        self.refresh_catalogs()

    def _hdf5_exported(self, payload: dict[str, Any]) -> None:
        self.stablewm_hdf5_edit.setText(str(payload.get("output_hdf5", "")))
        if payload.get("output_hdf5"):
            self.training_output_edit.setText(str(payload.get("output_hdf5", "")))
        self.dataset_summary.setText(_compact_json(payload))
        self._set_training_run_views(payload)
        self.log(f"HDF5 导出完成: {payload.get('output_hdf5', services.NAN_TEXT)}")

        self.refresh_catalogs()

    def _visible_demo_finished(self, payload: dict[str, Any]) -> None:
        self.beamng_summary.setText(_compact_json(payload))
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        self._update_metrics(metrics)
        path = payload.get("episode_path")
        self.trajectory.set_trace(services.load_episode_trace(path) if path else [])
        self.log(f"BeamNG 可视自动驾驶完成: {payload.get('episode_id', services.NAN_TEXT)}")
        self.refresh_catalogs()

    def _navigation_preview_finished(self, payload: dict[str, Any]) -> None:
        self.beamng_summary.setText(_compact_json(payload))
        analysis = payload.get("analysis", {}) if isinstance(payload.get("analysis"), dict) else {}
        self.log(
            "BeamNG 预览完成: "
            f"start={services.display_value(analysis.get('start_in_region'))}, "
            f"goal={services.display_value(analysis.get('goal_in_region'))}, "
            f"route={services.display_value(analysis.get('route_in_region'))}"
        )
        self.refresh_catalogs()
        self._finish_navigation_preview_task()

    def _update_metrics(self, metrics: dict[str, Any]) -> None:
        diagnostics = metrics.get("agent_diagnostics", {}) if isinstance(metrics.get("agent_diagnostics"), dict) else {}
        final_state = diagnostics.get("final_state", {}) if isinstance(diagnostics.get("final_state"), dict) else {}
        planning = diagnostics.get("planning", {}) if isinstance(diagnostics.get("planning"), dict) else {}
        prediction = planning.get("prediction", {}) if isinstance(planning.get("prediction"), dict) else diagnostics.get("prediction", {})
        values = {
            "steps": metrics.get("steps", math.nan),
            "done": metrics.get("done", math.nan),
            "best_cost": diagnostics.get("best_cost", math.nan),
            "final_speed": final_state.get("speed", math.nan),
            "max_risk": prediction.get("max_risk", math.nan) if isinstance(prediction, dict) else math.nan,
            "reward": metrics.get("reward", math.nan),
        }
        for key, value in values.items():
            self.metric_cards[key].set_value(value)

    def _fill_training_config_combo(self) -> None:
        current = self.training_config_combo.currentData()
        current_id = current.get("id") if isinstance(current, dict) else ""
        self.training_config_combo.blockSignals(True)
        self.training_config_combo.clear()
        self.training_config_combo.addItem("Manual training config", {})
        selected_index = 0
        for row in self.catalog.get("training_configs", []):
            config_id = str(row.get("id") or "")
            if not config_id:
                continue
            label = str(row.get("label") or config_id)
            preset = str(row.get("training_preset_id") or "")
            suffix = f" ({preset})" if preset else ""
            self.training_config_combo.addItem(f"{label}{suffix}", dict(row))
            if config_id == current_id:
                selected_index = self.training_config_combo.count() - 1
        self.training_config_combo.setCurrentIndex(selected_index)
        self.training_config_combo.blockSignals(False)

    def _select_training_config(self, config_id: str) -> None:
        if not config_id:
            return
        for index in range(self.training_config_combo.count()):
            data = self.training_config_combo.itemData(index)
            if isinstance(data, dict) and str(data.get("id") or "") == config_id:
                self.training_config_combo.setCurrentIndex(index)
                self._apply_training_config_row(data)
                return

    def _sync_training_config_selection(self) -> None:
        data = self.training_config_combo.currentData()
        if isinstance(data, dict) and data.get("id"):
            self._apply_training_config_row(data)

    def _apply_training_config_row(self, row: dict[str, Any]) -> None:
        self.training_config_name_edit.setText(str(row.get("label") or ""))
        dataset_root = str(row.get("dataset_root") or "").strip()
        if dataset_root:
            self.dataset_root_edit.setText(dataset_root)
        adapter = str(row.get("adapter") or "")
        self.adapter_edit.setText(adapter)
        self._set_custom_dataset_controls(adapter == "manifest_dataset")
        sequence_id = str(row.get("sequence_id") or "")
        if sequence_id:
            self.sequence_combo.setCurrentText(sequence_id)
        if adapter and adapter != "manifest_dataset":
            self._set_detected_dataset_sequence_rows([sequence_id] if sequence_id else [], adapter)
        self.dataset_split_path_edit.setText(str(row.get("split_path") or ""))
        preset_id = str(row.get("training_preset_id") or "")
        if preset_id:
            self._select_training_preset(preset_id)
        output_path = str(row.get("output_path") or "").strip()
        if output_path:
            self.training_output_edit.setText(output_path)
            if preset_id == "stablewm_hdf5":
                self.stablewm_hdf5_edit.setText(output_path)
            else:
                self.home_model_combo.setCurrentText(output_path)
                self.model_path_edit.setText(output_path)
        parameters = row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
        text = json.dumps(parameters, indent=2, ensure_ascii=False) if parameters else "{}"
        self._trainer_params_autofill = text
        self.trainer_params_edit.setPlainText(text)
        self._set_trainer_parameter_values(parameters)
        self.model_summary.setText(_compact_json({"training_config": row}))

    def _current_training_output_path(self, preset_id: str = "") -> str:
        output = self.training_output_edit.text().strip() if hasattr(self, "training_output_edit") else ""
        if output:
            return output
        if preset_id == "stablewm_hdf5":
            return self.stablewm_hdf5_edit.text().strip()
        return self.model_path_edit.text().strip() or self._path_combo_value(self.home_model_combo).strip()

    def _fill_dataset_manifest_combo(self) -> None:
        current = self.dataset_catalog_combo.currentData()
        current_id = current.get("id") if isinstance(current, dict) else ""
        self.dataset_catalog_combo.blockSignals(True)
        self.dataset_catalog_combo.clear()
        self.dataset_catalog_combo.addItem("临时数据集（直接选择文件夹）", {})
        selected_index = 0
        for row in self.catalog.get("dataset_manifests", []):
            dataset_id = str(row.get("id") or "")
            if not dataset_id:
                continue
            label = str(row.get("label") or dataset_id)
            adapter = str(row.get("adapter") or "manifest_dataset")
            self.dataset_catalog_combo.addItem(f"{label} ({adapter})", dict(row))
            if dataset_id == current_id:
                selected_index = self.dataset_catalog_combo.count() - 1
        self.dataset_catalog_combo.setCurrentIndex(selected_index)
        self.dataset_catalog_combo.blockSignals(False)

    def _select_dataset_manifest(self, dataset_id: str) -> None:
        if not dataset_id:
            return
        for index in range(self.dataset_catalog_combo.count()):
            data = self.dataset_catalog_combo.itemData(index)
            if isinstance(data, dict) and str(data.get("id") or "") == dataset_id:
                self.dataset_catalog_combo.setCurrentIndex(index)
                self._apply_dataset_manifest_row(data)
                return

    def _sync_dataset_manifest_selection(self) -> None:
        data = self.dataset_catalog_combo.currentData()
        if isinstance(data, dict) and data.get("id"):
            self._apply_dataset_manifest_row(data)

    def _apply_dataset_manifest_row(self, row: dict[str, Any]) -> None:
        dataset_root = str(row.get("dataset_root") or "").strip()
        if dataset_root:
            self.dataset_root_edit.setText(dataset_root)
        self.adapter_edit.setText(str(row.get("adapter") or "manifest_dataset"))
        self._set_custom_dataset_controls(str(row.get("adapter") or "manifest_dataset") == "manifest_dataset")
        sequences = row.get("sequences") if isinstance(row.get("sequences"), list) else []
        self._set_dataset_sequence_rows(
            [item if isinstance(item, dict) else {"id": str(item), "root": "."} for item in sequences]
        )
        self.sequence_combo.clear()
        for sequence in sequences:
            sequence_id = sequence.get("id") if isinstance(sequence, dict) else sequence
            self.sequence_combo.addItem(str(sequence_id))
        if sequences:
            self.sequence_combo.setCurrentIndex(0)
        self.dataset_summary.setText(_compact_json({"dataset": row}))

    def _set_custom_dataset_controls(self, enabled: bool) -> None:
        self._dataset_sequence_view_mode = "mapping" if enabled else "detected"
        self.dataset_manifest_name_edit.setEnabled(enabled)
        self.dataset_sequence_add_button.setEnabled(enabled)
        self.dataset_sequence_remove_button.setEnabled(enabled)
        self.dataset_save_button.setEnabled(enabled)
        self.dataset_sequence_table.setEnabled(True)
        header = self.dataset_sequence_table.horizontalHeader()
        if enabled:
            self.dataset_sequence_table.setColumnCount(5)
            self.dataset_sequence_table.setHorizontalHeaderLabels(
                ["序列 ID", "相对目录", "位姿 CSV", "动作 CSV", "资产映射 JSON"]
            )
            self.dataset_sequence_table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
            self.dataset_sequence_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            for column in range(4):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            self.dataset_sequence_table.setToolTip("编辑没有内置适配器的数据集序列目录、CSV 和资产映射。")
        else:
            self.dataset_sequence_table.setColumnCount(2)
            self.dataset_sequence_table.setHorizontalHeaderLabels(["已识别序列", "适配器"])
            self.dataset_sequence_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.dataset_sequence_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.dataset_sequence_table.setToolTip("只读显示适配器识别到的序列；选择一行可切换当前序列。")

    def _set_detected_dataset_sequence_rows(self, sequence_ids: list[str], adapter: str) -> None:
        self._set_custom_dataset_controls(False)
        self.dataset_sequence_table.blockSignals(True)
        self.dataset_sequence_table.setRowCount(0)
        for sequence_id in sequence_ids:
            row = self.dataset_sequence_table.rowCount()
            self.dataset_sequence_table.insertRow(row)
            self.dataset_sequence_table.setItem(row, 0, QTableWidgetItem(sequence_id))
            self.dataset_sequence_table.setItem(row, 1, QTableWidgetItem(adapter))
        self.dataset_sequence_table.blockSignals(False)
        self._sync_sequence_table_selection(self.sequence_combo.currentText())

    def _sync_sequence_from_table_selection(self) -> None:
        if self._dataset_sequence_view_mode != "detected":
            return
        row = self.dataset_sequence_table.currentRow()
        item = self.dataset_sequence_table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        index = self.sequence_combo.findText(item.text())
        if index >= 0 and index != self.sequence_combo.currentIndex():
            self.sequence_combo.setCurrentIndex(index)

    def _sync_sequence_table_selection(self, sequence_id: str) -> None:
        if self._dataset_sequence_view_mode != "detected":
            return
        target_row = -1
        for row in range(self.dataset_sequence_table.rowCount()):
            item = self.dataset_sequence_table.item(row, 0)
            if item is not None and item.text() == sequence_id:
                target_row = row
                break
        self.dataset_sequence_table.blockSignals(True)
        if target_row >= 0:
            self.dataset_sequence_table.selectRow(target_row)
        else:
            self.dataset_sequence_table.clearSelection()
        self.dataset_sequence_table.blockSignals(False)

    def _fill_training_preset_combo(self) -> None:
        current = self.training_preset_combo.currentData()
        current_id = current.get("id") if isinstance(current, dict) else "lewm_cost_model"
        self.training_preset_combo.blockSignals(True)
        self.training_preset_combo.clear()
        selected_index = -1
        for row in self.catalog.get("training_presets", []):
            preset_id = str(row.get("id") or "")
            if not preset_id:
                continue
            label = str(row.get("label") or preset_id)
            if row.get("available") is False:
                label = f"{label} ({row.get('status', services.UNFINISHED_TEXT)})"
            self.training_preset_combo.addItem(label, dict(row))
            if preset_id == current_id:
                selected_index = self.training_preset_combo.count() - 1
        if self.training_preset_combo.count():
            self.training_preset_combo.setCurrentIndex(max(0, selected_index))
        self.training_preset_combo.blockSignals(False)
        self._sync_training_preset_selection()

    def _select_training_preset(self, preset_id: str) -> None:
        if not preset_id:
            return
        for index in range(self.training_preset_combo.count()):
            data = self.training_preset_combo.itemData(index)
            if isinstance(data, dict) and str(data.get("id") or "") == preset_id:
                self.training_preset_combo.setCurrentIndex(index)
                return

    def _sync_training_preset_selection(self, *, force: bool = False) -> None:
        self._sync_training_preset_summary()
        self._sync_training_preset_params(force=force)

    def _sync_training_preset_summary(self) -> None:
        if not hasattr(self, "training_preset_summary"):
            return
        data = self.training_preset_combo.currentData()
        preset = dict(data) if isinstance(data, dict) else {}
        if not preset:
            self.training_preset_summary.setPlainText("Training config: NaN")
            return
        lines = [
            f"Name: {preset.get('label') or preset.get('id') or services.NAN_TEXT}",
            f"Kind: {preset.get('kind') or services.NAN_TEXT}",
            f"Status: {preset.get('status') or ('available' if preset.get('available', True) else services.UNFINISHED_TEXT)}",
        ]
        description = str(preset.get("description") or "").strip()
        if description:
            lines.append(f"Description: {description}")
        manifest_path = str(preset.get("manifest_path") or "").strip()
        if manifest_path:
            lines.append(f"Trainer manifest: {manifest_path}")
        outputs = preset.get("outputs") if isinstance(preset.get("outputs"), dict) else {}
        artifact_type = outputs.get("artifact_type")
        if artifact_type:
            lines.append(f"Artifact: {artifact_type}")
        self.training_preset_summary.setPlainText("\n".join(lines))

    def _sync_training_preset_params(self, *, force: bool = False) -> None:
        if not hasattr(self, "trainer_params_edit"):
            return
        data = self.training_preset_combo.currentData()
        preset = dict(data) if isinstance(data, dict) else {}
        schema = preset.get("parameters") if isinstance(preset.get("parameters"), dict) else {}
        defaults = self._trainer_parameter_defaults(schema)
        text = json.dumps(defaults, indent=2, ensure_ascii=False) if defaults else "{}"
        current = self.trainer_params_edit.toPlainText().strip()
        if not force and current and current != self._trainer_params_autofill:
            return
        self._build_trainer_parameter_form(schema, defaults)
        self._trainer_params_autofill = text
        self.trainer_params_edit.setPlainText(text)

    def _trainer_parameter_defaults(self, schema: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for key, spec in schema.items():
            if isinstance(spec, dict) and "default" in spec:
                defaults[str(key)] = spec.get("default")
        return defaults

    def _trainer_parameters_from_text(self) -> dict[str, Any]:
        text = self.trainer_params_edit.toPlainText().strip() if hasattr(self, "trainer_params_edit") else ""
        payload = self._trainer_parameter_control_values()
        if text and text != self._trainer_params_autofill:
            try:
                overrides = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Expected JSON object: {exc}") from exc
            if not isinstance(overrides, dict):
                raise ValueError("Expected JSON object.")
            payload.update(overrides)
        return payload

    def _build_trainer_parameter_form(self, schema: dict[str, Any], values: dict[str, Any]) -> None:
        while self.trainer_parameter_form_layout.count():
            item = self.trainer_parameter_form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.trainer_parameter_controls.clear()
        self.trainer_parameter_specs = {
            str(name): dict(spec) if isinstance(spec, dict) else {}
            for name, spec in schema.items()
        }
        if not self.trainer_parameter_specs:
            empty = QLabel("此训练器没有可配置参数")
            empty.setObjectName("mutedText")
            self.trainer_parameter_form_layout.addRow(empty)
            return
        for name, spec in self.trainer_parameter_specs.items():
            value_type = str(spec.get("type") or "str").lower()
            choices = spec.get("enum") if isinstance(spec.get("enum"), list) else []
            value = values.get(name, spec.get("default"))
            form_control: QWidget
            if choices:
                control: QWidget = QComboBox()
                for choice in choices:
                    control.addItem(str(choice), choice)
                selected = next(
                    (index for index in range(control.count()) if control.itemData(index) == value),
                    0,
                )
                control.setCurrentIndex(selected)
                control.currentIndexChanged.connect(self._trainer_parameter_control_changed)
            elif value_type in {"bool", "boolean"}:
                control = QCheckBox()
                control.setChecked(bool(value))
                control.toggled.connect(self._trainer_parameter_control_changed)
            elif value_type in {"int", "integer"}:
                control = QSpinBox()
                control.setRange(int(spec.get("min", -1_000_000_000)), int(spec.get("max", 1_000_000_000)))
                if "step" in spec:
                    control.setSingleStep(max(1, int(spec["step"])))
                control.setValue(int(value or 0))
                control.valueChanged.connect(self._trainer_parameter_control_changed)
            elif value_type in {"float", "number"}:
                control = QDoubleSpinBox()
                control.setRange(float(spec.get("min", -1e12)), float(spec.get("max", 1e12)))
                control.setDecimals(int(spec.get("decimals", 8)))
                control.setSingleStep(float(spec.get("step", 0.01)))
                control.setValue(float(value or 0.0))
                control.valueChanged.connect(self._trainer_parameter_control_changed)
            elif value_type in {"path", "file", "directory"}:
                control = QLineEdit()
                control.setText("" if value is None else str(value))
                control.textChanged.connect(self._trainer_parameter_control_changed)
                browse = QPushButton("选择")
                self._configure_button(browse)
                if value_type == "directory":
                    browse.clicked.connect(lambda _=False, target=control: self._browse_dir(target))
                else:
                    browse.clicked.connect(lambda _=False, target=control: self._browse_any_file(target))
                form_control = self._with_button(control, browse)
            else:
                control = QLineEdit()
                control.setText("" if value is None else str(value))
                control.textChanged.connect(self._trainer_parameter_control_changed)
            if value_type not in {"path", "file", "directory"}:
                form_control = control
            control.setObjectName(f"trainerParameter_{name}")
            control.setMinimumHeight(CONTROL_HEIGHT)
            description = str(spec.get("description") or "").strip()
            if description:
                control.setToolTip(description)
            label = str(spec.get("label") or name)
            if spec.get("required") is True:
                label += " *"
            self.trainer_parameter_controls[name] = control
            self.trainer_parameter_form_layout.addRow(label, form_control)
        self._update_trainer_parameter_dependencies()

    def _set_trainer_parameter_values(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            control = self.trainer_parameter_controls.get(str(name))
            if control is None:
                continue
            control.blockSignals(True)
            if isinstance(control, QComboBox):
                index = next(
                    (item for item in range(control.count()) if control.itemData(item) == value),
                    control.findText(str(value)),
                )
                if index >= 0:
                    control.setCurrentIndex(index)
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(value)
            elif isinstance(control, QLineEdit):
                control.setText(str(value))
            control.blockSignals(False)
        self._update_trainer_parameter_dependencies()

    def _trainer_parameter_control_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, control in self.trainer_parameter_controls.items():
            if not control.isEnabled():
                continue
            if isinstance(control, QComboBox):
                values[name] = control.currentData()
            elif isinstance(control, QCheckBox):
                values[name] = control.isChecked()
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                values[name] = control.value()
            elif isinstance(control, QLineEdit):
                text = control.text().strip()
                if text or "default" in self.trainer_parameter_specs.get(name, {}):
                    values[name] = text
        return values

    def _trainer_parameter_control_changed(self, *args: Any) -> None:
        del args
        self._update_trainer_parameter_dependencies()
        values = self._trainer_parameter_control_values()
        text = json.dumps(values, indent=2, ensure_ascii=False)
        self._trainer_params_autofill = text
        self.trainer_params_edit.setPlainText(text)

    def _update_trainer_parameter_dependencies(self) -> None:
        values: dict[str, Any] = {}
        for name, control in self.trainer_parameter_controls.items():
            if isinstance(control, QComboBox):
                values[name] = control.currentData()
            elif isinstance(control, QCheckBox):
                values[name] = control.isChecked()
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                values[name] = control.value()
            elif isinstance(control, QLineEdit):
                values[name] = control.text().strip()
        for name, control in self.trainer_parameter_controls.items():
            dependency = self.trainer_parameter_specs.get(name, {}).get("depends_on")
            enabled = True
            if isinstance(dependency, str):
                enabled = bool(values.get(dependency))
            elif isinstance(dependency, dict):
                dependency_name = str(dependency.get("parameter") or dependency.get("name") or "")
                current = values.get(dependency_name)
                if "equals" in dependency:
                    enabled = current == dependency.get("equals")
                elif "not_equals" in dependency:
                    enabled = current != dependency.get("not_equals")
                else:
                    enabled = bool(current)
            control.setEnabled(enabled)

    def _dataset_sequences_from_text(self) -> list[dict[str, Any]]:
        if hasattr(self, "dataset_sequence_table") and self.dataset_sequence_table.rowCount() > 0:
            rows: list[dict[str, Any]] = []
            for row_index in range(self.dataset_sequence_table.rowCount()):
                values = [
                    self.dataset_sequence_table.item(row_index, column).text().strip()
                    if self.dataset_sequence_table.item(row_index, column) is not None
                    else ""
                    for column in range(self.dataset_sequence_table.columnCount())
                ]
                if not values[0]:
                    continue
                row: dict[str, Any] = {"id": values[0], "root": values[1] or "."}
                if values[2]:
                    row["pose_csv"] = values[2]
                if values[3]:
                    row["actions_csv"] = values[3]
                if values[4]:
                    try:
                        assets = json.loads(values[4])
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"序列 {values[0]} 的资产映射不是有效 JSON：{exc}") from exc
                    if not isinstance(assets, dict):
                        raise ValueError(f"序列 {values[0]} 的资产映射必须是 JSON 对象。")
                    row["assets"] = assets
                rows.append(row)
            if rows:
                self.dataset_manifest_sequences_edit.setPlainText(json.dumps(rows, indent=2, ensure_ascii=False))
                return rows
        text = self.dataset_manifest_sequences_edit.toPlainText().strip() if hasattr(self, "dataset_manifest_sequences_edit") else ""
        if not text:
            sequence_id = self.sequence_combo.currentText().strip() or "sequence_001"
            return [{"id": sequence_id, "root": "."}]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON list or object for dataset sequences: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("sequences", [payload])
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("Expected dataset sequences to be a JSON list of objects.")
        return [dict(item) for item in payload]

    def _set_dataset_sequence_rows(self, sequences: list[dict[str, Any]]) -> None:
        self.dataset_sequence_table.setRowCount(0)
        for sequence in sequences:
            row_index = self.dataset_sequence_table.rowCount()
            self.dataset_sequence_table.insertRow(row_index)
            values = [
                str(sequence.get("id") or sequence.get("sequence_id") or ""),
                str(sequence.get("root") or "."),
                str(sequence.get("pose_csv") or ""),
                str(sequence.get("actions_csv") or ""),
                json.dumps(sequence.get("assets") or {}, ensure_ascii=False),
            ]
            for column, value in enumerate(values):
                self.dataset_sequence_table.setItem(row_index, column, QTableWidgetItem(value))
        self.dataset_manifest_sequences_edit.setPlainText(json.dumps(sequences, indent=2, ensure_ascii=False))

    def _add_dataset_sequence_row(self) -> None:
        row = self.dataset_sequence_table.rowCount()
        self.dataset_sequence_table.insertRow(row)
        defaults = [f"sequence_{row + 1:03d}", ".", "poses.csv", "", "{}"]
        for column, value in enumerate(defaults):
            self.dataset_sequence_table.setItem(row, column, QTableWidgetItem(value))
        self.dataset_sequence_table.setCurrentCell(row, 0)

    def _remove_dataset_sequence_rows(self) -> None:
        rows = sorted({index.row() for index in self.dataset_sequence_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.dataset_sequence_table.removeRow(row)

    def _trainer_arguments_from_text(self) -> list[str]:
        text = self.trainer_arguments_edit.toPlainText().strip() if hasattr(self, "trainer_arguments_edit") else ""
        if not text:
            return ["{dataset_root}", "--output", "{output_dir}"]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON list for trainer arguments: {exc}") from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("Expected trainer arguments to be a JSON list of strings.")
        return payload

    def _trainer_arguments_for_script_config(self) -> list[str] | None:
        arguments = self._trainer_arguments_from_text()
        default_arguments = ["{dataset_root}", "--output", "{output_dir}"]
        return None if arguments == default_arguments else arguments

    def _trainer_schema_from_text(self) -> dict[str, Any]:
        text = self.trainer_schema_edit.toPlainText().strip() if hasattr(self, "trainer_schema_edit") else ""
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON object for trainer parameter schema: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Expected trainer parameter schema to be a JSON object.")
        return payload

    def _fill_training_run_list(self) -> None:
        if not hasattr(self, "training_run_list"):
            return
        runs = [dict(run) for run in self.catalog.get("training_runs", [])]
        self._fill_experiment_filter_combo(
            self.experiment_status_filter,
            [(value, value) for value in sorted({str(run.get("status") or "") for run in runs if run.get("status")})],
            "全部状态",
        )
        self._fill_experiment_filter_combo(
            self.experiment_preset_filter,
            sorted(
                {
                    (str(run.get("preset_label") or run.get("preset_id") or ""), str(run.get("preset_id") or ""))
                    for run in runs
                    if run.get("preset_id")
                }
            ),
            "全部模型/训练器",
        )
        self._fill_experiment_filter_combo(
            self.experiment_dataset_filter,
            sorted(
                {
                    (Path(str(run.get("dataset_root"))).name or str(run.get("dataset_root")), str(run.get("dataset_root")))
                    for run in runs
                    if run.get("dataset_root")
                }
            ),
            "全部数据集",
        )
        self._filtered_training_runs = services.filter_training_runs(
            runs,
            query=self.experiment_query_edit.text(),
            preset_id=str(self.experiment_preset_filter.currentData() or ""),
            dataset_root=str(self.experiment_dataset_filter.currentData() or ""),
            status=str(self.experiment_status_filter.currentData() or ""),
            date_from=self.experiment_date_from_edit.text().strip(),
            date_to=self.experiment_date_to_edit.text().strip(),
        )
        self.training_run_list.clear()
        for run in self._filtered_training_runs[:200]:
            label = str(run.get("preset_label") or run.get("preset_id") or services.NAN_TEXT)
            status = str(run.get("status") or services.NAN_TEXT)
            artifact = str(run.get("artifact_path") or run.get("relative_path") or "")
            best_mark = "★ " if run.get("best_marks") else ""
            item = QListWidgetItem(f"{best_mark}{label} | {status} | {artifact}")
            item.setData(Qt.ItemDataRole.UserRole, dict(run))
            self.training_run_list.addItem(item)

    def _fill_experiment_filter_combo(
        self,
        combo: QComboBox,
        rows: list[tuple[str, str]],
        all_label: str,
    ) -> None:
        current = str(combo.currentData() or "")
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(all_label, "")
        for label, value in rows:
            combo.addItem(label, value)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _fill_training_artifact_combo(self) -> None:
        if not hasattr(self, "training_artifact_combo"):
            return
        current = self.training_artifact_combo.currentData()
        current_path = str(current.get("artifact_path") or "") if isinstance(current, dict) else ""
        self.training_artifact_combo.blockSignals(True)
        self.training_artifact_combo.clear()
        selected_index = 0
        for artifact in self.catalog.get("training_artifacts", []):
            if not artifact.get("exists") or str(artifact.get("status") or "").lower() != "completed":
                continue
            label = str(artifact.get("label") or artifact.get("id") or services.NAN_TEXT)
            capability = "可推理" if artifact.get("inference_available") else "无推理入口"
            tags = []
            if artifact.get("latest"):
                tags.append("latest")
            if artifact.get("best"):
                tags.append("best")
            if artifact.get("favorite"):
                tags.append("★")
            tag_text = f" [{' / '.join(tags)}]" if tags else ""
            self.training_artifact_combo.addItem(f"{label}{tag_text} | {capability}", dict(artifact))
            if str(artifact.get("artifact_path") or "") == current_path:
                selected_index = self.training_artifact_combo.count() - 1
        if self.training_artifact_combo.count():
            self.training_artifact_combo.setCurrentIndex(selected_index)
        self.training_artifact_combo.blockSignals(False)
        self._update_training_artifact_details()

    def _update_training_artifact_details(self) -> None:
        if not hasattr(self, "training_artifact_detail_label"):
            return
        data = self.training_artifact_combo.currentData()
        artifact = dict(data) if isinstance(data, dict) else {}
        if not artifact:
            self.inference_params_edit.clear()
            self._inference_params_artifact_path = ""
            self.training_artifact_detail_label.setText("Checkpoint：NaN")
            self.favorite_artifact_button.setText("收藏")
            return
        artifact_path = str(artifact.get("artifact_path") or "")
        if artifact_path != self._inference_params_artifact_path:
            manifest_path = str(artifact.get("trainer_manifest_path") or "")
            try:
                defaults = services.inference_parameter_defaults(manifest_path) if manifest_path else {}
            except Exception:
                defaults = {}
            self.inference_params_edit.setPlainText(json.dumps(defaults, indent=2, ensure_ascii=False))
            self._inference_params_artifact_path = artifact_path
        tags = [name for name in ("latest", "best", "favorite") if artifact.get(name)]
        metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
        parameters = artifact.get("parameters") if isinstance(artifact.get("parameters"), dict) else {}
        self.training_artifact_detail_label.setText(
            f"Checkpoint：{artifact.get('artifact_path', services.NAN_TEXT)} | "
            f"标签：{', '.join(tags) if tags else '无'} | epoch: {services.display_value(artifact.get('epoch'))} | "
            f"metrics: {_mapping_preview(metrics)} | parameters: {_mapping_preview(parameters)}"
        )
        self.favorite_artifact_button.setText("取消收藏" if artifact.get("favorite") else "收藏")

    def toggle_selected_artifact_favorite(self) -> None:
        data = self.training_artifact_combo.currentData()
        artifact = dict(data) if isinstance(data, dict) else {}
        path = str(artifact.get("training_run_path") or "")
        if not path:
            return
        try:
            services.set_training_artifact_favorite(path, not bool(artifact.get("favorite")))
        except Exception as exc:
            self.training_artifact_detail_label.setText(f"Checkpoint 收藏失败：{exc}")
            return
        self.refresh_catalogs()

    def delete_selected_training_artifact(self) -> None:
        data = self.training_artifact_combo.currentData()
        artifact = dict(data) if isinstance(data, dict) else {}
        path = str(artifact.get("training_run_path") or "")
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "删除训练产物",
            "只会删除 checkpoint/模型产物并保留训练记录；被配置引用的产物会拒绝删除。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            payload = services.delete_training_artifact(path)
        except Exception as exc:
            self.training_artifact_detail_label.setText(f"Checkpoint 删除失败：{exc}")
            return
        self.log(f"训练产物已删除: {payload['artifact_path']}")
        self.refresh_catalogs()

    def run_selected_artifact_inference(self) -> None:
        data = self.training_artifact_combo.currentData()
        artifact = dict(data) if isinstance(data, dict) else {}
        if not artifact:
            self.inference_metric_summary.setText("推理指标：未选择 checkpoint")
            return
        if not artifact.get("inference_available"):
            self.inference_metric_summary.setText("推理指标：该训练器没有声明 inference 入口")
            return
        try:
            inference_parameters = json.loads(self.inference_params_edit.toPlainText().strip() or "{}")
            if not isinstance(inference_parameters, dict):
                raise ValueError("推理参数必须是 JSON 对象。")
        except (json.JSONDecodeError, ValueError) as exc:
            self.inference_metric_summary.setText(f"推理参数无效：{exc}")
            return
        manifest_path = str(artifact.get("trainer_manifest_path") or "")
        self._run_task(
            lambda: services.run_inference_manifest_job(
                manifest_path,
                artifact_path=str(artifact.get("artifact_path") or ""),
                dataset_root=self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                split_path=str(artifact.get("split_path") or self.dataset_split_path_edit.text().strip()),
                parameters=inference_parameters,
            ),
            self._inference_finished,
            "推理运行失败",
            task_label=f"推理 {artifact.get('label', artifact.get('id', 'checkpoint'))}",
        )

    def _inference_finished(self, payload: dict[str, Any]) -> None:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        self.inference_metric_summary.setText("推理指标：" + _compact_json(metrics))
        predictions = payload.get("predictions") if isinstance(payload.get("predictions"), list) else []
        columns, normalized_rows = _inference_prediction_rows(predictions[:100])
        self.inference_prediction_table.setColumnCount(len(columns))
        self.inference_prediction_table.setHorizontalHeaderLabels(columns)
        self.inference_prediction_table.setRowCount(min(100, len(predictions)))
        for row_index, row in enumerate(normalized_rows):
            values = [row.get(column) for column in columns]
            for column, value in enumerate(values):
                self.inference_prediction_table.setItem(row_index, column, QTableWidgetItem(services.display_value(value)))
        previews = payload.get("previews") if isinstance(payload.get("previews"), dict) else {}
        preview_path = next(iter(previews.values()), "")
        self._set_preview(self.inference_preview, preview_path, "推理预览：NaN")
        self.model_summary.setText(_compact_json(payload))
        self.log(f"推理完成: {payload.get('path', services.NAN_TEXT)}")
        self.refresh_catalogs()

    def _fill_training_job_table(self, jobs: list[dict[str, Any]] | None = None) -> None:
        if not hasattr(self, "training_job_table"):
            return
        rows = list(jobs if jobs is not None else self.training_job_queue.snapshots())
        self.training_job_table.setRowCount(len(rows))
        for row_index, job in enumerate(rows):
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            label = str(metadata.get("trainer_label") or metadata.get("trainer_id") or job.get("job_id") or services.NAN_TEXT)
            progress = job.get("progress")
            progress_text = f"{float(progress) * 100:.0f}%" if isinstance(progress, (int, float)) else services.NAN_TEXT
            values = [
                label,
                str(job.get("status") or services.NAN_TEXT),
                progress_text,
                str(job.get("output_dir") or ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, dict(job))
                self.training_job_table.setItem(row_index, column, item)

    def _refresh_training_jobs(self) -> None:
        snapshots = self.training_job_queue.snapshots()
        self._fill_training_job_table(snapshots)
        current = next(
            (row for row in snapshots if str(row.get("job_id") or "") == self._current_training_job_id),
            snapshots[0] if snapshots else {},
        )
        if not current:
            self.training_job_status_label.setText("训练任务：NaN")
            self.training_job_progress.setValue(0)
            self.training_job_progress.setFormat("等待任务")
            self.cancel_training_button.setEnabled(False)
            self.training_job_timer.stop()
            return
        status = str(current.get("status") or services.NAN_TEXT)
        message = str(current.get("message") or "")
        progress = float(current.get("progress") or 0.0)
        eta_seconds = current.get("eta_seconds")
        eta_text = (
            f" | ETA {_format_duration(float(eta_seconds))}"
            if isinstance(eta_seconds, (int, float)) and math.isfinite(float(eta_seconds))
            else ""
        )
        resources = current.get("resources") if isinstance(current.get("resources"), dict) else {}
        resource_parts = []
        for key, label, suffix in (
            ("cpu_percent", "CPU", "%"),
            ("memory_mb", "RAM", " MB"),
            ("gpu_percent", "GPU", "%"),
            ("gpu_memory_mb", "VRAM", " MB"),
        ):
            value = resources.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                resource_parts.append(f"{label} {float(value):.1f}{suffix}")
        resource_text = f" | {' / '.join(resource_parts)}" if resource_parts else ""
        self.training_job_status_label.setText(
            f"训练任务：{current.get('job_id', services.NAN_TEXT)} | {status} | {message}{eta_text}{resource_text}"
        )
        self.training_job_progress.setValue(max(0, min(100, int(round(progress * 100)))))
        self.training_job_progress.setFormat(f"{status} %p%")
        active = status in {"queued", "running", "canceling"}
        self.cancel_training_button.setEnabled(status in {"queued", "running"})
        log_path = Path(str(current.get("stdout_path") or ""))
        stderr_path = Path(str(current.get("stderr_path") or ""))
        log_text = self._training_job_log_tail(log_path, stderr_path)
        if log_text:
            self.latest_training_log.setPlainText(log_text)
            cursor = self.latest_training_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.latest_training_log.setTextCursor(cursor)
        if active:
            self._set_live_training_metric_views(services.live_training_metric_record(current))
        job_id = str(current.get("job_id") or "")
        if not active and job_id and job_id not in self._handled_training_job_ids:
            self._handled_training_job_ids.add(job_id)
            result = current.get("result") if isinstance(current.get("result"), dict) else {}
            if status == "completed":
                self._training_config_finished(result)
            else:
                payload = {
                    "status": status,
                    "message": current.get("error") or message,
                    "training_job": current,
                    **result,
                }
                self.model_summary.setText(_compact_json(payload))
                self._set_training_run_views(payload)
                self.log(f"训练任务{status}: {payload['message']}")
                self.refresh_catalogs()
            if log_text:
                self.latest_training_log.setPlainText(log_text)
        if not any(str(row.get("status") or "") in {"queued", "running", "canceling"} for row in snapshots):
            self.training_job_timer.stop()

    def cancel_current_training_job(self) -> None:
        if not self._current_training_job_id:
            return
        if self.training_job_queue.cancel(self._current_training_job_id):
            self.log(f"正在取消训练任务: {self._current_training_job_id}")
            self._refresh_training_jobs()

    def _select_training_job_row(self, row: int, column: int) -> None:
        item = self.training_job_table.item(row, column) or self.training_job_table.item(row, 0)
        data = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(data, dict):
            return
        job_id = str(data.get("job_id") or "")
        if job_id:
            self._current_training_job_id = job_id
            self._refresh_training_jobs()

    def _training_job_log_tail(self, stdout_path: Path, stderr_path: Path, max_chars: int = 12000) -> str:
        chunks: list[str] = []
        for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()
            except OSError:
                continue
            if text:
                chunks.append(f"[{label}]\n{text}")
        return "\n\n".join(chunks)

    def _load_selected_training_run(self, item: QListWidgetItem | None) -> None:
        if item is None or not hasattr(self, "training_run_summary"):
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        run = dict(data) if isinstance(data, dict) else {}
        self._set_training_run_views(run)
        artifact_path = str(run.get("artifact_path") or "")
        artifact_type = str(run.get("artifact_type") or "")
        if artifact_path and artifact_type in {"checkpoint", "world_model"}:
            self._select_path_combo_value(self.home_model_combo, artifact_path)
            for index in range(self.training_artifact_combo.count()):
                artifact = self.training_artifact_combo.itemData(index)
                if isinstance(artifact, dict) and str(artifact.get("artifact_path") or "") == artifact_path:
                    self.training_artifact_combo.setCurrentIndex(index)
                    break
        elif artifact_path and artifact_type == "hdf5":
            self.stablewm_hdf5_edit.setText(artifact_path)

    def _selected_training_runs(self) -> list[dict[str, Any]]:
        items = self.training_run_list.selectedItems()
        if not items and self.training_run_list.currentItem() is not None:
            items = [self.training_run_list.currentItem()]
        return [
            dict(data)
            for item in items
            if isinstance((data := item.data(Qt.ItemDataRole.UserRole)), dict)
        ]

    def compare_selected_training_runs(self) -> None:
        runs = self._selected_training_runs()
        names = services.experiment_metric_names(runs)
        current_metric = self.experiment_metric_combo.currentText()
        self.experiment_metric_combo.blockSignals(True)
        self.experiment_metric_combo.clear()
        self.experiment_metric_combo.addItems(names)
        if current_metric in names:
            self.experiment_metric_combo.setCurrentText(current_metric)
        self.experiment_metric_combo.blockSignals(False)
        try:
            comparison = services.compare_training_runs(
                runs,
                metric=self.experiment_metric_combo.currentText(),
                direction=str(self.experiment_direction_combo.currentData() or "auto"),
            )
        except ValueError as exc:
            self.experiment_comparison_summary.setText(f"实验对比：{exc}")
            return
        self.latest_experiment_comparison = comparison
        selected_metric = str(comparison["metric"])
        if self.experiment_metric_combo.findText(selected_metric) >= 0:
            self.experiment_metric_combo.setCurrentText(selected_metric)
        self.experiment_comparison_summary.setText(
            f"最佳实验：{comparison['best_run_id']} | {selected_metric}="
            f"{services.display_value(comparison['best_value'])} | "
            f"方向：{comparison['direction']} | 对比 {comparison['run_count']} 次"
        )
        rows = comparison.get("rows") if isinstance(comparison.get("rows"), list) else []
        self.experiment_comparison_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            parameter_text = json.dumps(row.get("parameters") or {}, ensure_ascii=False, sort_keys=True)
            values = [
                f"★ {row['rank']}" if row.get("best") else row.get("rank"),
                row.get("run_id"),
                services.display_value(row.get("value")),
                row.get("status"),
                parameter_text,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, dict(row))
                self.experiment_comparison_table.setItem(row_index, column, item)
        curves = comparison.get("curves") if isinstance(comparison.get("curves"), dict) else {}
        curve_steps = comparison.get("curve_steps") if isinstance(comparison.get("curve_steps"), dict) else {}
        self.experiment_comparison_curve.set_history(curves, curve_steps)
        self.experiment_comparison_curve.set_selected_metrics(list(curves))
        self.training_result_tabs.setCurrentIndex(1)

    def rerun_selected_training_run(self) -> None:
        runs = self._selected_training_runs()
        if not runs:
            return
        run = runs[0]
        row = {
            "id": f"{run.get('run_id') or run.get('preset_id') or 'experiment'} rerun",
            "label": f"{run.get('preset_label') or run.get('preset_id') or 'Experiment'} rerun",
            "training_preset_id": str(run.get("preset_id") or ""),
            "dataset_root": str(run.get("dataset_root") or ""),
            "adapter": str(run.get("adapter") or ""),
            "sequence_id": str(run.get("sequence_id") or ""),
            "split_path": str(run.get("split_snapshot_path") or run.get("split_path") or ""),
            "output_path": "",
            "parameters": dict(run.get("parameters") if isinstance(run.get("parameters"), dict) else {}),
        }
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        manifest_path = str(
            run.get("trainer_manifest_snapshot_path")
            or summary.get("trainer_manifest_path")
            or ""
        )
        trainer_root = str(Path(manifest_path).parent) if manifest_path else None
        try:
            job = services.queue_training_config_job(
                self.training_job_queue,
                row,
                trainer_root=trainer_root,
            )
        except Exception as exc:
            self.training_run_overview.setText(f"重新运行失败：{exc}")
            return
        self._current_training_job_id = job.job_id
        self._handled_training_job_ids.discard(job.job_id)
        self.training_job_timer.start()
        self.model_summary.setText(_compact_json(job.snapshot()))
        self.log(f"实验已重新入队：{job.job_id}")
        self._refresh_training_jobs()

    def mark_best_experiment(self) -> None:
        comparison = self.latest_experiment_comparison
        rows = comparison.get("rows") if isinstance(comparison.get("rows"), list) else []
        best = rows[0] if rows else {}
        path = str(best.get("path") or "")
        if not path:
            self.experiment_comparison_summary.setText("实验对比：最佳实验没有可写入的 training_run.json。")
            return
        try:
            services.mark_best_training_run(
                path,
                metric=str(comparison.get("metric") or "metric"),
                direction=str(comparison.get("direction") or "min"),
                value=float(best.get("value")),
            )
        except Exception as exc:
            self.experiment_comparison_summary.setText(f"实验对比：标记失败：{exc}")
            return
        self.log(f"已标记最佳实验: {best.get('run_id', services.NAN_TEXT)}")
        self.refresh_catalogs()

    def clone_selected_training_run(self) -> None:
        runs = self._selected_training_runs()
        if not runs:
            return
        try:
            config = services.clone_training_config_from_run(runs[0])
        except Exception as exc:
            self.training_run_overview.setText(f"复制训练配置失败：{exc}")
            return
        self.refresh_catalogs()
        self._select_training_config(str(config.get("id") or ""))
        self.data_training_tabs.setCurrentIndex(1)
        self.log(f"训练配置已复制: {config.get('label', services.NAN_TEXT)}")

    def cleanup_selected_training_runs(self) -> None:
        paths = [str(run.get("path") or "") for run in self._selected_training_runs() if run.get("path")]
        if not paths:
            return
        preview = services.cleanup_training_runs(paths, dry_run=True)
        candidates = preview.get("candidates") if isinstance(preview.get("candidates"), list) else []
        if not candidates:
            reasons = "; ".join(str(row.get("reason") or "") for row in preview.get("refused", []))
            self.training_run_overview.setText(f"没有可安全清理的实验。{reasons}")
            return
        answer = QMessageBox.question(
            self,
            "清理失败/无效实验",
            f"将删除 {len(candidates)} 个未被配置引用的失败或无效实验目录。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        result = services.cleanup_training_runs(paths, dry_run=False)
        self.log(f"已清理无效实验: {len(result.get('removed', []))}")
        self.refresh_catalogs()

    def export_experiment_comparison_report(self) -> None:
        if not self.latest_experiment_comparison:
            self.experiment_comparison_summary.setText("实验对比：请先运行对比。")
            return
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "选择实验报告目录",
            str(services.ROOT / "outputs" / "training_reports"),
        )
        if not output_dir:
            return
        payload = services.export_experiment_report(self.latest_experiment_comparison, output_dir)
        curve_path = Path(output_dir) / "experiment_curves.png"
        saved = self.experiment_comparison_curve.grab().save(str(curve_path), "PNG")
        payload["curve_path"] = str(curve_path.resolve()) if saved else services.NAN_TEXT
        self.experiment_comparison_summary.setText(
            f"报告已导出：{payload['markdown_path']} | {payload['html_path']}"
        )
        self.log(f"实验对比报告已导出: {payload['output_dir']}")

    def _set_training_run_views(self, payload: dict[str, Any]) -> dict[str, Any]:
        run = _training_run_record_from_payload(payload)
        self.latest_training_run_record = dict(run)
        if hasattr(self, "training_run_overview"):
            self.training_run_overview.setText(_training_run_overview_text(run))
        if hasattr(self, "training_run_summary"):
            self.training_run_summary.setText(_compact_json(run))
        history = services.training_metric_history(run)
        history_steps = services.training_metric_steps(run)
        diagnostics = services.training_metric_diagnostics(run)
        metric_summary = _metric_history_summary(history)
        diagnostic_summary = _metric_diagnostics_summary(diagnostics)
        if hasattr(self, "training_run_metric_summary"):
            self.training_run_metric_summary.setText(metric_summary)
        if hasattr(self, "latest_metric_summary"):
            self.latest_metric_summary.setText(metric_summary)
        if hasattr(self, "training_metric_warning_label"):
            self.training_metric_warning_label.setText(diagnostic_summary)
        if hasattr(self, "latest_metric_warning_label"):
            self.latest_metric_warning_label.setText(diagnostic_summary)
        if hasattr(self, "latest_training_log"):
            self.latest_training_log.setText(_training_log_summary(run))
        if hasattr(self, "training_curve"):
            self.training_curve.set_history(history, history_steps)
            self.training_curve.set_diagnostics(diagnostics)
            self._fill_training_metric_combo(self.training_metric_combo, self.training_curve)
        if hasattr(self, "latest_training_curve"):
            self.latest_training_curve.set_history(history, history_steps)
            self.latest_training_curve.set_diagnostics(diagnostics)
            self._fill_training_metric_combo(self.latest_metric_combo, self.latest_training_curve)
        return run

    def _set_live_training_metric_views(self, run: dict[str, Any]) -> None:
        history = services.training_metric_history(run)
        history_steps = services.training_metric_steps(run)
        if not history:
            return
        diagnostics = run.get("metric_diagnostics") if isinstance(run.get("metric_diagnostics"), dict) else {}
        self.latest_training_curve.set_history(history, history_steps)
        self.latest_training_curve.set_diagnostics(diagnostics)
        self._fill_training_metric_combo(self.latest_metric_combo, self.latest_training_curve)
        self.latest_metric_summary.setText(_metric_history_summary(history))
        self.latest_metric_warning_label.setText(_metric_diagnostics_summary(diagnostics))

    def _fill_training_metric_combo(self, combo: QComboBox, curve: TrainingCurveWidget) -> None:
        current = combo.currentText()
        names = curve.metric_names()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        selected = current if current in names else curve.primary_metric
        if selected:
            combo.setCurrentText(selected)
        combo.blockSignals(False)
        curve.set_primary_metric(selected)

    def _select_training_metric(self, curve: TrainingCurveWidget, metric: str) -> None:
        curve.set_primary_metric(metric)

    def export_selected_training_metrics(self) -> None:
        record = dict(self.latest_training_run_record)
        if not services.training_metric_history(record):
            self.training_metric_warning_label.setText("训练诊断：没有可导出的指标。")
            return
        record_path = Path(str(record.get("path") or "")) if record.get("path") else services.ROOT / "outputs"
        default_dir = record_path.parent / "metric_exports" if record_path.suffix else record_path / "metric_exports"
        selected = QFileDialog.getExistingDirectory(self, "选择指标导出目录", str(default_dir))
        if not selected:
            return
        payload = services.export_training_metrics(record, selected)
        curve_path = Path(selected) / "training_curves.png"
        saved = self.training_curve.grab().save(str(curve_path), "PNG")
        payload["curve_path"] = str(curve_path.resolve()) if saved else services.NAN_TEXT
        self.model_summary.setText(_compact_json(payload))
        self.log(f"训练指标已导出: {payload['output_dir']}")

    def _set_beamng_quality_views(self, payload: dict[str, Any]) -> None:
        if hasattr(self, "beamng_quality_report"):
            self.beamng_quality_report.setText(_beamng_quality_report_text(payload))
        run = _training_run_record_from_payload(payload)
        history = services.training_metric_history(run)
        if hasattr(self, "beamng_quality_curve"):
            self.beamng_quality_curve.set_history(history)
        plot_path = _payload_trajectory_plot_path(payload, run)
        if hasattr(self, "beamng_trajectory_plot"):
            self._set_preview(self.beamng_trajectory_plot, plot_path, "Trajectory plot: NaN")

    def _fill_episode_list(self) -> None:
        self.episode_list.clear()
        for episode in self.catalog.get("episodes", [])[:80]:
            item = QListWidgetItem(str(episode.get("episode_id", services.NAN_TEXT)))
            item.setData(Qt.ItemDataRole.UserRole, episode.get("path"))
            self.episode_list.addItem(item)

    def _refresh_planner_summary(self) -> None:
        if not hasattr(self, "planner_summary"):
            return
        payload = {
            "planner": self.planner_combo.currentData() or self.planner_combo.currentText() or None,
            "horizon": self.settings.planner_horizon,
            "samples": self.settings.planner_samples,
            "iterations": self.settings.planner_iterations,
            "image_size": self.settings.image_size,
            "max_steps": self.settings.max_steps,
            "record": self.settings.record,
            "load_assets": self.settings.load_assets,
        }
        self.planner_summary.setText(_compact_json(payload))

    def _refresh_demo_status(self) -> None:
        if not hasattr(self, "demo_result_summary"):
            return
        beamng = _find_named(self.catalog.get("backends", []), "beamng") if self.catalog else None
        demo = self._combo_config_row(self.demo_config_combo)
        world_model_config_id = str(demo.get("world_model_config_id") or services.DEFAULT_WORLD_MODEL_CONFIG_ID)
        model_config = next(
            (row for row in self.catalog.get("world_model_configs", []) if str(row.get("id") or "") == world_model_config_id),
            {},
        )
        payload = {
            "demo_config": demo.get("label") or services.NAN_TEXT,
            "beamng_available": beamng.get("available") if beamng else None,
            "region_task": demo.get("task_relative_path") or demo.get("task_path") or services.NAN_TEXT,
            "world_model_config": model_config.get("label") or services.NAN_TEXT,
            "planner": demo.get("planner") or services.NAN_TEXT,
            "last_result": services.NAN_TEXT,
        }
        self.demo_result_summary.setText(_compact_json(payload))

    def _current_request(self) -> services.RunRequest:
        config = self._combo_config_row(self.world_model_config_combo)
        return services.RunRequest(
            backend=self.backend_combo.currentData() or self.backend_combo.currentText(),
            scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText(),
            agent=self.agent_combo.currentData() or self.agent_combo.currentText(),
            seed=self.settings.seed,
            max_steps=self.settings.max_steps,
            record=self.settings.record,
            record_arrays=self.settings.record_arrays,
            world_model_type=config.get("world_model") or self.world_model_combo.currentData() or self.world_model_combo.currentText(),
            world_model_path=str(config.get("model_path") or self.model_path_edit.text().strip()),
            planner=self.planner_combo.currentData() or "",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            dataset_root=self.dataset_root_edit.text().strip(),
            sequence_id=self.sequence_combo.currentText().strip(),
            adapter=self.adapter_edit.text().strip(),
            load_assets=self.settings.load_assets,
        )

    def _run_task(
        self,
        fn: Callable[[], Any],
        on_success: Callable[[Any], None],
        failure_label: str,
        *,
        task_label: str = "",
        cancel_hook: Callable[[], None] | None = None,
    ) -> None:
        worker = TaskWorker(fn, cancel_hook=cancel_hook)
        thread = threading.Thread(
            target=worker.run,
            name=f"desktop-task-{len(self.threads) + 1}",
            daemon=True,
        )
        self._set_busy(True, task_label or _task_label_from_failure(failure_label))
        worker.finished.connect(lambda result: None if worker.is_canceled() else on_success(result))
        worker.failed.connect(
            lambda message: None if worker.is_canceled() else self._task_failed(failure_label, message)
        )
        worker.finished.connect(lambda _: None if worker.is_canceled() else self._set_busy(False))
        worker.failed.connect(lambda _: None if worker.is_canceled() else self._set_busy(False))
        worker.settled.connect(lambda: self._remove_thread(thread, worker))
        self.threads.append(thread)
        self.workers.append(worker)
        thread.start()

    def _set_busy(self, active: bool, label: str = "") -> None:
        if active:
            self._busy_depth += 1
            self.busy_label.setText(f"正在执行：{label}" if label else "正在执行")
            self.busy_bar.setRange(0, 0)
            self.busy_label.setVisible(True)
            self.busy_bar.setVisible(True)
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            return
        self._busy_depth = max(0, self._busy_depth - 1)
        if self._busy_depth > 0:
            return
        self.busy_label.setVisible(False)
        self.busy_bar.setVisible(False)
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def _task_failed(self, failure_label: str, message: str) -> None:
        self.log(f"{failure_label}: {message}")
        if failure_label == "数据集预览失败":
            self._dataset_preview_busy = False
            self.dataset_play_button.setChecked(False)
        if failure_label == "navigation realtime preview failed":
            self.beamng_summary.setText(_compact_json({"status": "preview_failed", "message": message}))
            self._finish_navigation_preview_task()

    def _remove_thread(self, thread: threading.Thread, worker: TaskWorker) -> None:
        if thread in self.threads:
            self.threads.remove(thread)
        if worker in self.workers:
            self.workers.remove(worker)
        if thread.name in self.detached_task_names:
            self.detached_task_names.remove(thread.name)

    def detached_task_entries(self) -> list[dict[str, Any]]:
        return [
            {"name": thread.name, "alive": thread.is_alive(), "daemon": thread.daemon}
            for thread in self.threads
        ]

    def log(self, text: str) -> None:
        self.log_view.append(text)

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(PAGE_SPACING)
        return page, layout

    def _row_layout(self, *, spacing: int = PAGE_SPACING) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        return layout

    def _new_group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(*CARD_MARGINS)
        layout.setSpacing(CARD_SPACING)
        return group, layout

    def _configure_control(self, widget: QWidget) -> None:
        widget.setMinimumHeight(CONTROL_HEIGHT)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _configure_button(self, button: QPushButton, *, primary: bool = False, height: int = BUTTON_HEIGHT) -> None:
        if primary:
            button.setObjectName("primaryButton")
            height = max(height, PRIMARY_BUTTON_HEIGHT)
        button.setMinimumHeight(height)
        button.setCursor(Qt.CursorShape.PointingHandCursor)

    def _combo(self, *, editable: bool = False) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(editable)
        self._configure_control(combo)
        if editable and combo.lineEdit() is not None:
            self._configure_control(combo.lineEdit())
        return combo

    def _fill_combo(self, combo: QComboBox, rows: list[dict[str, Any]], key: str, *, default: str = "") -> None:
        current = combo.currentData() or combo.currentText()
        combo.clear()
        selected_index = 0
        for row in rows:
            value = str(row.get(key, ""))
            label = value or "none"
            if row.get("available") is False:
                label = f"{label} ({row.get('message', services.UNFINISHED_TEXT)})"
            combo.addItem(label, value)
            if value == (current or default):
                selected_index = combo.count() - 1
        combo.setCurrentIndex(selected_index)

    def _fill_demo_config_combo(self) -> None:
        current = self.demo_config_combo.currentData()
        current_id = current.get("id") if isinstance(current, dict) else ""
        self.demo_config_combo.blockSignals(True)
        self.demo_config_combo.clear()
        selected_index = 0
        for row in self.catalog.get("demo_configs", []):
            config_id = str(row.get("id") or "")
            if not config_id:
                continue
            label = str(row.get("label") or config_id)
            self.demo_config_combo.addItem(label, dict(row))
            if config_id == (current_id or services.DEFAULT_DEMO_CONFIG_ID):
                selected_index = self.demo_config_combo.count() - 1
        if self.demo_config_combo.count():
            self.demo_config_combo.setCurrentIndex(selected_index)
        self.demo_config_combo.blockSignals(False)

    def _fill_world_model_config_combo(self, combo: QComboBox, rows: list[dict[str, Any]], *, default_id: str) -> None:
        current = combo.currentData()
        current_id = current.get("id") if isinstance(current, dict) else ""
        combo.blockSignals(True)
        combo.clear()
        selected_index = -1
        for row in rows:
            config_id = str(row.get("id") or "")
            if not config_id:
                continue
            label = str(row.get("label") or config_id)
            combo.addItem(label, dict(row))
            if config_id == (current_id or default_id):
                selected_index = combo.count() - 1
        if combo.count():
            combo.setCurrentIndex(max(0, selected_index))
        combo.blockSignals(False)

    def _fill_path_combo(self, combo: QComboBox, rows: list[dict[str, Any]], *, default_path: str) -> None:
        current = self._path_combo_value(combo) or default_path
        combo.blockSignals(True)
        combo.clear()
        selected_index = -1
        for row in rows:
            path = str(row.get("path") or "")
            if not path:
                continue
            label = str(row.get("label") or row.get("relative_path") or row.get("id") or path)
            combo.addItem(label, path)
            if _same_path_text(path, current):
                selected_index = combo.count() - 1
        if default_path and selected_index < 0:
            existing = next((index for index in range(combo.count()) if _same_path_text(str(combo.itemData(index)), default_path)), -1)
            if existing >= 0:
                selected_index = existing
            else:
                combo.addItem(default_path, default_path)
                selected_index = combo.count() - 1
        if combo.count():
            combo.setCurrentIndex(max(0, selected_index))
        combo.blockSignals(False)
        if combo is self.home_task_combo:
            self._sync_home_task_to_edit()
        elif combo is self.beamng_task_combo:
            self._sync_beamng_task_to_edit()
        elif combo is self.home_model_combo:
            self._sync_home_model_to_edit()

    def _path_combo_value(self, combo: QComboBox) -> str:
        text = combo.currentText().strip()
        index = combo.currentIndex()
        if index >= 0 and text == combo.itemText(index):
            data = combo.itemData(index)
            return str(data or text).strip()
        return text

    def _sync_home_task_to_edit(self) -> None:
        path = self._path_combo_value(self.home_task_combo)
        if path:
            self.task_path_edit.setText(path)

    def _sync_beamng_task_to_edit(self) -> None:
        path = self._path_combo_value(self.beamng_task_combo)
        if path:
            self.task_path_edit.setText(path)

    def _sync_home_model_to_edit(self) -> None:
        path = self._path_combo_value(self.home_model_combo)
        if path:
            self.model_path_edit.setText(path)

    def _combo_config_row(self, combo: QComboBox) -> dict[str, Any]:
        data = combo.currentData()
        return dict(data) if isinstance(data, dict) else {}

    def _sync_home_world_model_config(self) -> None:
        row = self._combo_config_row(self.world_model_config_combo)
        if row:
            self._apply_world_model_config(row, sync_editor=True)

    def _sync_edit_world_model_config(self) -> None:
        row = self._combo_config_row(self.world_model_config_edit_combo)
        if row:
            self._apply_world_model_config(row, sync_editor=False)
            self._select_world_model_config(
                str(row.get("id") or ""),
                combos=[self.world_model_config_combo, self.beamng_model_config_combo],
            )

    def _sync_beamng_world_model_config(self) -> None:
        row = self._combo_config_row(self.beamng_model_config_combo)
        if row:
            self._apply_world_model_config(row, sync_editor=True)
            self._select_world_model_config(str(row.get("id") or ""), combos=[self.world_model_config_combo])

    def _apply_world_model_config(self, row: dict[str, Any], *, sync_editor: bool) -> None:
        if hasattr(self, "model_config_name_edit"):
            self.model_config_name_edit.setText(str(row.get("label") or row.get("id") or ""))
        algorithm = str(row.get("algorithm") or "")
        world_model = str(row.get("world_model") or "")
        model_path = str(row.get("model_path") or "")
        if algorithm:
            self._select_combo_value(self.algorithm_combo, algorithm)
        if world_model:
            self._select_combo_value(self.world_model_combo, world_model)
        if model_path:
            self._select_path_combo_value(self.home_model_combo, model_path)
            self.model_path_edit.setText(model_path)
        if sync_editor and hasattr(self, "world_model_config_edit_combo"):
            self._select_world_model_config(
                str(row.get("id") or ""),
                combos=[self.world_model_config_edit_combo, self.beamng_model_config_combo],
            )

    def _select_path_combo_value(self, combo: QComboBox, path: str) -> None:
        for index in range(combo.count()):
            data = combo.itemData(index)
            if _same_path_text(str(data or ""), path):
                combo.setCurrentIndex(index)
                return
        combo.setCurrentText(path)

    def _select_world_model_config(self, config_id: str, *, combos: list[QComboBox] | None = None) -> None:
        for combo in combos or [self.world_model_config_combo, self.world_model_config_edit_combo, self.beamng_model_config_combo]:
            for index in range(combo.count()):
                data = combo.itemData(index)
                if isinstance(data, dict) and str(data.get("id") or "") == config_id:
                    combo.setCurrentIndex(index)
                    break

    def _select_combo_value(self, combo: QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value or combo.itemText(index) == value:
                combo.setCurrentIndex(index)
                return

    def _group(self, title: str, widgets: list[QWidget]) -> QGroupBox:
        group, layout = self._new_group(title)
        for widget in widgets:
            layout.addWidget(widget)
        return group

    def _field(self, label: str, widget: QWidget) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(FIELD_SPACING)
        caption = QLabel(label)
        caption.setObjectName("fieldLabel")
        caption.setFixedHeight(18)
        if isinstance(widget, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            self._configure_control(widget)
        layout.addWidget(caption)
        layout.addWidget(widget)
        return frame

    def _compact_field(self, label: str, widget: QWidget) -> QWidget:
        frame = QWidget()
        frame.setObjectName("compactField")
        frame.setMaximumHeight(58)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        caption = QLabel(label)
        caption.setObjectName("fieldLabel")
        caption.setFixedWidth(128)
        caption.setWordWrap(False)
        if isinstance(widget, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            self._configure_control(widget)
        layout.addWidget(caption)
        layout.addWidget(widget, 1)
        return frame

    def _with_button(self, widget: QWidget, button: QPushButton) -> QWidget:
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        if isinstance(widget, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            self._configure_control(widget)
        self._configure_button(button)
        layout.addWidget(widget, 1)
        layout.addWidget(button)
        return frame

    def _action_button(self, text: str, slot: Callable[[], None], *, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        self._configure_button(button, primary=primary)
        button.clicked.connect(slot)
        return button

    def _action_toolbar(self, buttons: list[QPushButton], *, object_name: str, columns: int = 2) -> QWidget:
        frame = QWidget()
        frame.setObjectName(object_name)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QGridLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        for index, button in enumerate(buttons):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(button, index // columns, index % columns)
        return frame

    def _tab_header(self, title: str, subtitle: str) -> QWidget:
        frame = QWidget()
        frame.setObjectName("tabHeader")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("tabTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("mutedText")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return frame

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        label.setFixedHeight(20)
        return label

    def _preview_panel(self, title: str, preview: QLabel) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._section_label(title))
        layout.addWidget(preview, 1)
        return frame

    def _preview_label(self, placeholder: str) -> StablePreviewLabel:
        return StablePreviewLabel(placeholder)

    def _set_preview(self, label: QLabel, path: Any, placeholder: str) -> None:
        if not path or not Path(str(path)).exists():
            if isinstance(label, StablePreviewLabel):
                label.clear_preview(placeholder)
            else:
                label.setPixmap(QPixmap())
                label.setText(placeholder)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            if isinstance(label, StablePreviewLabel):
                label.clear_preview(placeholder)
            else:
                label.setText(placeholder)
            return
        if isinstance(label, StablePreviewLabel):
            label.set_preview_pixmap(pixmap)
        else:
            label.setText("")
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", target.text() or str(services.ROOT))
        if path:
            target.setText(path)

    def _browse_path_or_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", target.text() or str(services.ROOT))
        if path:
            target.setText(path)

    def _browse_file(self, target: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, target.text() or str(services.ROOT), "Python/scripts (*.py *.bat *.cmd *.exe);;All files (*)")
        if path:
            target.setText(path)

    def _browse_any_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",
            target.text() or str(services.ROOT),
            "All files (*)",
        )
        if path:
            target.setText(path)

    def _browse_path_combo(self, target: QComboBox) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select directory", self._path_combo_value(target) or str(services.ROOT))
        if path:
            target.setCurrentText(path)
            if target is self.home_model_combo:
                self._sync_home_model_to_edit()


def _find_named(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def _dataset_detail_text(payload: dict[str, Any]) -> str:
    details = payload.get("details") if isinstance(payload.get("details"), dict) else payload
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    selected_scope = details.get("analysis_scope") == "selected_sequence"
    sample_count = details.get(
        "selected_sequence_frame_count" if selected_scope else "sample_count",
        payload.get("frame_count", services.NAN_TEXT),
    )
    sequence_count = details.get(
        "dataset_sequence_count" if selected_scope else "sequence_count",
        len(payload.get("sequences", [])),
    )
    lines = [
        f"数据集：{payload.get('dataset_id', details.get('dataset_root', services.NAN_TEXT))}",
        f"适配器：{payload.get('adapter', services.NAN_TEXT)}",
        f"统计范围：{'当前序列 ' + str(details.get('selected_sequence_id', '')) if selected_scope else '完整数据集'}",
        f"序列数：{services.display_value(sequence_count)}",
        f"{'当前序列样本数' if selected_scope else '样本总数'}：{services.display_value(sample_count)}",
        f"模态：{', '.join(details.get('modalities', [])) or services.NAN_TEXT}",
        f"分辨率：{json.dumps(details.get('resolutions', {}), ensure_ascii=False)}",
        f"时间范围：{services.display_value(details.get('time_start'))} - {services.display_value(details.get('time_end'))}",
        f"时长：{services.display_value(details.get('duration_sec'))} s",
        f"数据集大小：{_format_bytes(details.get('dataset_disk_usage_bytes'))}",
        f"文件数：{services.display_value(details.get('dataset_file_count'))}{'（扫描未完成）' if details.get('disk_usage_truncated') else ''}",
        f"已引用资产大小：{_format_bytes(details.get('referenced_disk_usage_bytes'))}",
    ]
    if quality:
        lines.extend(
            [
                "",
                f"快速检查：{quality.get('status', services.NAN_TEXT)}",
                f"可用于训练：{services.display_value(quality.get('training_ready'))}",
                f"错误 / 警告：{quality.get('error_count', 0)} / {quality.get('warning_count', 0)}",
            ]
        )
    return "\n".join(lines)


def _dataset_quality_text(
    analysis: dict[str, Any],
    *,
    report_json: str = "",
    report_markdown: str = "",
) -> str:
    if not analysis:
        return "质量报告：NaN"
    lines = [
        f"状态：{analysis.get('status', services.NAN_TEXT)}",
        f"可用于训练：{services.display_value(analysis.get('training_ready'))}",
        f"序列 / 样本：{analysis.get('sequence_count', 0)} / {analysis.get('sample_count', 0)}",
        f"模态：{', '.join(analysis.get('modalities', [])) or services.NAN_TEXT}",
        f"检查资产：{analysis.get('checked_asset_count', 0)} / {analysis.get('available_asset_count', 0)}",
        f"检查方式：{'全量' if analysis.get('asset_check_mode') == 'full' else '抽样'}",
        f"未检查资产：{analysis.get('unchecked_asset_count', 0)}",
        f"损坏资产：{analysis.get('corrupt_asset_count', 0)}",
        f"缺失资产：{json.dumps(analysis.get('missing_asset_counts', {}), ensure_ascii=False)}",
        f"错误 / 警告：{analysis.get('error_count', 0)} / {analysis.get('warning_count', 0)}",
    ]
    if report_json:
        lines.append(f"JSON 报告：{report_json}")
    if report_markdown:
        lines.append(f"Markdown 报告：{report_markdown}")
    issues = analysis.get("issues") if isinstance(analysis.get("issues"), list) else []
    if issues:
        lines.append("")
        lines.append("问题：")
        for issue in issues[:30]:
            lines.append(
                f"- [{str(issue.get('severity', 'warning')).upper()}] {issue.get('code', '')}: {issue.get('message', '')}"
            )
        if len(issues) > 30 or analysis.get("issues_truncated"):
            lines.append("- 其余问题请查看报告文件。")
    return "\n".join(lines)


def _dataset_split_text(payload: dict[str, Any]) -> str:
    if not payload:
        return "数据划分：NaN"
    ratios = payload.get("ratios") if isinstance(payload.get("ratios"), dict) else {}
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    return "\n".join(
        [
            f"划分单位：{payload.get('split_unit', services.NAN_TEXT)}",
            f"策略：{payload.get('strategy', services.NAN_TEXT)}",
            f"比例：train={ratios.get('train', services.NAN_TEXT)}, validation={ratios.get('validation', services.NAN_TEXT)}, test={ratios.get('test', services.NAN_TEXT)}",
            f"样本数：train={counts.get('train', 0)}, validation={counts.get('validation', 0)}, test={counts.get('test', 0)}",
            f"随机种子：{payload.get('seed', services.NAN_TEXT)}（{'已应用' if payload.get('seed_applied') else '连续帧划分，不参与'}）",
            f"文件：{payload.get('path', services.NAN_TEXT)}",
        ]
    )


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return services.NAN_TEXT
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return services.NAN_TEXT


def _same_path_text(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def _task_label_from_failure(failure_label: str) -> str:
    text = str(failure_label or "").strip()
    lowered = text.lower()
    if lowered.endswith(" failed"):
        return text[:-7].strip() or text
    if text.endswith("失败"):
        return text[:-2].strip() or text
    return text or "任务"


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first_goal(trace: list[dict[str, Any]]) -> tuple[float, float] | None:
    for row in trace:
        goal = row.get("goal")
        if isinstance(goal, list) and len(goal) >= 2 and _is_finite(goal[0]) and _is_finite(goal[1]):
            return float(goal[0]), float(goal[1])
    return None


def _validation_float(validation: dict[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        value = float(validation.get(key))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _number_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _inference_prediction_rows(predictions: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rows = [row if isinstance(row, dict) else {"sample": index, "value": row} for index, row in enumerate(predictions)]
    coordinate_rows = any(isinstance(row.get("predicted"), dict) and isinstance(row.get("actual"), dict) for row in rows)
    if coordinate_rows:
        columns = ["sample", "predicted.x", "predicted.y", "actual.x", "actual.y", "position_error"]
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            predicted = row.get("predicted") if isinstance(row.get("predicted"), dict) else {}
            actual = row.get("actual") if isinstance(row.get("actual"), dict) else {}
            px = _number_or_nan(predicted.get("x"))
            py = _number_or_nan(predicted.get("y"))
            ax = _number_or_nan(actual.get("x"))
            ay = _number_or_nan(actual.get("y"))
            error = math.hypot(px - ax, py - ay) if all(math.isfinite(value) for value in (px, py, ax, ay)) else math.nan
            normalized.append(
                {
                    "sample": row.get("sample", index),
                    "predicted.x": px,
                    "predicted.y": py,
                    "actual.x": ax,
                    "actual.y": ay,
                    "position_error": error,
                }
            )
        return columns, normalized

    flattened = [_flatten_prediction_row(row) for row in rows]
    names = {key for row in flattened for key in row}
    preferred = [key for key in ("sample", "frame_id", "index") if key in names]
    columns = preferred + sorted(name for name in names if name not in preferred)[: max(0, 8 - len(preferred))]
    return columns or ["sample"], flattened


def _flatten_prediction_row(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_prediction_row(value, name))
        elif not isinstance(value, (list, tuple, dict)):
            flattened[name] = value
    return flattened


def _validation_int(validation: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(validation.get(key))
    except (TypeError, ValueError):
        return int(default)


def _compact_json(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _mapping_preview(payload: dict[str, Any], *, limit: int = 4) -> str:
    if not payload:
        return services.NAN_TEXT
    items = list(payload.items())
    preview = ", ".join(f"{key}={services.display_value(value)}" for key, value in items[:limit])
    remaining = len(items) - limit
    return preview + (f" (+{remaining})" if remaining > 0 else "")


def _metric_history_summary(history: dict[str, list[float]]) -> str:
    if not history:
        return "Metric curves: NaN"
    parts = [f"{key} ({len(values)} pts)" for key, values in sorted(history.items()) if values]
    return "Metric curves: " + (", ".join(parts) if parts else services.NAN_TEXT)


def _metric_diagnostics_summary(diagnostics: dict[str, Any]) -> str:
    status = str(diagnostics.get("status") or services.NAN_TEXT)
    warnings = diagnostics.get("warnings") if isinstance(diagnostics.get("warnings"), list) else []
    if not warnings:
        resources = diagnostics.get("resource_metrics") if isinstance(diagnostics.get("resource_metrics"), list) else []
        resource_text = f" | resources: {', '.join(resources)}" if resources else ""
        return f"Training diagnostics: {status}{resource_text}"
    messages = [str(row.get("message") or "") for row in warnings if isinstance(row, dict)]
    return f"Training diagnostics: {status} | " + " | ".join(message for message in messages if message)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _demo_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {}
    metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    return {
        "status": payload.get("status") or services.NAN_TEXT,
        "goal_success": acceptance.get("goal_success", services.NAN_TEXT),
        "goal_reached": acceptance.get("goal_reached", services.NAN_TEXT),
        "collision_count": acceptance.get("collision_count", metrics.get("collision_count", services.NAN_TEXT)),
        "final_distance": acceptance.get("final_goal_distance", services.NAN_TEXT),
        "steps": metrics.get("steps", payload.get("steps", services.NAN_TEXT)),
        "episode_path": evaluation.get("episode_path") or payload.get("episode_path") or services.NAN_TEXT,
        "summary_path": payload.get("summary_path") or services.NAN_TEXT,
    }


def _training_run_overview_text(run: dict[str, Any]) -> str:
    run_id = str(run.get("run_id") or services.NAN_TEXT)
    pretty_run_id = run_id.replace("_", " ").replace("-", " ").title() if run_id and run_id != services.NAN_TEXT else services.NAN_TEXT
    lines = [
        f"Run: {pretty_run_id}",
        f"Preset: {run.get('preset_label') or run.get('preset_id') or services.NAN_TEXT}",
        f"Status: {run.get('status') or services.NAN_TEXT}",
    ]
    artifact = str(run.get("artifact_path") or run.get("relative_path") or "").strip()
    lines.append(f"artifact: {artifact or services.NAN_TEXT}")
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    promoted_config = summary.get("world_model_config") if isinstance(summary.get("world_model_config"), dict) else {}
    if promoted_config:
        lines.append(f"world_model_config: {promoted_config.get('id') or services.NAN_TEXT}")
        config_model_path = str(promoted_config.get("model_path") or "").strip()
        if config_model_path:
            lines.append(f"config_model_path: {config_model_path}")
        validation = promoted_config.get("validation") if isinstance(promoted_config.get("validation"), dict) else {}
        if "goal_success" in validation:
            lines.append(f"config_goal_success: {services.display_value(validation['goal_success'])}")
    diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else {}
    if diagnostics:
        if diagnostics.get("status"):
            lines.append(f"diagnostic_status: {diagnostics['status']}")
        if diagnostics.get("message"):
            lines.append(f"diagnostic_message: {diagnostics['message']}")
        next_actions = diagnostics.get("next_actions") if isinstance(diagnostics.get("next_actions"), list) else []
        if next_actions:
            lines.append(f"diagnostic_next: {next_actions[0]}")
    logs = run.get("logs") if isinstance(run.get("logs"), dict) else {}
    for key in ("stdout", "stderr"):
        path = str(logs.get(key) or "").strip()
        if path:
            lines.append(f"{key}: {path}")
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    flat_metrics = _flatten_numeric_mapping(metrics)
    history = services.training_metric_history(run)
    metric_lines: list[str] = []
    for key in [
        "loss",
        "final_loss",
        "validation_rmse",
        "validation_mse",
        "validation_sample_count",
        "segment_rmse.goal",
        "segment_rmse.middle",
        "segment_rmse.start",
        "train_rmse",
        "train_mse",
        "goal_success",
        "min_goal_distance",
        "collection_min_goal_distance",
        "collection_distance_traveled",
    ]:
        if key in metrics:
            metric_lines.append(f"{key}: {services.display_value(metrics[key])}")
        elif key in flat_metrics:
            metric_lines.append(f"{key}: {services.display_value(flat_metrics[key])}")
        elif key in history and history[key]:
            metric_lines.append(f"{key}: {services.display_value(history[key][-1])}")
    if metric_lines:
        lines.extend(metric_lines[:6])
    else:
        lines.append(f"metrics: {services.NAN_TEXT}")
    return "\n".join(lines)


def _flatten_numeric_mapping(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            values.update(_flatten_numeric_mapping(value, name))
            continue
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values[name] = value
    return values


def _training_log_summary(run: dict[str, Any]) -> str:
    logs = run.get("logs") if isinstance(run.get("logs"), dict) else {}
    lines: list[str] = []
    for key in ("stdout", "stderr"):
        path = str(logs.get(key) or "").strip()
        if path:
            lines.append(f"{key}: {path}")
    return "\n".join(lines) if lines else f"Training logs: {services.NAN_TEXT}"


def _region_world_model_summary_text(payload: dict[str, Any]) -> str:
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    region = payload.get("region_navigation") if isinstance(payload.get("region_navigation"), dict) else {}
    quality = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    if not acceptance and not region:
        return ""

    lines = ["Region world-model evaluation"]
    for key in [
        "goal_success",
        "goal_reached",
        "final_goal_reached",
        "model_controlled",
        "min_goal_distance",
        "final_goal_distance",
        "collision_count",
        "distance_traveled",
        "stuck_recovery_count",
        "reverse_count",
        "max_collision_count",
    ]:
        if key in acceptance:
            lines.append(f"{key}: {services.display_value(acceptance[key])}")
    for key in ["evaluation_agent", "evaluation_route_mode", "route_free", "planner", "algorithm_model_path"]:
        if key in region:
            lines.append(f"{key}: {services.display_value(region[key])}")
    if quality:
        if "passed" in quality:
            lines.append(f"quality_gate_passed: {services.display_value(quality['passed'])}")
        if "progress_ratio" in quality:
            lines.append(f"quality_progress_ratio: {services.display_value(quality['progress_ratio'])}")
        if quality.get("reason"):
            lines.append(f"quality_reason: {quality['reason']}")
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    if comparison:
        for key in [
            "route_free_goal_success",
            "route_free_min_goal_distance",
            "route_free_final_goal_distance",
            "route_free_collision_count",
            "route_free_distance_traveled",
            "route_free_stuck_recovery_count",
            "route_free_reverse_count",
            "route_guided_goal_success",
            "route_guided_min_goal_distance",
            "route_guided_final_goal_distance",
            "route_guided_collision_count",
            "route_guided_distance_traveled",
        ]:
            if key in comparison:
                lines.append(f"{key}: {services.display_value(comparison[key])}")
    trajectory_plot_path = str(payload.get("trajectory_plot_path") or "").strip()
    if trajectory_plot_path:
        lines.append(f"trajectory_plot_path: {trajectory_plot_path}")
    for key in ["model_dir", "training_run_path", "summary_path"]:
        value = str(payload.get(key) or "").strip()
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _payload_trajectory_plot_path(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        direct = str(payload.get("trajectory_plot_path") or "").strip()
        if direct:
            return direct
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        summary_path = str(summary.get("trajectory_plot_path") or "").strip()
        if summary_path:
            return summary_path
    return ""


def _first_nested_dict(payloads: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for payload in payloads:
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, dict) and value:
            return value
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        value = summary.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _beamng_quality_report_text(payload: dict[str, Any]) -> str:
    run = _training_run_record_from_payload(payload)
    payloads = [payload, run]
    lines = ["BeamNG training quality report"]

    label = str(payload.get("preset_label") or run.get("preset_label") or run.get("preset_id") or "").strip()
    status = str(payload.get("status") or run.get("status") or "").strip()
    if label:
        lines.append(f"workflow: {label}")
    if status:
        lines.append(f"status: {status}")

    quality = _first_nested_dict(payloads, "quality_gate")
    if quality:
        for key in [
            "passed",
            "reason",
            "progress_ratio",
            "required_progress_ratio",
            "route_coverage_ratio",
            "goal_zone_coverage",
            "collection_min_goal_distance",
            "unique_region_cells",
        ]:
            if key in quality:
                lines.append(f"quality_gate.{key}: {services.display_value(quality[key])}")

    merged_metrics: dict[str, Any] = {}
    for source in payloads:
        metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
        merged_metrics.update(metrics)
        training = source.get("training") if isinstance(source.get("training"), dict) else {}
        training_metrics = training.get("metrics") if isinstance(training.get("metrics"), dict) else {}
        merged_metrics.update(training_metrics)
        summary = source.get("summary") if isinstance(source.get("summary"), dict) else {}
        summary_metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        merged_metrics.update(summary_metrics)
    for key in [
        "route_coverage_ratio",
        "goal_zone_coverage",
        "collection_min_goal_distance",
        "unique_region_cells",
        "train_rmse",
        "validation_rmse",
        "validation_sample_count",
    ]:
        if key in merged_metrics:
            lines.append(f"{key}: {services.display_value(merged_metrics[key])}")
    segment_rmse = merged_metrics.get("segment_rmse")
    if isinstance(segment_rmse, dict):
        for segment in ["start", "middle", "goal"]:
            if segment in segment_rmse:
                lines.append(f"segment_rmse.{segment}: {services.display_value(segment_rmse[segment])}")

    comparison = _first_nested_dict(payloads, "comparison")
    if comparison:
        for key in [
            "route_free_goal_success",
            "route_free_min_goal_distance",
            "route_free_final_goal_distance",
            "route_free_collision_count",
            "route_free_distance_traveled",
            "route_free_stuck_recovery_count",
            "route_free_reverse_count",
            "route_guided_goal_success",
            "route_guided_min_goal_distance",
            "route_guided_final_goal_distance",
            "route_guided_collision_count",
        ]:
            if key in comparison:
                lines.append(f"{key}: {services.display_value(comparison[key])}")

    acceptance = _first_nested_dict(payloads, "acceptance")
    if acceptance:
        for key in ["goal_success", "min_goal_distance", "final_goal_distance", "collision_count", "stuck_recovery_count", "reverse_count"]:
            if key in acceptance:
                lines.append(f"acceptance.{key}: {services.display_value(acceptance[key])}")

    trajectory_plot_path = _payload_trajectory_plot_path(payload, run)
    paths = {
        "collection_manifest_path": payload.get("collection_manifest_path") or run.get("collection_manifest_path"),
        "model_dir": payload.get("model_dir") or run.get("model_dir") or run.get("artifact_path"),
        "training_run_path": payload.get("training_run_path") or run.get("path"),
        "trajectory_plot_path": trajectory_plot_path,
    }
    for key, value in paths.items():
        text = str(value or "").strip()
        if text:
            lines.append(f"{key}: {text}")

    return "\n".join(lines)


def _training_run_record_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("training_run_path") or payload.get("path") or "").strip()
    if path:
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        if isinstance(record, dict) and record:
            record["path"] = str(Path(path).resolve())
            return record
    return dict(payload)


def run() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


STYLESHEET = """
QWidget {
    background: #f5f5f7;
    color: #1d1d1f;
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 13px;
}
#sidebar {
    background: #ffffff;
    border-right: 1px solid #d2d2d7;
}
#appTitle {
    font-size: 22px;
    font-weight: 700;
    color: #1d1d1f;
}
#pageTitle {
    font-size: 24px;
    font-weight: 700;
    color: #1d1d1f;
}
#mutedText, .mutedText {
    color: #6e6e73;
}
#navButton {
    text-align: left;
    padding: 0 12px;
    border-radius: 8px;
    background: transparent;
    border: 1px solid transparent;
    color: #1d1d1f;
}
#navButton:hover {
    background: #f2f2f7;
    border-color: #e5e5ea;
}
#navButton:checked {
    background: #eaf4ff;
    border-color: #007aff;
    color: #0066cc;
}
QGroupBox {
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 4px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #6e6e73;
    background: #ffffff;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #007aff;
    selection-color: #ffffff;
    color: #1d1d1f;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #c7c7cc;
    border-radius: 8px;
    padding: 6px 12px;
    color: #1d1d1f;
}
QPushButton:hover {
    background: #f2f2f7;
    border-color: #a8a8ad;
}
QPushButton#modeButton {
    min-height: 24px;
}
QPushButton#modeButton:checked {
    background: #007aff;
    border-color: #007aff;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#modeButton:checked:hover {
    background: #0a84ff;
}
QPushButton:disabled {
    color: #a1a1a6;
    background: #f2f2f7;
    border-color: #e5e5ea;
}
#primaryButton {
    background: #007aff;
    color: #ffffff;
    font-weight: 700;
    border-color: #007aff;
}
#primaryButton:hover {
    background: #0a84ff;
}
#fieldLabel {
    color: #6e6e73;
}
#sectionLabel {
    color: #1d1d1f;
    font-weight: 700;
}
#tabHeader {
    background: transparent;
}
#tabTitle {
    color: #1d1d1f;
    font-size: 18px;
    font-weight: 700;
}
QTabWidget::pane {
    border-top: 1px solid #d2d2d7;
    top: -1px;
}
QTabBar::tab {
    background: #ffffff;
    color: #3a3a3c;
    border: 1px solid #d2d2d7;
    border-bottom-color: #d2d2d7;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 14px;
    margin-right: 4px;
    min-width: 86px;
}
QTabBar::tab:hover {
    background: #f2f2f7;
    color: #1d1d1f;
}
QTabBar::tab:selected {
    background: #007aff;
    color: #ffffff;
    border-color: #007aff;
    font-weight: 700;
}
#busyLabel {
    color: #0066cc;
    font-weight: 600;
}
QProgressBar#busyBar {
    background: #e5e5ea;
    border: 1px solid #d2d2d7;
    border-radius: 4px;
    max-height: 8px;
}
QProgressBar#busyBar::chunk {
    background: #007aff;
    border-radius: 3px;
}
#metricCard, #todoCard, #previewPane {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
}
#metricTitle {
    color: #6e6e73;
}
#metricValue {
    font-size: 24px;
    font-weight: 700;
    color: #1d1d1f;
}
#todoStatus {
    color: #b26a00;
    font-weight: 700;
}
QTextEdit, QListWidget, QTableWidget {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 8px;
    selection-background-color: #007aff;
    selection-color: #ffffff;
    color: #1d1d1f;
}
QListWidget::item {
    min-height: 28px;
    padding: 4px 6px;
}
QListWidget::item:selected {
    background: #eaf4ff;
    color: #0066cc;
}
QHeaderView::section {
    background: #f2f2f7;
    color: #3a3a3c;
    border: 0;
    padding: 7px;
}
QSplitter::handle {
    background: #f5f5f7;
}
QSplitter::handle:horizontal {
    width: 10px;
}
QSplitter::handle:vertical {
    height: 10px;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #f5f5f7;
    border: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #c7c7cc;
    border-radius: 4px;
    min-height: 28px;
    min-width: 28px;
}
"""

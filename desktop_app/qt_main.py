"""PySide6 desktop application for OffroadSimBench."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QObject, QRectF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
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

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))


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
    def __init__(self) -> None:
        super().__init__()
        self.history: dict[str, list[float]] = {}
        self.primary_metric = ""
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_history(self, history: dict[str, list[float]]) -> None:
        self.history = {
            str(key): [float(value) for value in values if math.isfinite(float(value))]
            for key, values in history.items()
            if values
        }
        priority = ["loss", "final_loss", "train_rmse", "train_mse", "metadata.mean_goal_distance", "total_frames"]
        self.primary_metric = next((key for key in priority if key in self.history), next(iter(self.history), ""))
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        rect = self.rect().adjusted(18, 20, -18, -28)
        painter.setPen(QPen(QColor("#d2d2d7"), 1))
        painter.drawRect(rect)
        for index in range(1, 4):
            y = rect.top() + rect.height() * index / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        if not self.primary_metric:
            painter.setPen(QPen(QColor("#6e6e73"), 1))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Training metrics: NaN")
            return

        values = self.history[self.primary_metric]
        low = min(values)
        high = max(values)
        if math.isclose(low, high):
            low -= 1.0
            high += 1.0

        def project(index: int, value: float) -> tuple[float, float]:
            x = rect.left() + (index / max(len(values) - 1, 1)) * rect.width()
            y = rect.bottom() - ((value - low) / (high - low)) * rect.height()
            return x, y

        painter.setPen(QPen(QColor("#007aff"), 2))
        points = [project(index, value) for index, value in enumerate(values)]
        if len(points) == 1:
            x, y = points[0]
            painter.setBrush(QColor("#007aff"))
            painter.drawEllipse(QRectF(x - 4, y - 4, 8, 8))
        else:
            for start, end in zip(points, points[1:], strict=False):
                painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

        painter.setPen(QPen(QColor("#1d1d1f"), 1))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(18, 16, f"{self.primary_metric}: {services.display_value(values[-1])}")
        painter.setPen(QPen(QColor("#6e6e73"), 1))
        painter.drawText(rect.left(), self.height() - 8, f"points={len(values)}")


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
        self.resize(1380, 880)
        self.settings = GuiSettings()
        self.catalog: dict[str, list[dict[str, Any]]] = {}
        self.threads: list[QThread] = []
        self.workers: list[TaskWorker] = []
        self.metric_cards: dict[str, MetricCard] = {}
        self.nav_buttons: list[QPushButton] = []
        self.dataset_info: dict[str, Any] | None = None
        self.navigation_preview_session = services.BeamNGNavigationPreviewSession()
        self.region_task_dialog: NavigationTaskDialog | None = None
        self._navigation_preview_busy = False
        self._navigation_preview_pending: tuple[str, str, float] | None = None
        self._busy_depth = 0

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

    def _init_shared_controls(self) -> None:
        self.backend_combo = self._combo()
        self.scenario_combo = self._combo()
        self.agent_combo = self._combo()
        self.world_model_combo = self._combo()
        self.planner_combo = self._combo()
        self.algorithm_combo = self._combo()
        self.demo_preset_combo = self._combo()
        self.demo_preset_combo.addItem("Johnson Valley LE-WM navigation", "johnson_valley_lewm_navigation")
        self.training_config_combo = self._combo()
        self.training_preset_combo = self._combo()
        self.training_preset_summary = QTextEdit()
        self.training_preset_summary.setReadOnly(True)
        self.training_preset_summary.setFixedHeight(126)
        self.training_preset_summary.setPlaceholderText("Training config: NaN")
        self.world_model_config_combo = self._combo()
        self.world_model_config_edit_combo = self._combo()
        self.beamng_model_config_combo = self._combo()
        self.trainer_params_edit = QTextEdit()
        self.trainer_params_edit.setPlaceholderText('{"epochs": 10, "batch_size": 16}')
        self.trainer_params_edit.setFixedHeight(96)
        self._trainer_params_autofill = ""
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
        self.model_config_name_edit.setPlaceholderText("Johnson Valley LE-WM validated")

        self.dataset_catalog_combo = self._combo()
        self.dataset_manifest_name_edit = QLineEdit()
        self.dataset_manifest_name_edit.setPlaceholderText("Custom driving dataset")
        self.dataset_manifest_sequences_edit = QTextEdit()
        self.dataset_manifest_sequences_edit.setPlaceholderText(
            '[{"id": "clip_001", "root": ".", "assets": {"front_rgb": "images/*.png"}}]'
        )
        self.dataset_manifest_sequences_edit.setFixedHeight(92)
        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.setPlaceholderText(r"datasets\ORFD_Dataset_ICRA2022_ZIP")
        self.sequence_combo = self._combo(editable=True)
        self.adapter_edit = QLineEdit("orfd")
        self.stablewm_hdf5_edit = QLineEdit()
        self.stablewm_hdf5_edit.setPlaceholderText(r"outputs\stablewm\orfd_gui.h5")
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(str(services.DEFAULT_LEWM_CHECKPOINT_PATH))
        self.task_path_edit = QLineEdit(str(services.DEFAULT_NAVIGATION_TASK_PATH))
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
        return area

    def _build_overview_page(self) -> QWidget:
        page, layout = self._page()

        body = self._row_layout()
        launcher_box, launcher_layout = self._new_group("Guided demo launcher")
        guide = QLabel(
            "按步骤检查 demo 配置，然后运行 BeamNG 可视自动驾驶。复杂的数据集、训练和任务编辑放到对应工作台。"
        )
        guide.setObjectName("mutedText")
        guide.setWordWrap(True)
        launcher_layout.addWidget(guide)
        launcher_layout.addWidget(self._field("Demo preset", self.demo_preset_combo))
        launcher_layout.addWidget(self._field("BeamNG region task", self.home_task_combo))
        launcher_layout.addWidget(self._field("World model config", self.world_model_config_combo))
        launcher_layout.addWidget(self._field("Planner", self.planner_combo))

        run_button = QPushButton("Run guided demo")
        self._configure_button(run_button, primary=True)
        run_button.clicked.connect(self.run_guided_demo)
        launcher_layout.addWidget(run_button)

        shortcut_row = self._row_layout(spacing=8)
        dataset_button = QPushButton("Open Dataset & Training")
        self._configure_button(dataset_button)
        dataset_button.clicked.connect(lambda: self.select_page(1))
        beamng_button = QPushButton("Open BeamNG Simulation")
        self._configure_button(beamng_button)
        beamng_button.clicked.connect(lambda: self.select_page(2))
        records_button = QPushButton("Open Records")
        self._configure_button(records_button)
        records_button.clicked.connect(lambda: self.select_page(3))
        shortcut_row.addWidget(dataset_button)
        shortcut_row.addWidget(beamng_button)
        shortcut_row.addWidget(records_button)
        launcher_layout.addLayout(shortcut_row)
        body.addWidget(launcher_box, 2)

        status_box, status_layout = self._new_group("Demo status")
        self.demo_status_summary = QTextEdit()
        self.demo_status_summary.setReadOnly(True)
        self.demo_status_summary.setPlaceholderText("Demo status: NaN")
        status_layout.addWidget(self.demo_status_summary, 1)
        body.addWidget(status_box, 1)
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

        data_tab = QWidget()
        data_root = QVBoxLayout(data_tab)
        data_root.setContentsMargins(0, 0, 0, 0)
        data_root.setSpacing(CARD_SPACING)
        data_root.addWidget(self._tab_header("Dataset import", "Register a dataset, inspect sequences, and preview RGB/depth or label frames."))
        data_layout = self._row_layout()
        dataset_browse = QPushButton("选择")
        self._configure_button(dataset_browse)
        dataset_browse.clicked.connect(lambda: self._browse_dir(self.dataset_root_edit))
        dataset_import = QPushButton("导入 dataset manifest")
        self._configure_button(dataset_import)
        dataset_import.clicked.connect(self.import_dataset_manifest)
        dataset_save = QPushButton("Save dataset manifest")
        self._configure_button(dataset_save)
        dataset_save.clicked.connect(self.save_dataset_manifest_from_gui)
        controls = self._group(
            "Data source",
            [
                self._field("Dataset catalog", self.dataset_catalog_combo),
                dataset_import,
                self._field("Dataset root", self._with_button(self.dataset_root_edit, dataset_browse)),
                self._field("Dataset name", self.dataset_manifest_name_edit),
                self._field("Manifest sequences", self.dataset_manifest_sequences_edit),
                dataset_save,
                self._field("Sequence", self.sequence_combo),
                self._field("Adapter", self.adapter_edit),
                self._field("StableWM HDF5", self.stablewm_hdf5_edit),
                self._action_button("检查数据集", self.inspect_dataset),
                self._action_button("Preview dataset frame", self.preview_dataset),
                self._action_button("导出 StableWM HDF5", self.export_stablewm_hdf5),
            ],
        )
        data_layout.addWidget(controls, 1)
        preview_box, preview_layout = self._new_group("Dataset preview")
        image_row = self._row_layout(spacing=CARD_SPACING)
        self.rgb_preview = self._preview_label("RGB: NaN")
        self.depth_preview = self._preview_label("Depth/Label: NaN")
        image_row.addWidget(self._preview_panel("RGB preview", self.rgb_preview), 1)
        image_row.addWidget(self._preview_panel("Depth / Label preview", self.depth_preview), 1)
        preview_layout.addLayout(image_row, 2)
        self.dataset_summary = QTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setPlaceholderText("数据集检查结果：NaN")
        preview_layout.addWidget(self._section_label("Frame metadata"))
        preview_layout.addWidget(self.dataset_summary, 1)
        data_layout.addWidget(preview_box, 2)
        data_root.addLayout(data_layout, 1)
        tabs.addTab(data_tab, "数据集")

        training_tab = QWidget()
        training_root = QVBoxLayout(training_tab)
        training_root.setContentsMargins(0, 0, 0, 0)
        training_root.setSpacing(CARD_SPACING)
        training_root.addWidget(self._tab_header("Model training", "Choose a reusable training config or bind a local trainer script with parameters."))
        training_layout = self._row_layout()
        model_browse = QPushButton("选择")
        self._configure_button(model_browse)
        model_browse.clicked.connect(lambda: self._browse_path_combo(self.home_model_combo))
        model_import = QPushButton("Import model/checkpoint")
        self._configure_button(model_import)
        model_import.clicked.connect(self.import_world_model_config)
        model_dir_import = QPushButton("Import model folder")
        self._configure_button(model_dir_import)
        model_dir_import.clicked.connect(self.import_world_model_directory_config)
        trainer_import = QPushButton("导入训练器 manifest")
        self._configure_button(trainer_import)
        trainer_import.clicked.connect(self.import_trainer_manifest)
        trainer_entry_browse = QPushButton("Select")
        self._configure_button(trainer_entry_browse)
        trainer_entry_browse.clicked.connect(lambda: self._browse_file(self.trainer_entrypoint_edit, "Select trainer entrypoint"))
        self.save_trainer_button = QPushButton("Save trainer from script")
        self._configure_button(self.save_trainer_button)
        self.save_trainer_button.clicked.connect(self.save_trainer_manifest_from_gui)
        training_config_import = QPushButton("Import training config")
        self._configure_button(training_config_import)
        training_config_import.clicked.connect(self.import_training_config)
        training_controls = self._group(
            "Training config",
            [
                self._field("Training config", self.training_config_combo),
                training_config_import,
                self._field("Config name", self.training_config_name_edit),
                self._field("Training preset", self.training_preset_combo),
                self._section_label("Training config summary"),
                self.training_preset_summary,
                self._field("Training parameters", self.trainer_params_edit),
                self._field("Training output", self.training_output_edit),
                self._action_button("Validate config", self.validate_training_config),
                self._action_button("Save training config", self.save_training_config),
                self._action_button("Start training/export", self.run_training_preset, primary=True),
            ],
        )
        trainer_box = self._group(
            "Trainer / algorithm",
            [
                trainer_import,
                self._field("Trainer entrypoint", self._with_button(self.trainer_entrypoint_edit, trainer_entry_browse)),
                self._field("Trainer arguments", self.trainer_arguments_edit),
                self._field("Trainer parameter schema", self.trainer_schema_edit),
                self.save_trainer_button,
            ],
        )
        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(CARD_SPACING)
        left_column.addWidget(training_controls, 3)
        left_column.addWidget(trainer_box, 1)
        training_layout.addLayout(left_column, 1)
        output_box, output_layout = self._new_group("Latest training result")
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setPlaceholderText("模型训练/推理结果：NaN")
        self.latest_training_curve = TrainingCurveWidget()
        output_layout.addWidget(self._section_label("Latest metric curve"))
        self.latest_metric_summary = QLabel("Metric curves: NaN")
        self.latest_metric_summary.setObjectName("mutedText")
        self.latest_metric_summary.setWordWrap(True)
        output_layout.addWidget(self.latest_metric_summary)
        output_layout.addWidget(self.latest_training_curve)
        output_layout.addWidget(self._section_label("Training output"))
        output_layout.addWidget(self.model_summary, 1)
        registry_box = self._group(
            "Trained model registry",
            [
                self._field("World model config", self.world_model_config_edit_combo),
                self._field("Config name", self.model_config_name_edit),
                self._field("Model path", self._with_button(self.home_model_combo, model_browse)),
                model_import,
                model_dir_import,
                self._field("Algorithm", self.algorithm_combo),
                self._field("World model", self.world_model_combo),
                self._action_button("Save world model config", self.save_world_model_config, primary=True),
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
        runs_root.addWidget(self._tab_header("Training results", "Review completed runs, loss curves, artifacts, logs, and promoted model configs."))
        runs_layout = self._row_layout()
        run_list_box, run_list_layout = self._new_group("Training runs")
        self.training_run_list = QListWidget()
        self.training_run_list.itemClicked.connect(self._load_selected_training_run)
        run_list_layout.addWidget(self.training_run_list, 1)
        runs_layout.addWidget(run_list_box, 1)
        run_summary_box, run_summary_layout = self._new_group("Run details")
        self.training_run_overview = QTextEdit()
        self.training_run_overview.setReadOnly(True)
        self.training_run_overview.setMaximumHeight(150)
        self.training_run_overview.setPlaceholderText("Training run summary: NaN")
        run_summary_layout.addWidget(self._section_label("Run summary"))
        run_summary_layout.addWidget(self.training_run_overview)
        self.training_curve = TrainingCurveWidget()
        run_summary_layout.addWidget(self._section_label("Metric curve"))
        self.training_run_metric_summary = QLabel("Metric curves: NaN")
        self.training_run_metric_summary.setObjectName("mutedText")
        self.training_run_metric_summary.setWordWrap(True)
        run_summary_layout.addWidget(self.training_run_metric_summary)
        run_summary_layout.addWidget(self.training_curve)
        self.training_run_summary = QTextEdit()
        self.training_run_summary.setReadOnly(True)
        self.training_run_summary.setPlaceholderText("Training run: NaN")
        run_summary_layout.addWidget(self._section_label("Raw training_run.json"))
        run_summary_layout.addWidget(self.training_run_summary, 1)
        runs_layout.addWidget(run_summary_box, 2)
        runs_root.addLayout(runs_layout, 1)
        tabs.addTab(runs_tab, "Training results")

        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.setSpacing(PAGE_SPACING)
        processing_layout.addWidget(self._tab_header("Processing and labels", "Reserved tools for segmentation, masks, labels, and future dataset-to-map conversion."))
        processing_hint = QLabel(
            "图像分割、标签检查、terrain mask 和数据集到 BeamNG 地图转换会放在这里；未实现项保持 NaN/未完成。"
        )
        processing_hint.setObjectName("mutedText")
        processing_hint.setWordWrap(True)
        processing_layout.addWidget(self._group("Dataset processing and labels", [processing_hint]))
        processing_layout.addStretch(1)
        tabs.addTab(processing_tab, "处理/标注")

        layout.addWidget(tabs, 1)
        return page

    def _build_beamng_simulation_page(self) -> QWidget:
        page, layout = self._page()
        tabs = QTabWidget()

        setup_tab = QWidget()
        setup_layout = QHBoxLayout(setup_tab)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(PAGE_SPACING)
        controls = self._group(
            "BeamNG task and model",
            [
                self._field("Region task", self.beamng_task_combo),
                self._field("World model config", self.beamng_model_config_combo),
                self._field("Resolved task path", self.task_path_edit),
                self._action_button("区域自监督训练 world model", self.train_region_self_supervised_world_model),
                self._action_button("编辑/预览区域与起终点", self.open_region_task_editor),
                self._action_button("运行当前区域任务", self.run_region_navigation_loop, primary=True),
                self._action_button("检查 BeamNG", self.check_beamng),
            ],
        )
        setup_layout.addWidget(controls, 1)
        summary_box, summary_layout = self._new_group("Simulation status")
        self.beamng_summary = QTextEdit()
        self.beamng_summary.setReadOnly(True)
        self.beamng_summary.setPlaceholderText("BeamNG 状态、任务分析与运行结果：NaN")
        summary_layout.addWidget(self.beamng_summary, 1)
        setup_layout.addWidget(summary_box, 2)
        tabs.addTab(setup_tab, "运行配置")

        map_tab = QWidget()
        map_layout = QHBoxLayout(map_tab)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(PAGE_SPACING)
        map_controls = self._group(
            "Map and terrain tools",
            [
                self._action_button("编辑/预览区域与起终点", self.open_region_task_editor),
                self._action_button("导出 BeamNG 地形草案", self.export_beamng_terrain_draft),
            ],
        )
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
            self.catalog.get("world_model_configs", []),
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
        self._fill_episode_list()
        self._refresh_planner_summary()
        beamng = _find_named(self.catalog["backends"], "beamng")
        self.runtime_label.setText(f"BeamNG: {services.display_value(beamng.get('available') if beamng else None)}")
        if hasattr(self, "demo_status_summary"):
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
        if self.region_task_dialog is not None:
            self.region_task_dialog.close()
            self.region_task_dialog = None
        self.navigation_preview_session.close()
        super().closeEvent(event)

    def run_guided_demo(self) -> None:
        self._select_combo_value(self.backend_combo, "beamng")
        self._select_combo_value(self.agent_combo, "model_mpc")
        self._select_combo_value(self.planner_combo, "navigation_mpc")
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
                eval_steps=max(int(self.settings.max_steps), 1000),
                seed=self.settings.seed,
                planner="navigation_mpc",
                planner_horizon=self.settings.planner_horizon,
                planner_samples=self.settings.planner_samples,
                planner_iterations=self.settings.planner_iterations,
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
            eval_steps=max(int(self.settings.max_steps), 1000),
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
            "dataset inspect failed",
            task_label="检查数据集",
        )

    def preview_dataset(self) -> None:
        self.log("生成数据集帧预览...")
        self._run_task(
            lambda: services.preview_dataset_frame(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                frame_index=self.settings.preview_frame_index,
            ),
            self._preview_ready,
            "dataset preview failed",
            task_label="预览数据集",
        )

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
        self.log(f"Starting training config: {row.get('label', row.get('id', services.NAN_TEXT))}")
        self._run_task(
            lambda: services.run_training_config_job(row, trainer_root=trainer_root),
            self._training_config_finished,
            "training config failed",
            task_label=f"Train {preset.get('label', preset_id)}",
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
        algorithm = self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "local_lewm_cost"
        request = services.RegionNavigationClosedLoopRequest(
            task_path=self.task_path_edit.text().strip() or str(services.DEFAULT_NAVIGATION_TASK_PATH),
            algorithm=algorithm,
            algorithm_model_path=self.model_path_edit.text().strip() if algorithm == "stablewm_lewm" else "",
            collect_steps=max(int(self.settings.max_steps), 1000),
            eval_steps=max(int(self.settings.max_steps), 1000),
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

    def train_region_self_supervised_world_model(self) -> None:
        task_path = self.task_path_edit.text().strip() or self._path_combo_value(self.beamng_task_combo).strip()
        if not task_path:
            self.log("区域自监督训练需要先选择 BeamNG region task。")
            return
        request = services.RegionSelfSupervisedWorldModelRequest(
            task_path=task_path,
            world_model_type="tiny_learned",
            collect_steps=max(int(self.settings.max_steps), 1000),
            collect_rollouts=3,
            min_collection_goal_progress_ratio=0.25,
            collection_coverage_grid_size=4,
            collection_coverage_target_interval=1,
            collection_max_target_steps=40,
            eval_steps=max(int(self.settings.max_steps), 1000),
            seed=self.settings.seed,
            planner=self.planner_combo.currentData() or self.planner_combo.currentText() or "navigation_mpc",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            evaluation_agent="world_model_direct",
            evaluation_route_mode="route_free",
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
        region_summary = _region_world_model_summary_text(payload)
        if region_summary and hasattr(self, "beamng_summary"):
            self.beamng_summary.setText(region_summary)
        self.log("流程完成")
        self.refresh_catalogs()
        if saved_config_id:
            self._select_world_model_config(saved_config_id)

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
        self.dataset_summary.setText(_compact_json(payload))
        self.sequence_combo.clear()
        for sequence_id in payload.get("sequences", []):
            self.sequence_combo.addItem(str(sequence_id))
        selected = str(payload.get("selected_sequence", ""))
        index = self.sequence_combo.findText(selected)
        if index >= 0:
            self.sequence_combo.setCurrentIndex(index)
        self.log(f"数据集 OK: {payload.get('dataset_id')} / frames={payload.get('frame_count')}")

    def _preview_ready(self, payload: dict[str, Any]) -> None:
        previews = payload.get("previews", {}) if isinstance(payload.get("previews"), dict) else {}
        self._set_preview(self.rgb_preview, previews.get("front_rgb"), "RGB: NaN")
        self._set_preview(self.depth_preview, previews.get("depth") or previews.get("label"), "Depth/Label: NaN")
        self.dataset_summary.setText(_compact_json(payload))
        self.log(f"预览完成: frame={payload.get('frame_id', services.NAN_TEXT)}")

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
        self.adapter_edit.setText(str(row.get("adapter") or ""))
        sequence_id = str(row.get("sequence_id") or "")
        if sequence_id:
            self.sequence_combo.setCurrentText(sequence_id)
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
        self.dataset_catalog_combo.addItem("Manual dataset path", {})
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
        sequences = row.get("sequences") if isinstance(row.get("sequences"), list) else []
        self.sequence_combo.clear()
        for sequence_id in sequences:
            self.sequence_combo.addItem(str(sequence_id))
        if sequences:
            self.sequence_combo.setCurrentIndex(0)
        self.dataset_summary.setText(_compact_json({"dataset": row}))

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
        defaults = self._trainer_parameter_defaults(preset.get("parameters") if isinstance(preset.get("parameters"), dict) else {})
        text = json.dumps(defaults, indent=2, ensure_ascii=False) if defaults else "{}"
        current = self.trainer_params_edit.toPlainText().strip()
        if not force and current and current != self._trainer_params_autofill:
            return
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
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON object: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object.")
        return payload

    def _dataset_sequences_from_text(self) -> list[dict[str, Any]]:
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
        self.training_run_list.clear()
        for run in self.catalog.get("training_runs", [])[:100]:
            label = str(run.get("preset_label") or run.get("preset_id") or services.NAN_TEXT)
            status = str(run.get("status") or services.NAN_TEXT)
            artifact = str(run.get("artifact_path") or run.get("relative_path") or "")
            item = QListWidgetItem(f"{label} | {status} | {artifact}")
            item.setData(Qt.ItemDataRole.UserRole, dict(run))
            self.training_run_list.addItem(item)

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
        elif artifact_path and artifact_type == "hdf5":
            self.stablewm_hdf5_edit.setText(artifact_path)

    def _set_training_run_views(self, payload: dict[str, Any]) -> dict[str, Any]:
        run = _training_run_record_from_payload(payload)
        if hasattr(self, "training_run_overview"):
            self.training_run_overview.setText(_training_run_overview_text(run))
        if hasattr(self, "training_run_summary"):
            self.training_run_summary.setText(_compact_json(run))
        history = services.training_metric_history(run)
        metric_summary = _metric_history_summary(history)
        if hasattr(self, "training_run_metric_summary"):
            self.training_run_metric_summary.setText(metric_summary)
        if hasattr(self, "latest_metric_summary"):
            self.latest_metric_summary.setText(metric_summary)
        if hasattr(self, "training_curve"):
            self.training_curve.set_history(history)
        if hasattr(self, "latest_training_curve"):
            self.latest_training_curve.set_history(history)
        return run

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
        if not hasattr(self, "demo_status_summary"):
            return
        beamng = _find_named(self.catalog.get("backends", []), "beamng") if self.catalog else None
        config = self._combo_config_row(self.world_model_config_combo)
        payload = {
            "demo_preset": self.demo_preset_combo.currentText() or "NaN",
            "beamng_available": beamng.get("available") if beamng else None,
            "region_task": self._path_combo_value(self.home_task_combo) or services.NAN_TEXT,
            "world_model_config": config.get("label") or services.NAN_TEXT,
            "model_path": config.get("model_path") or services.NAN_TEXT,
            "planner": self.planner_combo.currentData() or self.planner_combo.currentText() or services.NAN_TEXT,
            "last_result": services.NAN_TEXT,
        }
        self.demo_status_summary.setText(_compact_json(payload))

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
    ) -> None:
        thread = QThread(self)
        worker = TaskWorker(fn)
        worker.moveToThread(thread)
        self._set_busy(True, task_label or _task_label_from_failure(failure_label))
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.failed.connect(lambda message: self._task_failed(failure_label, message))
        worker.finished.connect(lambda _: self._set_busy(False))
        worker.failed.connect(lambda _: self._set_busy(False))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_thread(thread, worker))
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
        if failure_label == "navigation realtime preview failed":
            self.beamng_summary.setText(_compact_json({"status": "preview_failed", "message": message}))
            self._finish_navigation_preview_task()

    def _remove_thread(self, thread: QThread, worker: TaskWorker) -> None:
        if thread in self.threads:
            self.threads.remove(thread)
        if worker in self.workers:
            self.workers.remove(worker)

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

    def _preview_label(self, placeholder: str) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(PREVIEW_MIN_HEIGHT)
        label.setObjectName("previewPane")
        label.setScaledContents(False)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return label

    def _set_preview(self, label: QLabel, path: Any, placeholder: str) -> None:
        if not path or not Path(str(path)).exists():
            label.setPixmap(QPixmap())
            label.setText(placeholder)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText(placeholder)
            return
        label.setText("")
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

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


def _compact_json(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _metric_history_summary(history: dict[str, list[float]]) -> str:
    if not history:
        return "Metric curves: NaN"
    parts = [f"{key} ({len(values)} pts)" for key, values in sorted(history.items()) if values]
    return "Metric curves: " + (", ".join(parts) if parts else services.NAN_TEXT)


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
    history = services.training_metric_history(run)
    metric_lines: list[str] = []
    for key in [
        "loss",
        "final_loss",
        "train_rmse",
        "train_mse",
        "goal_success",
        "min_goal_distance",
        "collection_min_goal_distance",
        "collection_distance_traveled",
    ]:
        if key in metrics:
            metric_lines.append(f"{key}: {services.display_value(metrics[key])}")
        elif key in history and history[key]:
            metric_lines.append(f"{key}: {services.display_value(history[key][-1])}")
    if metric_lines:
        lines.extend(metric_lines[:6])
    else:
        lines.append(f"metrics: {services.NAN_TEXT}")
    return "\n".join(lines)


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
    for key in ["model_dir", "training_run_path", "summary_path"]:
        value = str(payload.get(key) or "").strip()
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _training_run_record_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("training_run_path") or "").strip()
    if path:
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        if isinstance(record, dict) and record:
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

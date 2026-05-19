"""PySide6 desktop application for OffroadSimBench."""

from __future__ import annotations

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
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
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
    max_steps: int = 5
    seed: int = 7
    planner_horizon: int = 4
    planner_samples: int = 16
    planner_iterations: int = 2
    image_size: int = 64
    preview_frame_index: int = 0
    terrain_frame_index: int = 0
    terrain_grid_size: int = 64
    terrain_size_m: int = 40
    record: bool = True
    record_arrays: bool = False
    load_assets: bool = False
    pipeline_runs_beamng: bool = True


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
            ("pipeline_runs_beamng", "一键流程完成后运行 BeamNG"),
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

    def __init__(self) -> None:
        super().__init__()
        self.mode = "region"
        self.region: list[tuple[float, float]] = [(0.0, -160.0), (30.0, -160.0), (30.0, -230.0), (0.0, -230.0)]
        self.start: tuple[float, float, float] = (1.0, -170.0, 100.6)
        self.goal: tuple[float, float] = (6.0, -215.0)
        self.route: list[tuple[float, float]] = [(1.0, -170.0), (6.0, -215.0)]
        self.beamng_pose: tuple[float, float] | None = None
        self._drag_target: tuple[str, int] | None = None
        self._drag_pick_radius_px = 12.0
        self.bounds = (-8.0, 36.0, -240.0, -150.0)
        self.setMinimumHeight(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

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
        self.start = (float(point[0]), float(point[1]), self.start[2] if z is None else float(z))
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

        self._draw_point(painter, (self.start[0], self.start[1]), QColor("#f6c85f"), 7)
        self._draw_point(painter, self.goal, QColor("#ff6b6b"), 8)
        if self.beamng_pose is not None:
            projected_pose = self._to_canvas(self.beamng_pose)
            painter.setPen(QPen(QColor("#d77bff"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(projected_pose.x() - 9, projected_pose.y() - 9, 18, 18))
        painter.setPen(QPen(QColor("#8da1af"), 1))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(18, 22, f"mode: {self.mode} | click to set points")

    def mousePressEvent(self, event: Any) -> None:
        canvas_point = event.position() if hasattr(event, "position") else QPointF(event.x(), event.y())
        point = self._from_canvas(canvas_point)
        self._drag_target = self._nearest_drag_target(canvas_point)
        if self._drag_target is not None:
            return
        if self.mode == "region":
            self.region.append(point)
        elif self.mode == "start":
            self.start = (point[0], point[1], self.start[2])
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
        min_x, max_x, min_y, max_y = self.bounds
        plot = self._plot_rect()
        x = plot.left() + (point[0] - min_x) / max(max_x - min_x, 1e-6) * plot.width()
        y = plot.bottom() - (point[1] - min_y) / max(max_y - min_y, 1e-6) * plot.height()
        return QPointF(x, y)

    def _from_canvas(self, point: QPointF) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = self.bounds
        plot = self._plot_rect()
        x = min_x + (point.x() - plot.left()) / max(plot.width(), 1e-6) * (max_x - min_x)
        y = min_y + (plot.bottom() - point.y()) / max(plot.height(), 1e-6) * (max_y - min_y)
        return (round(x, 3), round(y, 3))

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
        elif self.mode == "start":
            handles.append(("start", 0, (self.start[0], self.start[1])))
        elif self.mode == "goal":
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
            self.start = (point[0], point[1], self.start[2])
            if self.route:
                self.route[0] = point
        elif kind == "goal":
            self.goal = point
            if self.route:
                self.route[-1] = point
        self._fit_bounds()
        self.update()
        self.changed.emit()

    def _fit_bounds(self) -> None:
        points = [*self.region, (self.start[0], self.start[1]), self.goal, *self.route]
        if self.beamng_pose is not None:
            points.append(self.beamng_pose)
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
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑/预览区域与起终点")
        self.resize(760, 680)
        self.saved_task_path = ""
        self.preview_callback = preview_callback
        self.pose_callback = pose_callback
        self.current_beamng_pose: dict[str, Any] = {"available": False}
        self.canvas = NavigationTaskCanvas()
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(650)
        self.preview_timer.timeout.connect(self._run_realtime_preview)

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
        self.preview_height_spin = self._double_spin(90.0, 10.0, 500.0)
        self.realtime_preview_check = QCheckBox("实时预览")
        self.realtime_preview_check.setChecked(True)
        self.beamng_pose_label = QLabel("BeamNG 当前位置：未读取")
        self.beamng_pose_label.setObjectName("mutedText")

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

        mode_row = QHBoxLayout()
        for label, mode in [
            ("选区域点", "region"),
            ("选起点", "start"),
            ("选终点", "goal"),
            ("加专家路径点", "route"),
        ]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, value=mode: self.canvas.set_mode(value))
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
        start = (self.canvas.start[0], self.canvas.start[1], float(self.start_z_spin.value()))
        try:
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
        ("总览", "配置最常用的运行项并开始测试。"),
        ("数据集", "导入、检查和预览 ORFD 数据，导出 StableWM HDF5。"),
        ("世界模型", "训练 tiny/LE-WM cost model，或运行一键 ORFD 到 BeamNG 流程。"),
        ("路径规划", "查看当前规划器和 CEM 参数，复杂参数从高级参数调整。"),
        ("BeamNG", "检查 BeamNG 运行时，导出 ORFD 局部地形草案。"),
        ("实验记录", "浏览 episode、轨迹和运行日志。"),
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

        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.setPlaceholderText(r"datasets\ORFD_Dataset_ICRA2022_ZIP")
        self.sequence_combo = self._combo(editable=True)
        self.adapter_edit = QLineEdit("orfd")
        self.stablewm_hdf5_edit = QLineEdit()
        self.stablewm_hdf5_edit.setPlaceholderText(r"outputs\stablewm\orfd_gui.h5")
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(r"outputs\models\lewm_orfd_gui")
        self.task_path_edit = QLineEdit(r"configs\tasks\beamng_johnson_valley_nav_001.yaml")
        for edit in (self.dataset_root_edit, self.adapter_edit, self.stablewm_hdf5_edit, self.model_path_edit, self.task_path_edit):
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
        self.page_stack.addWidget(self._build_dataset_page())
        self.page_stack.addWidget(self._build_world_model_page())
        self.page_stack.addWidget(self._build_planning_page())
        self.page_stack.addWidget(self._build_beamng_page())
        self.page_stack.addWidget(self._build_records_page())
        layout.addWidget(self.page_stack, 1)
        return area

    def _build_overview_page(self) -> QWidget:
        page, layout = self._page()

        config_row = self._row_layout()
        config_row.addWidget(
            self._group(
                "基础运行配置",
                [
                    self._field("Backend", self.backend_combo),
                    self._field("Scenario", self.scenario_combo),
                    self._field("Agent", self.agent_combo),
                    self._field("Algorithm", self.algorithm_combo),
                    self._field("World model", self.world_model_combo),
                    self._field("Planner", self.planner_combo),
                ],
            ),
            1,
        )

        action_panel, action_layout = self._new_group("开始")
        run_button = QPushButton("开始测试")
        self._configure_button(run_button, primary=True)
        run_button.clicked.connect(self.run_episode)
        hint = QLabel("数据集路径、模型训练、地形草案等操作请从左侧对应页面进入。")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        action_layout.addWidget(run_button)
        action_layout.addWidget(hint)
        action_layout.addStretch(1)
        config_row.addWidget(action_panel, 1)
        layout.addLayout(config_row)

        metrics_box, metrics_layout = self._new_group("运行指标")
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

    def _build_dataset_page(self) -> QWidget:
        page, layout = self._page()
        body = self._row_layout()

        dataset_browse = QPushButton("选择")
        self._configure_button(dataset_browse)
        dataset_browse.clicked.connect(lambda: self._browse_dir(self.dataset_root_edit))
        controls = self._group(
            "数据集导入与转换",
            [
                self._field("Dataset root", self._with_button(self.dataset_root_edit, dataset_browse)),
                self._field("Sequence", self.sequence_combo),
                self._field("Adapter", self.adapter_edit),
                self._field("StableWM HDF5", self.stablewm_hdf5_edit),
                self._action_button("检查数据集", self.inspect_dataset),
                self._action_button("预览 ORFD 图像", self.preview_dataset),
                self._action_button("导出 StableWM HDF5", self.export_stablewm_hdf5),
            ],
        )
        body.addWidget(controls, 1)

        preview_box, preview_layout = self._new_group("数据预览")
        image_row = self._row_layout(spacing=CARD_SPACING)
        self.rgb_preview = self._preview_label("RGB: NaN")
        self.depth_preview = self._preview_label("Depth/Label: NaN")
        image_row.addWidget(self.rgb_preview, 1)
        image_row.addWidget(self.depth_preview, 1)
        preview_layout.addLayout(image_row, 2)
        self.dataset_summary = QTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setPlaceholderText("数据集检查结果：NaN")
        preview_layout.addWidget(self.dataset_summary, 1)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)
        return page

    def _build_world_model_page(self) -> QWidget:
        page, layout = self._page()
        body = self._row_layout()

        model_browse = QPushButton("选择")
        self._configure_button(model_browse)
        model_browse.clicked.connect(lambda: self._browse_path_or_dir(self.model_path_edit))
        controls = self._group(
            "模型训练与加载",
            [
                self._field("Model path", self._with_button(self.model_path_edit, model_browse)),
                self._action_button("训练 LE-WM cost model", self.train_lewm_cost_model),
                self._action_button("训练 tiny world model", self.train_tiny_model),
                self._action_button("一键 ORFD → LE-WM → BeamNG", self.run_orfd_lewm_pipeline, primary=True),
            ],
        )
        body.addWidget(controls, 1)
        output_box, output_layout = self._new_group("模型输出")
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setPlaceholderText("模型训练/一键流程结果：NaN")
        output_layout.addWidget(self.model_summary, 1)
        body.addWidget(output_box, 2)
        layout.addLayout(body, 1)
        return page

    def _build_planning_page(self) -> QWidget:
        page, layout = self._page()
        body = self._row_layout()
        hint = QLabel("规划器选择在总览页，详细 CEM 参数在高级参数中调整。")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        self.planner_summary = QTextEdit()
        self.planner_summary.setReadOnly(True)
        self.planner_summary.setMaximumHeight(220)
        body.addWidget(self._group("当前规划参数", [hint, self.planner_summary]), 2)
        advanced_button = QPushButton("打开高级参数")
        self._configure_button(advanced_button)
        advanced_button.clicked.connect(self.open_advanced_settings)
        body.addWidget(self._group("操作", [advanced_button]), 1)
        layout.addLayout(body)
        layout.addStretch(1)
        return page

    def _build_beamng_page(self) -> QWidget:
        page, layout = self._page()
        body = self._row_layout()

        controls = self._group(
            "BeamNG 与地形草案",
            [
                self._field("Region task", self.task_path_edit),
                self._action_button("编辑/预览区域与起终点", self.open_region_task_editor),
                self._action_button("Johnson Valley LE-WM 演示", self.run_johnson_valley_demo_loop, primary=True),
                self._action_button("区域起终点 LE-WM 闭环", self.run_region_navigation_loop, primary=True),
                self._action_button("启动 BeamNG 可视自动驾驶", self.run_visible_beamng_demo, primary=True),
                self._action_button("BeamNG LE-WM 闭环训练评估", self.run_beamng_lewm_closed_loop),
                self._action_button("检查 BeamNG", self.check_beamng),
                self._action_button("导出 BeamNG 地形草案", self.export_beamng_terrain_draft),
            ],
        )
        body.addWidget(controls, 1)
        preview_box, preview_layout = self._new_group("地形草案预览")
        self.terrain_preview = self._preview_label("Terrain: NaN")
        preview_layout.addWidget(self.terrain_preview, 2)
        self.beamng_summary = QTextEdit()
        self.beamng_summary.setReadOnly(True)
        self.beamng_summary.setPlaceholderText("BeamNG 状态与地形草案信息：NaN")
        preview_layout.addWidget(self.beamng_summary, 1)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)
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
        if index == 3:
            self._refresh_planner_summary()

    def refresh_catalogs(self) -> None:
        self.catalog = services.catalog_snapshot()
        self._fill_combo(self.backend_combo, self.catalog["backends"], "name", default="gym_heightmap")
        self._fill_combo(self.scenario_combo, self.catalog["scenarios"], "id", default="beamng_visible_autodrive")
        self._fill_combo(self.agent_combo, self.catalog["agents"], "name", default="world_model")
        self._fill_combo(self.algorithm_combo, self.catalog["algorithms"], "name", default="local_lewm_cost")
        self._fill_combo(self.world_model_combo, self.catalog["world_models"], "name", default="le_wm")
        self._fill_combo(self.planner_combo, [{"name": ""}] + self.catalog["planners"], "name", default="le_wm_cem")
        self._fill_episode_list()
        self._refresh_planner_summary()
        beamng = _find_named(self.catalog["backends"], "beamng")
        self.runtime_label.setText(f"BeamNG: {services.display_value(beamng.get('available') if beamng else None)}")
        self.log("状态已刷新")

    def open_advanced_settings(self) -> None:
        dialog = AdvancedSettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.values()
            self._refresh_planner_summary()
            self.log(f"高级参数已更新: {_compact_json(asdict(self.settings))}")

    def open_region_task_editor(self) -> None:
        dialog = NavigationTaskDialog(
            self.task_path_edit.text().strip(),
            self,
            preview_callback=self._preview_task_from_editor,
            pose_callback=self._read_navigation_preview_pose,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.saved_task_path:
            self.task_path_edit.setText(dialog.saved_task_path)
            self.beamng_summary.setText(_compact_json({"status": "task_saved", "task_path": dialog.saved_task_path}))
            self.log(f"区域任务已保存: {dialog.saved_task_path}")

    def _preview_task_from_editor(self, task_path: str, camera_mode: str, camera_height_m: float) -> None:
        self.task_path_edit.setText(task_path)
        self.log(f"BeamNG 预览区域/起终点: {task_path}, camera={camera_mode}, height={camera_height_m:.1f}m")
        self._run_task(
            lambda: self.navigation_preview_session.update(
                task_path,
                camera_mode=camera_mode,
                camera_height_m=camera_height_m,
            ),
            self._navigation_preview_finished,
            "navigation realtime preview failed",
        )

    def _read_navigation_preview_pose(self) -> dict[str, Any]:
        return self.navigation_preview_session.current_pose()

    def closeEvent(self, event: Any) -> None:
        self.navigation_preview_session.close()
        super().closeEvent(event)

    def run_episode(self) -> None:
        request = self._current_request()
        self.log(f"开始运行：backend={request.backend}, agent={request.agent}, planner={request.planner or 'none'}")
        self._run_task(lambda: services.run_episode_from_request(request), self._episode_finished, "episode failed")

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
        )

    def preview_dataset(self) -> None:
        self.log("生成 ORFD 预览...")
        self._run_task(
            lambda: services.preview_dataset_frame(
                self.dataset_root_edit.text().strip(),
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                frame_index=self.settings.preview_frame_index,
            ),
            self._preview_ready,
            "dataset preview failed",
        )

    def train_tiny_model(self) -> None:
        root = self.dataset_root_edit.text().strip()
        if not root:
            self.log("训练 tiny world model 需要 dataset root")
            return
        output = self.model_path_edit.text().strip() or str(services.ROOT / "outputs" / "models" / "gui_tiny_world_model")
        self.log(f"训练 tiny world model -> {output}")
        self._run_task(
            lambda: services.train_tiny_world_model(
                root,
                output,
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
            ),
            self._training_finished,
            "training failed",
        )

    def export_stablewm_hdf5(self) -> None:
        root = self.dataset_root_edit.text().strip()
        output = self.stablewm_hdf5_edit.text().strip() or str(services.ROOT / "outputs" / "stablewm" / "gui_export.h5")
        self.log(f"导出 StableWM HDF5 -> {output}")
        self._run_task(
            lambda: services.export_lewm_hdf5(
                root,
                output,
                adapter=self.adapter_edit.text().strip(),
                sequence_id=self.sequence_combo.currentText().strip(),
                image_size=self.settings.image_size,
            ),
            self._hdf5_exported,
            "stablewm export failed",
        )

    def train_lewm_cost_model(self) -> None:
        hdf5_path = self.stablewm_hdf5_edit.text().strip()
        output = self.model_path_edit.text().strip() or str(services.ROOT / "outputs" / "models" / "gui_lewm_cost")
        self.log(f"训练 LE-WM cost model -> {output}")
        self._run_task(
            lambda: services.train_lewm_cost_model(hdf5_path, output),
            self._training_finished,
            "lewm training failed",
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
            run_beamng=self.settings.pipeline_runs_beamng,
            beamng_scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText(),
        )
        self.log("启动一键流程：ORFD -> HDF5 -> LE-WM cost -> dataset replay -> BeamNG")
        self._run_task(lambda: services.run_orfd_lewm_pipeline(request), self._pipeline_finished, "pipeline failed")

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
        self._run_task(lambda: services.run_visible_beamng_demo(request), self._visible_demo_finished, "visible BeamNG demo failed")

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
        )

    def run_region_navigation_loop(self) -> None:
        request = services.RegionNavigationClosedLoopRequest(
            task_path=self.task_path_edit.text().strip() or r"configs\tasks\beamng_region_nav_001.yaml",
            algorithm=self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "local_lewm_cost",
            collect_steps=max(int(self.settings.max_steps), 160),
            eval_steps=max(int(self.settings.max_steps), 80),
            seed=self.settings.seed,
            planner=self.planner_combo.currentData() or self.planner_combo.currentText() or "le_wm_cem",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            close_beamng=False,
        )
        self.log("区域导航闭环：task -> expert采集 -> HDF5 -> 训练 -> start/goal评估")
        self._run_task(
            lambda: services.run_region_navigation_closed_loop(request),
            self._pipeline_finished,
            "region navigation loop failed",
        )

    def run_johnson_valley_demo_loop(self) -> None:
        self.task_path_edit.setText(r"configs\tasks\beamng_johnson_valley_nav_001.yaml")
        planner = self.planner_combo.currentData() or self.planner_combo.currentText() or "le_wm_cem"
        request = services.RegionNavigationClosedLoopRequest(
            task_path=r"configs\tasks\beamng_johnson_valley_nav_001.yaml",
            algorithm=self.algorithm_combo.currentData() or self.algorithm_combo.currentText() or "local_lewm_cost",
            collect_steps=max(int(self.settings.max_steps), 240),
            eval_steps=max(int(self.settings.max_steps), 300),
            seed=self.settings.seed,
            planner=planner if planner != "none" else "le_wm_cem",
            planner_horizon=max(self.settings.planner_horizon, 6),
            planner_samples=max(self.settings.planner_samples, 32),
            planner_iterations=max(self.settings.planner_iterations, 3),
            close_beamng=False,
            step_delay_sec=0.02,
            post_run_hold_sec=20.0,
        )
        self.log("Johnson Valley 演示：ai_line 采集 -> LE-WM cost 训练 -> manual 模型控车评估")
        self._run_task(
            lambda: services.run_region_navigation_closed_loop(request),
            self._pipeline_finished,
            "johnson valley demo loop failed",
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
        self.log("一键流程完成")
        self.refresh_catalogs()

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

    def _training_finished(self, payload: dict[str, Any]) -> None:
        self.model_path_edit.setText(str(payload.get("output_dir", "")))
        self.model_summary.setText(_compact_json(payload))
        self.log(f"模型训练完成: {payload.get('model_path', payload.get('checkpoint_path', services.NAN_TEXT))}")
        self.refresh_catalogs()

    def _hdf5_exported(self, payload: dict[str, Any]) -> None:
        self.stablewm_hdf5_edit.setText(str(payload.get("output_hdf5", "")))
        self.dataset_summary.setText(_compact_json(payload))
        self.log(f"HDF5 导出完成: {payload.get('output_hdf5', services.NAN_TEXT)}")

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

    def _current_request(self) -> services.RunRequest:
        return services.RunRequest(
            backend=self.backend_combo.currentData() or self.backend_combo.currentText(),
            scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText(),
            agent=self.agent_combo.currentData() or self.agent_combo.currentText(),
            seed=self.settings.seed,
            max_steps=self.settings.max_steps,
            record=self.settings.record,
            record_arrays=self.settings.record_arrays,
            world_model_type=self.world_model_combo.currentData() or self.world_model_combo.currentText(),
            world_model_path=self.model_path_edit.text().strip(),
            planner=self.planner_combo.currentData() or "",
            planner_horizon=self.settings.planner_horizon,
            planner_samples=self.settings.planner_samples,
            planner_iterations=self.settings.planner_iterations,
            dataset_root=self.dataset_root_edit.text().strip(),
            sequence_id=self.sequence_combo.currentText().strip(),
            adapter=self.adapter_edit.text().strip(),
            load_assets=self.settings.load_assets,
        )

    def _run_task(self, fn: Callable[[], Any], on_success: Callable[[Any], None], failure_label: str) -> None:
        thread = QThread(self)
        worker = TaskWorker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.failed.connect(lambda message: self.log(f"{failure_label}: {message}"))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_thread(thread, worker))
        self.threads.append(thread)
        self.workers.append(worker)
        thread.start()

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


def _find_named(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("name") == name:
            return row
    return None


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


def run() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


STYLESHEET = """
QWidget {
    background: #0b1117;
    color: #e9eff4;
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 13px;
}
#sidebar {
    background: #0f1820;
    border-right: 1px solid #233340;
}
#appTitle {
    font-size: 22px;
    font-weight: 700;
    color: #f3f7fa;
}
#pageTitle {
    font-size: 24px;
    font-weight: 700;
    color: #f3f7fa;
}
#mutedText, .mutedText {
    color: #8da1af;
}
#navButton {
    text-align: left;
    padding: 0 12px;
    border-radius: 6px;
    background: transparent;
    border: 1px solid transparent;
}
#navButton:hover {
    background: #152631;
    border-color: #315064;
}
#navButton:checked {
    background: #14352f;
    border-color: #3bd1a6;
    color: #6ee8c5;
}
QGroupBox {
    border: 1px solid #263946;
    border-radius: 7px;
    margin-top: 14px;
    padding-top: 4px;
    background: #101922;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #a6b7c3;
}
QLineEdit, QComboBox, QSpinBox {
    background: #0a1219;
    border: 1px solid #2a3e4d;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #1ba783;
}
QPushButton {
    background: #162635;
    border: 1px solid #315064;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #203748;
}
QPushButton:disabled {
    color: #667582;
    background: #141c24;
}
#primaryButton {
    background: #20b58f;
    color: #06130f;
    font-weight: 700;
    border-color: #58e0bd;
}
#primaryButton:hover {
    background: #2cc6a0;
}
#fieldLabel {
    color: #a7b7c4;
}
#metricCard, #todoCard, #previewPane {
    background: #0d161e;
    border: 1px solid #253846;
    border-radius: 7px;
}
#metricTitle {
    color: #91a7b6;
}
#metricValue {
    font-size: 24px;
    font-weight: 700;
    color: #f4f7fa;
}
#todoStatus {
    color: #f6c85f;
    font-weight: 700;
}
QTextEdit, QListWidget, QTableWidget {
    background: #0a1219;
    border: 1px solid #253846;
    border-radius: 7px;
    padding: 8px;
    selection-background-color: #1ba783;
}
QListWidget::item {
    min-height: 28px;
    padding: 4px 6px;
}
QListWidget::item:selected {
    background: #14352f;
    color: #6ee8c5;
}
QHeaderView::section {
    background: #172633;
    color: #c7d4df;
    border: 0;
    padding: 7px;
}
QSplitter::handle {
    background: #0b1117;
}
QSplitter::handle:horizontal {
    width: 10px;
}
QSplitter::handle:vertical {
    height: 10px;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0b1117;
    border: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #334857;
    border-radius: 4px;
    min-height: 28px;
    min-width: 28px;
}
"""

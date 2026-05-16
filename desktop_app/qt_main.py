"""PySide6 desktop application for OffroadSimBench."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop_app import services


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


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = services.NAN_TEXT) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
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
        for start, end in zip(projected, projected[1:]):
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OffroadSimBench Desktop")
        self.resize(1360, 860)
        self.catalog: dict[str, list[dict[str, Any]]] = {}
        self.threads: list[QThread] = []
        self.workers: list[TaskWorker] = []
        self.metric_cards: dict[str, MetricCard] = {}
        self.dataset_info: dict[str, Any] | None = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_main_area(), 1)
        self.refresh_catalogs()

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 18, 18, 18)
        title = QLabel("OffroadSimBench")
        title.setObjectName("appTitle")
        subtitle = QLabel("本地越野仿真实验台")
        subtitle.setObjectName("mutedText")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(18)
        for label in ["总览", "数据集", "世界模型", "路径规划", "BeamNG", "实验记录", "设置"]:
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setEnabled(label in {"总览", "数据集", "世界模型", "路径规划", "BeamNG", "实验记录"})
            layout.addWidget(button)
        layout.addStretch(1)
        self.runtime_label = QLabel("Runtime: NaN")
        self.runtime_label.setObjectName("mutedText")
        layout.addWidget(self.runtime_label)
        return sidebar

    def _build_main_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("实验控制台")
        title.setObjectName("pageTitle")
        subtitle = QLabel("已实现接口接真实运行链路；未完成能力只显示 NaN 或 未完成")
        subtitle.setObjectName("mutedText")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.refresh_button = QPushButton("刷新状态")
        self.refresh_button.clicked.connect(self.refresh_catalogs)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_workbench())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        return area

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("controlScroll")
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(12)

        self.backend_combo = self._combo()
        self.scenario_combo = self._combo()
        self.agent_combo = self._combo()
        self.world_model_combo = self._combo()
        self.planner_combo = self._combo()
        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.setPlaceholderText("D:\\datasets\\ORFD 或 outputs\\mock_orfd_phase3")
        self.sequence_combo = self._combo(editable=True)
        self.adapter_edit = QLineEdit()
        self.adapter_edit.setPlaceholderText("orfd / 留空自动识别")
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("outputs\\models\\phase3_tiny_world_model")
        self.stablewm_hdf5_edit = QLineEdit()
        self.stablewm_hdf5_edit.setPlaceholderText("outputs\\stablewm\\orfd_or_beamng.h5")

        layout.addWidget(self._group("运行配置", [
            self._field("Backend", self.backend_combo),
            self._field("Scenario", self.scenario_combo),
            self._field("Agent", self.agent_combo),
            self._field("World model", self.world_model_combo),
            self._field("Planner", self.planner_combo),
        ]))

        dataset_browse = QPushButton("选择")
        dataset_browse.clicked.connect(lambda: self._browse_dir(self.dataset_root_edit))
        model_browse = QPushButton("选择")
        model_browse.clicked.connect(lambda: self._browse_path_or_dir(self.model_path_edit))
        layout.addWidget(self._group("数据与模型", [
            self._field("Dataset root", self._with_button(self.dataset_root_edit, dataset_browse)),
            self._field("Sequence", self.sequence_combo),
            self._field("Adapter", self.adapter_edit),
            self._field("Model path", self._with_button(self.model_path_edit, model_browse)),
            self._field("StableWM HDF5", self.stablewm_hdf5_edit),
        ]))

        self.max_steps_spin = self._spin(1, 100000, 120)
        self.seed_spin = self._spin(0, 999999, 7)
        self.planner_horizon_spin = self._spin(1, 200, 10)
        self.planner_samples_spin = self._spin(4, 5000, 128)
        self.planner_iterations_spin = self._spin(1, 100, 4)
        self.record_check = QCheckBox("记录 episode")
        self.record_check.setChecked(True)
        self.record_arrays_check = QCheckBox("记录数组")
        self.load_assets_check = QCheckBox("加载数据集资产")
        layout.addWidget(self._group("参数", [
            self._field("Max steps", self.max_steps_spin),
            self._field("Seed", self.seed_spin),
            self._field("Planner horizon", self.planner_horizon_spin),
            self._field("Planner samples", self.planner_samples_spin),
            self._field("Planner iterations", self.planner_iterations_spin),
            self.record_check,
            self.record_arrays_check,
            self.load_assets_check,
        ]))

        run_button = QPushButton("开始测试")
        run_button.setObjectName("primaryButton")
        run_button.clicked.connect(self.run_episode)
        inspect_button = QPushButton("检查数据集")
        inspect_button.clicked.connect(self.inspect_dataset)
        train_button = QPushButton("训练 tiny world model")
        train_button.clicked.connect(self.train_tiny_model)
        export_hdf5_button = QPushButton("导出 StableWM HDF5")
        export_hdf5_button.clicked.connect(self.export_stablewm_hdf5)
        train_lewm_button = QPushButton("训练 LE-WM cost model")
        train_lewm_button.clicked.connect(self.train_lewm_cost_model)
        beamng_button = QPushButton("检查 BeamNG")
        beamng_button.clicked.connect(self.check_beamng)
        stop_button = QPushButton("停止 / 暂停")
        stop_button.setEnabled(False)
        stop_button.setToolTip("运行中断接口未完成")
        layout.addWidget(self._group("动作", [run_button, inspect_button, train_button, export_hdf5_button, train_lewm_button, beamng_button, stop_button]))
        layout.addStretch(1)
        return scroll

    def _build_workbench(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_overview_tab(), "总览")
        tabs.addTab(self._build_catalog_tab(), "运行时状态")
        tabs.addTab(self._build_unfinished_tab(), "预留接口")
        return tabs

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        metrics = QGridLayout()
        for index, key in enumerate(["steps", "done", "best_cost", "final_speed", "max_risk", "reward"]):
            card = MetricCard(key.replace("_", " ").title())
            self.metric_cards[key] = card
            metrics.addWidget(card, index // 3, index % 3)
        layout.addLayout(metrics)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.trajectory = TrajectoryCanvas()
        splitter.addWidget(self.trajectory)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("运行日志")
        self.episode_list = QListWidget()
        self.episode_list.itemClicked.connect(self.load_selected_episode)
        bottom.addWidget(self.log_view)
        bottom.addWidget(self.episode_list)
        bottom.setStretchFactor(0, 2)
        bottom.setStretchFactor(1, 1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return page

    def _build_catalog_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.catalog_table = QTableWidget(0, 4)
        self.catalog_table.setHorizontalHeaderLabels(["Type", "Name", "Available", "Message"])
        self.catalog_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.catalog_table)
        self.dataset_summary = QTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setMaximumHeight(180)
        self.dataset_summary.setPlaceholderText("数据集检查结果：NaN")
        layout.addWidget(self.dataset_summary)
        return page

    def _build_unfinished_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        for item in services.unfinished_features():
            frame = QFrame()
            frame.setObjectName("todoCard")
            row = QHBoxLayout(frame)
            row.addWidget(QLabel(item["name"]), 1)
            status = QLabel(item["status"])
            status.setObjectName("todoStatus")
            row.addWidget(status)
            layout.addWidget(frame)
        layout.addStretch(1)
        return page

    def refresh_catalogs(self) -> None:
        self.catalog = services.catalog_snapshot()
        self._fill_combo(self.backend_combo, self.catalog["backends"], "name", default="gym_heightmap")
        self._fill_combo(self.scenario_combo, self.catalog["scenarios"], "id", default="forest_trail_001")
        self._fill_combo(self.agent_combo, self.catalog["agents"], "name", default="rule_based")
        self._fill_combo(self.world_model_combo, self.catalog["world_models"], "name", default="simple_kinematic")
        self._fill_combo(self.planner_combo, [{"name": ""}] + self.catalog["planners"], "name", default="")
        self._fill_catalog_table()
        self._fill_episode_list()
        beamng = _find_named(self.catalog["backends"], "beamng")
        self.runtime_label.setText(f"BeamNG: {services.display_value(beamng.get('available') if beamng else None)}")
        self.log("状态已刷新")

    def run_episode(self) -> None:
        request = self._current_request()
        self.log(f"开始运行：backend={request.backend}, agent={request.agent}, planner={request.planner or 'none'}")
        self._run_task(
            lambda: services.run_episode_from_request(request),
            self._episode_finished,
            "episode failed",
        )

    def inspect_dataset(self) -> None:
        self.log("检查数据集...")
        root = self.dataset_root_edit.text().strip()
        adapter = self.adapter_edit.text().strip()
        sequence_id = self.sequence_combo.currentText().strip()
        self._run_task(
            lambda: services.inspect_dataset(root, adapter=adapter, sequence_id=sequence_id),
            self._dataset_inspected,
            "dataset inspect failed",
        )

    def train_tiny_model(self) -> None:
        root = self.dataset_root_edit.text().strip()
        if not root:
            self.log("训练 tiny world model 需要 dataset root")
            return
        output = self.model_path_edit.text().strip() or str(services.ROOT / "outputs" / "models" / "gui_tiny_world_model")
        adapter = self.adapter_edit.text().strip()
        sequence_id = self.sequence_combo.currentText().strip()
        self.log(f"训练 tiny world model -> {output}")
        self._run_task(
            lambda: services.train_tiny_world_model(root, output, adapter=adapter, sequence_id=sequence_id),
            self._training_finished,
            "training failed",
        )

    def export_stablewm_hdf5(self) -> None:
        root = self.dataset_root_edit.text().strip()
        output = self.stablewm_hdf5_edit.text().strip() or str(services.ROOT / "outputs" / "stablewm" / "gui_export.h5")
        adapter = self.adapter_edit.text().strip()
        sequence_id = self.sequence_combo.currentText().strip()
        self.log(f"导出 StableWM HDF5 -> {output}")
        self._run_task(
            lambda: services.export_lewm_hdf5(root, output, adapter=adapter, sequence_id=sequence_id, image_size=64),
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

    def check_beamng(self) -> None:
        status = services.beamng_status()
        self.log(f"BeamNG: {status.get('message', services.NAN_TEXT)}")
        self.refresh_catalogs()

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

    def _dataset_inspected(self, payload: dict[str, Any]) -> None:
        self.dataset_info = payload
        self.dataset_summary.setText(_compact_json(payload))
        self.sequence_combo.clear()
        for sequence_id in payload.get("sequences", []):
            self.sequence_combo.addItem(str(sequence_id))
        self.log(f"数据集 OK: {payload.get('dataset_id')} / frames={payload.get('frame_count')}")

    def _training_finished(self, payload: dict[str, Any]) -> None:
        self.model_path_edit.setText(str(payload.get("output_dir", "")))
        self.log(f"模型训练完成: {payload.get('model_path', services.NAN_TEXT)}")
        self.refresh_catalogs()

    def _hdf5_exported(self, payload: dict[str, Any]) -> None:
        self.stablewm_hdf5_edit.setText(str(payload.get("output_hdf5", "")))
        self.log(f"HDF5 导出完成: {payload.get('output_hdf5', services.NAN_TEXT)}")

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

    def _fill_catalog_table(self) -> None:
        rows: list[tuple[str, dict[str, Any]]] = []
        for kind in ["backends", "agents", "world_models", "planners"]:
            rows.extend((kind, item) for item in self.catalog.get(kind, []))
        self.catalog_table.setRowCount(len(rows))
        for row_index, (kind, item) in enumerate(rows):
            values = [kind, str(item.get("name", "")), services.display_value(item.get("available", True)), str(item.get("message", ""))]
            for column, value in enumerate(values):
                self.catalog_table.setItem(row_index, column, QTableWidgetItem(value))
        self.catalog_table.resizeColumnsToContents()

    def _fill_episode_list(self) -> None:
        self.episode_list.clear()
        for episode in self.catalog.get("episodes", [])[:40]:
            item = QListWidgetItem(str(episode.get("episode_id", services.NAN_TEXT)))
            item.setData(Qt.ItemDataRole.UserRole, episode.get("path"))
            self.episode_list.addItem(item)

    def _current_request(self) -> services.RunRequest:
        return services.RunRequest(
            backend=self.backend_combo.currentData() or self.backend_combo.currentText(),
            scenario=self.scenario_combo.currentData() or self.scenario_combo.currentText(),
            agent=self.agent_combo.currentData() or self.agent_combo.currentText(),
            seed=self.seed_spin.value(),
            max_steps=self.max_steps_spin.value(),
            record=self.record_check.isChecked(),
            record_arrays=self.record_arrays_check.isChecked(),
            world_model_type=self.world_model_combo.currentData() or self.world_model_combo.currentText(),
            world_model_path=self.model_path_edit.text().strip(),
            planner=self.planner_combo.currentData() or "",
            planner_horizon=self.planner_horizon_spin.value(),
            planner_samples=self.planner_samples_spin.value(),
            planner_iterations=self.planner_iterations_spin.value(),
            dataset_root=self.dataset_root_edit.text().strip(),
            sequence_id=self.sequence_combo.currentText().strip(),
            adapter=self.adapter_edit.text().strip(),
            load_assets=self.load_assets_check.isChecked(),
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

    def _combo(self, *, editable: bool = False) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(editable)
        return combo

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

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
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        for widget in widgets:
            layout.addWidget(widget)
        return group

    def _field(self, label: str, widget: QWidget) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(label)
        caption.setObjectName("fieldLabel")
        layout.addWidget(caption)
        layout.addWidget(widget)
        return frame

    def _with_button(self, widget: QWidget, button: QPushButton) -> QWidget:
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget, 1)
        layout.addWidget(button)
        return frame

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
    background: #0f151c;
    color: #e7edf3;
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 13px;
}
#sidebar {
    background: #111b24;
    border-right: 1px solid #22313d;
}
#appTitle {
    font-size: 22px;
    font-weight: 700;
}
#pageTitle {
    font-size: 24px;
    font-weight: 700;
}
#mutedText, .mutedText {
    color: #7f91a2;
}
#navButton {
    text-align: left;
    padding: 10px 12px;
    border-radius: 7px;
    background: transparent;
    border: 1px solid transparent;
}
#navButton:hover {
    background: #172633;
    border-color: #2a4357;
}
QGroupBox {
    border: 1px solid #253746;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    background: #121c25;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #9fb1c1;
}
QLineEdit, QComboBox, QSpinBox {
    background: #0d141b;
    border: 1px solid #2a3c4c;
    border-radius: 7px;
    padding: 8px;
}
QPushButton {
    background: #182735;
    border: 1px solid #2f485b;
    border-radius: 7px;
    padding: 9px 12px;
}
QPushButton:hover {
    background: #213648;
}
QPushButton:disabled {
    color: #667582;
    background: #141c24;
}
#primaryButton {
    background: #19a884;
    color: #06120f;
    font-weight: 700;
    border-color: #43d9ad;
}
#fieldLabel {
    color: #9fb1c1;
}
#metricCard, #todoCard {
    background: #121c25;
    border: 1px solid #253746;
    border-radius: 8px;
}
#metricTitle {
    color: #89a1b5;
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
QTabWidget::pane {
    border: 1px solid #253746;
    border-radius: 8px;
}
QTabBar::tab {
    background: #121c25;
    border: 1px solid #253746;
    padding: 9px 14px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}
QTabBar::tab:selected {
    background: #1a2a38;
    color: #43d9ad;
}
QTextEdit, QListWidget, QTableWidget {
    background: #0d141b;
    border: 1px solid #253746;
    border-radius: 8px;
}
QHeaderView::section {
    background: #172633;
    color: #c7d4df;
    border: 0;
    padding: 7px;
}
"""

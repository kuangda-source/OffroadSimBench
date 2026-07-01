# OffroadSimBench

<p align="center">
  <a href="#中文"><kbd>中文</kbd></a>
  <a href="#english"><kbd>English</kbd></a>
</p>

## 中文

OffroadSimBench 是一个面向越野自动驾驶的本地仿真、数据集回放、世界模型评测与桌面 GUI 平台。项目目标是把真实越野数据、可切换算法/模型、BeamNG 等仿真后端和可复现实验记录串成一条本地评测链路。

### 当前能力

- 多后端接口：`gym_heightmap`、`dataset_replay`、`beamng`、`ue5`。
- 数据集动态接入：内置 `offroad_sim_v1` 和 ORFD adapter。
- 算法可切换：`random`、`stop`、`rule_based`、`world_model`、`route_world_model`、`model_mpc`。
- 世界模型可切换：`simple_kinematic`、`tiny_learned`、`le_wm`。
- 路径规划可切换：`navigation_mpc`、`world_model_cem`、`le_wm_cem`。
- BeamNG 本地运行检测、连接 smoke、场景 reset/step、可视自动驾驶演示、episode 记录。
- stable-worldmodel HDF5 导出、LE-WM-compatible cost checkpoint 训练、`AutoCostModel + CEMSolver` 推理规划。
- PySide6 桌面 GUI：引导式 demo 首页、数据集与训练工作台、BeamNG 仿真工作台、ORFD 图像预览、HDF5 导出、LE-WM cost model 训练、BeamNG 可视自动驾驶、局部地形草案导出、episode 轨迹预览和日志。

### 环境

推荐使用仓库内 BeamNG 附带的 micromamba：

```powershell
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" create -y -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" -f environment.yml
```

运行测试：

```powershell
python -m pytest -q
```

### 常用命令

列出后端、算法、数据集适配器、世界模型和规划器：

```powershell
python -m offroad_sim.cli list --kind all
```

启动桌面 GUI：

```powershell
python -m desktop_app.main
# 或
offroad-sim-gui
```

创建并检查一个小 ORFD fixture：

```powershell
python scripts\create_mock_orfd_dataset.py outputs\mock_orfd_phase3 --frames 8
python examples\inspect_dataset.py outputs\mock_orfd_phase3 --adapter orfd
```

训练 tiny world model 并运行本地 CEM：

```powershell
python scripts\train_world_model.py outputs\mock_orfd_phase3 --adapter orfd --output outputs\models\phase3_tiny_world_model
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --planner world_model_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

导出 stable-worldmodel HDF5，训练 LE-WM-compatible cost checkpoint，并用 `le_wm_cem` 推理规划：

```powershell
python scripts\export_lewm_hdf5.py outputs\mock_orfd_phase3 outputs\stablewm\mock_orfd_phase3.h5 --adapter orfd --image-size 32
python scripts\train_lewm_cost_model.py outputs\stablewm\mock_orfd_phase3.h5 --output outputs\models\lewm_cost_smoke
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

The ORFD adapter can also read the official ICRA 2022 ZIP layout directly:

```powershell
python examples\inspect_dataset.py datasets\ORFD_Dataset_ICRA2022_ZIP --adapter orfd
python scripts\export_lewm_hdf5.py datasets\ORFD_Dataset_ICRA2022_ZIP outputs\stablewm\orfd_real_sample.h5 --adapter orfd --sequence-id training/c2021_0228_1819 --image-size 32
```

将 BeamNG 记录 episode 导出为 stable-worldmodel HDF5：

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_orfd_eval_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_lewm_smoke.h5
```

BeamNG 检查和 LE-WM CEM smoke：

```powershell
python examples\check_beamng_runtime.py
python examples\run_beamng_world_model.py --scenario configs\scenarios\beamng_orfd_eval.yaml --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

BeamNG 可视自动驾驶演示：

```powershell
python scripts\run_beamng_visible_demo.py --dataset-root datasets\ORFD_Dataset_ICRA2022_ZIP --adapter orfd --sequence-id training/c2021_0228_1819 --world-model-type le_wm --world-model outputs\models\lewm_orfd_real_c2021_0228_1819 --planner le_wm_cem --scenario beamng_visible_autodrive --vehicle configs\vehicles\ugv_medium.yaml --max-steps 600
```

可视演示脚本默认会等待 BeamNG 窗口出现、按可见节奏执行，并且跑完后不主动关闭 BeamNG。本机 BeamNG.tech 0.38.3 的 Direct3D11 自动启动路径可能黑屏，所以默认使用 Vulkan；需要回退 Direct3D11 时追加 `--beamng-gfx dx11`。当前可视演示使用 BeamNG 自带 `gridmap_v2` offroad 路段和 `drive_mode=ai_line` 原生执行器，不等同于 ORFD 真实场景重建。需要自动关闭时追加 `--close-beamng`；需要停留更久时追加 `--hold-open-sec 300`。

### 阶段三验收

默认使用小 ORFD fixture，不启动 BeamNG：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

同时启动 BeamNG 做 LE-WM CEM 规划 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -BeamNGConnect
```

### 阶段四验收

默认运行不启动 BeamNG 的配置、接口和 GUI smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1
```

启动 BeamNG 做 ORFD + LE-WM-compatible checkpoint + `le_wm_cem` 可视自动驾驶验收：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1 -BeamNGVisible -OrfdRoot datasets\ORFD_Dataset_ICRA2022_ZIP -SequenceId training/c2021_0228_1819 -WorldModelType le_wm -WorldModelPath outputs\models\lewm_orfd_real_c2021_0228_1819 -Planner le_wm_cem -MaxSteps 80
```

### LE-WM 边界

当前仓库已经打通 stable-worldmodel 运行依赖、HDF5 导出、LE-WM-compatible cost checkpoint 训练、`le_wm_cem` BeamNG 推理规划和可视 BeamNG 自动驾驶演示。这个 smoke checkpoint 用于验证系统闭环，不等同于 upstream LE-WM 大模型训练结果。后续要接完整 upstream LE-WM 训练，可复用当前 HDF5 边界、`AutoCostModel` 加载路径和 `route_world_model` Agent 入口。

### 数据与大文件

`BeamNG/`、`outputs/`、真实 ORFD 数据、模型 checkpoint、`node_modules/` 和打包产物不提交到 GitHub。

### Johnson Valley 闭环验收补充

当前仓库提供 `configs\tasks\beamng_johnson_valley_nav_001.yaml` 作为
Johnson Valley 原生越野地形上的当前验证区域/起终点任务。桌面 GUI 首页现在是
引导式 demo launcher：选择 `Demo config`，点击 `Start demo`，然后查看结果摘要。
Demo config 内部绑定 BeamNG region task、世界模型配置、planner 和可视运行默认参数。
默认配置指向这条已验证任务和
`outputs\region_navigation\johnson_valley_nav_test_train_v2_validated\model\lewm_cost_object.ckpt`；
点击首页 `Start demo` 会进入
`stablewm_lewm + navigation_mpc + model_mpc` 的 BeamNG 手动控车评估，默认规划参数为
horizon=6、samples=32、iterations=3；其他 Backend 会按当前 Backend/Scenario/Agent
运行普通 episode。桌面 GUI 的 BeamNG 页面只保留通用操作入口，固定 Johnson Valley 单次演示按钮已移除。
BeamNG 页面提供
`编辑/预览区域与起终点` 联合入口，可在同一个窗口里编辑区域、起点、终点和专家路线，
并可勾选实时预览，让后台 worker 在同一个 BeamNG 场景中增量刷新区域、起终点、路线标记和俯视相机。
预览默认使用俯视高视角，也可以在编辑窗口里切换相机模式和高度。编辑窗口还可以读取
当前 BeamNG 车辆世界坐标，并一键把当前位置写入区域点、起点、终点或路线点，用真实地图
画面辅助选点。区域编辑窗口是非模态独立窗口，主 GUI 可以最小化或移到一边，不会挡住
BeamNG。实时预览会自动加载 BeamNG 侧 `offroadSimBench/pointPicker` Lua 扩展；勾选
`BeamNG 窗口点击拾点` 后，先在编辑器里选择区域点/起点/终点/路线点模式，再直接在
BeamNG 窗口左键点击或短按地面，GUI 会以 50ms 轮询通过 Tech 通信消费
`cameraMouseRayCast()` 的世界坐标
并写入当前任务草稿。首次加载或切换地图时 GUI 不会阻塞，实时预览会合并加载期间产生的
重复请求，只保留最新一次草稿。BeamNG 预览会用更高对比度的半透明区域 mask、闭合边界线和区域点
标出当前可通行区域；区域点在编辑画布里可以拖动调整，未闭合/不合法的区域只在正式保存
任务时弹出错误提示。GUI 画布使用等比例世界坐标投影，避免编辑区形状相对 BeamNG 地图被拉伸。

评估阶段使用 `drive_mode=manual`，不是 BeamNG `ai_line`。区域闭环默认使用
`model_mpc`：它会在每一步生成候选 steer/throttle/brake 序列，用
LE-WM-compatible cost adapter 或世界模型预测为候选动作评分，再叠加目标距离、区域越界、
边界距离、动作平滑和低速恢复代价，最后只执行最优序列的第一步动作。`route_world_model`
仍保留为路线跟踪基线。2026-05-21 当前 checkpoint 在
`beamng_johnson_valley_nav_001.yaml` 上本地验收通过：272 步进入 12 m 目标半径，
最终距离 11.217 m，碰撞数为 0，评估阶段保持在区域内。运行时相机默认使用车辆后上方
约 45 度的 `follow` 视角，避免只看到车后尘土。

### 数据集与训练工作台

桌面 GUI 的 `数据集与训练` 页面现在作为独立训练可视化平台使用：可以检查和预览 ORFD 等数据集，选择训练/导出预设，查看当前训练配置摘要和最近指标曲线，运行 StableWM HDF5 导出、LE-WM-compatible cost model 训练或 tiny world model 训练。每次成功的训练或导出都会在产物目录旁写入 `training_run.json`，GUI 的 `Training results` 页会自动索引这些记录并展示产物路径、参数和指标。`LE-WM full self-supervised`、`TD-MPC2` 和 `DreamerV3` 目前作为未完成的可插拔预设保留，不会伪造训练结果。

训练记录支持 `history` 字段保存 loss、RMSE、frame count 等曲线数据；GUI 会在训练结果页绘制可用的主指标曲线，没有真实 history 时只展示单点指标或 NaN。

外部数据集可以通过 `dataset_manifest.yaml` 接入 `manifest_dataset` adapter，也可以在 GUI 的数据集页点击 `导入 dataset manifest` 导入。导入时系统会把 manifest 安装到 `configs/datasets/<dataset_id>/`，并把 sequence 的相对 root 改写为绝对路径，方便之后在数据集目录下复用。manifest 描述 sequence 根目录、pose CSV 和传感器/标签资产模板，系统会把它转换为统一的 `DatasetSequence`：

```yaml
adapter: manifest_dataset
dataset_id: custom_drive
dataset_type: camera_depth_labels
sequences:
  - id: clip_001
    root: clip_001
    pose_csv: poses.csv
    assets:
      front_rgb: images/{frame_id}.png
      depth: depth/{frame_id}.npy
      label: labels/{frame_id}.png
```

外部模型或算法训练可以通过 `configs/trainers/*/trainer.yaml` 或 `configs/trainers/*.yaml` 接入，也可以在 GUI 的训练页点击 `Import trainer manifest` 导入。训练器 manifest 声明 entrypoint、参数 schema 和命令参数模板；GUI 会把它显示为训练预设，并把参数 JSON、stdout/stderr、metrics/history 和 `training_run.json` 一起记录下来：

```yaml
trainer_id: my_world_model
display_name: My World Model
runtime: python
entrypoint: train.py
parameters:
  epochs:
    type: int
    default: 10
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
  - "--epochs"
  - "{params.epochs}"
outputs:
  artifact_type: checkpoint
```

## English

OffroadSimBench is a local off-road autonomous-driving simulation, dataset replay, world-model evaluation, and desktop GUI platform. It connects real off-road datasets, switchable algorithms/models, BeamNG simulator backends, and reproducible episode records into one local benchmark loop.

### Current Capabilities

- Multi-backend interface: `gym_heightmap`, `dataset_replay`, `beamng`, and `ue5`.
- Dynamic dataset ingestion with built-in `offroad_sim_v1` and ORFD adapters.
- Switchable agents: `random`, `stop`, `rule_based`, `world_model`, `route_world_model`, and `model_mpc`.
- Switchable world models: `simple_kinematic`, `tiny_learned`, and `le_wm`.
- Switchable planners: `navigation_mpc`, `world_model_cem`, and `le_wm_cem`.
- BeamNG runtime detection, connection smoke tests, scenario reset/step, visible autonomous-driving demo, and episode recording.
- stable-worldmodel HDF5 export, LE-WM-compatible cost checkpoint training, and `AutoCostModel + CEMSolver` planning.
- Region self-supervised BeamNG collection and training scaffold: `region_explorer`, `world_model_direct`, terminal goal braking, acceptance metrics, and `scripts\run_region_self_supervised_world_model.py`.
- PySide6 desktop GUI with a guided demo overview, Dataset and Training workbench, BeamNG Simulation workbench, generic dataset frame preview, HDF5 export, external trainer manifests, LE-WM cost-model training, visible BeamNG autodrive, local terrain draft export, episode trajectory preview, and logs.

### Dataset And Training Workbench

The desktop GUI `Dataset and Training` page is now a standalone training-visualization workbench. It can register manifest datasets from ordinary folders, inspect and preview registered datasets, choose a reusable training/export config, review the current training config summary and latest metric curve, run StableWM HDF5 export, train the local LE-WM-compatible cost model, run imported trainer scripts, or train the tiny world model. Successful training/export actions write a `training_run.json` record next to the produced artifact, and the GUI `Training results` tab indexes those records so users can review artifact paths, parameters, logs, and metrics. A successful run with a runnable checkpoint or `model.json` artifact can be promoted with `Register latest training artifact`, which saves a world-model config and writes the promoted config back into `training_run.json`. `LE-WM full self-supervised`, `TD-MPC2`, and `DreamerV3` are exposed as unfinished pluggable presets without fabricating results.

Training configs combine a dataset root, adapter, sequence, training preset, output path, and JSON parameters into one reusable GUI selection. Built-in configs include `ORFD StableWM HDF5 export`, `ORFD tiny world model`, and `Smoke tiny world model`. The smoke config creates a tiny ORFD-style dataset under `outputs/training_studio_smoke/datasets/mock_orfd`, trains a local tiny model, writes a `training_run.json`, and provides metric history for the GUI curves. Users can edit the current fields and click `Save training config` to persist a new entry in `configs/training_configs.json`. `Start training/export` now runs the current training config through a single service boundary, whether the preset is built in or backed by an imported trainer manifest. The `Validate config` action performs a dry run before training: it checks dataset availability, resolves the selected trainer manifest, coerces parameter types, reports missing required parameters, and shows the external command preview when the preset is backed by a local script.

Training records support a `history` field for loss, RMSE, frame count, or other curve data. The GUI plots the primary available metric in the training results tab, lists the available curve names, and falls back to single-point metrics or NaN when no real history exists.

External trainers can report metrics through a JSON object on stdout or through sidecar files in the selected output directory. Supported sidecars are `metrics.json` for final scalar metrics, `history.json` for metric arrays, and `events.jsonl` for per-step JSON events such as `{"step": 1, "loss": 0.9}`.

BeamNG region self-supervised runs also write a `training_run.json` with the trained model path plus acceptance metrics such as `goal_success`, `min_goal_distance`, `final_goal_distance`, and `collision_count`, so the Training Results tab can index the simulator-trained model instead of leaving it as an opaque folder. When the GUI self-supervised workflow reaches `goal_success`, it also saves a validated `world_model_direct + tiny_learned` world-model config with the source `training_run.json` and validation metrics, making the trained model selectable from the Overview and BeamNG Simulation pages. Failed or collection-insufficient runs remain visible in Training Results but are not promoted to runnable model configs.

External datasets can use `dataset_manifest.yaml` with the `manifest_dataset` adapter, be imported from the GUI with `Import dataset manifest`, or be registered directly from the current Dataset page fields with `Save dataset manifest`. Registering from the GUI writes `configs/datasets/<dataset_id>/dataset_manifest.yaml`, keeps the original dataset directory as `source_root`, and stores user-provided sequence roots and asset glob/templates. Imported manifests still rewrite relative sequence roots to absolute paths so the dataset remains reusable from the catalog. The manifest declares sequence roots, optional pose/action CSV files, and sensor/label asset templates, and the platform converts them into the unified `DatasetSequence` contract:

For common driving dataset layouts, the GUI can also fill the manifest sequence JSON automatically with `Auto-detect sequences`. It looks for sequence folders containing pose CSV files such as `poses.csv`, plus common asset folders such as `images/`, `rgb/`, `depth/`, `masks/`, `lidar/`, `local_bev/`, or `terrain_map/`. Users can then review or edit the generated manifest before saving it.

```yaml
adapter: manifest_dataset
dataset_id: custom_drive
dataset_type: camera_depth_labels
sequences:
  - id: clip_001
    root: clip_001
    pose_csv: poses.csv
    assets:
      front_rgb: images/{frame_id}.png
      depth: depth/{frame_id}.npy
      label: labels/{frame_id}.png
```

External model or algorithm training can be added through `configs/trainers/*/trainer.yaml` or `configs/trainers/*.yaml`, or imported from the GUI with `Import trainer manifest`. A trainer manifest declares the entrypoint, parameter schema, and command template; the GUI exposes it as a training preset and records the parameter JSON, stdout/stderr, metrics/history, and `training_run.json`:

```yaml
trainer_id: my_world_model
display_name: My World Model
runtime: python
entrypoint: train.py
parameters:
  epochs:
    type: int
    default: 10
    required: true
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
  - "--epochs"
  - "{params.epochs}"
outputs:
  artifact_type: checkpoint
  artifact_path: model.ckpt
  metrics_file: metrics.json
  history_file: history.json
  events_file: events.jsonl
```

If an external algorithm does not already ship a `trainer.yaml`, the GUI can
create one from the `Model training` tab: fill `Trainer entrypoint`, keep or
edit the JSON `Trainer arguments` template, optionally add a JSON `Trainer
parameter schema`, and click `Save trainer from script`. For a quicker
experiment, fill the dataset fields, script path, and `Training parameters`,
then click `Run script now`; the GUI saves the generated trainer plus a reusable
training config and launches the same training runner. When the argument
template is left at the default dataset/output pair, parameter names are turned
into command flags automatically, e.g. `{"epochs": 10}` becomes
`--epochs {params.epochs}`. The saved trainer is installed under
`configs/trainers/` and immediately appears in the `Training preset` selector.
A `training_config.yaml` bundle can also inline the same trainer block under
`trainer:` instead of referencing `trainer_manifest`.

### Environment

The recommended setup uses the micromamba runtime bundled with the local BeamNG install:

```powershell
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" create -y -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" -f environment.yml
```

Run tests:

```powershell
python -m pytest -q
```

### Common Commands

List backends, agents, dataset adapters, world models, and planners:

```powershell
python -m offroad_sim.cli list --kind all
```

Start the desktop GUI:

```powershell
python -m desktop_app.main
# or
offroad-sim-gui
```

Create and inspect a small ORFD fixture:

```powershell
python scripts\create_mock_orfd_dataset.py outputs\mock_orfd_phase3 --frames 8
python examples\inspect_dataset.py outputs\mock_orfd_phase3 --adapter orfd
```

Train the tiny world model and run local CEM:

```powershell
python scripts\train_world_model.py outputs\mock_orfd_phase3 --adapter orfd --output outputs\models\phase3_tiny_world_model
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --planner world_model_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

Export stable-worldmodel HDF5, train an LE-WM-compatible cost checkpoint, and run `le_wm_cem` planning:

```powershell
python scripts\export_lewm_hdf5.py outputs\mock_orfd_phase3 outputs\stablewm\mock_orfd_phase3.h5 --adapter orfd --image-size 32
python scripts\train_lewm_cost_model.py outputs\stablewm\mock_orfd_phase3.h5 --output outputs\models\lewm_cost_smoke
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

The ORFD adapter can also read the official ICRA 2022 ZIP layout directly:

```powershell
python examples\inspect_dataset.py datasets\ORFD_Dataset_ICRA2022_ZIP --adapter orfd
python scripts\export_lewm_hdf5.py datasets\ORFD_Dataset_ICRA2022_ZIP outputs\stablewm\orfd_real_sample.h5 --adapter orfd --sequence-id training/c2021_0228_1819 --image-size 32
```

Export recorded BeamNG episodes to stable-worldmodel HDF5:

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_orfd_eval_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_lewm_smoke.h5
```

Derive actions from BeamNG state deltas when the episode was driven by the
simulator route controller:

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_visible_autodrive_route_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_map_lewm.h5 --actions-from-state
```

Run the BeamNG-map LE-WM closed loop:

```powershell
python scripts\run_beamng_lewm_closed_loop.py --collect-steps 160 --eval-steps 120 --output-dir outputs\beamng_map_lewm\demo
```

Run the region self-supervised world-model scaffold:

```powershell
python scripts\run_region_self_supervised_world_model.py configs\tasks\beamng_johnson_valley_nav_001.yaml --evaluation-agent world_model_direct --evaluation-route-mode route_free --collect-steps 1000 --eval-steps 1200
```

The GUI uses this route-free mode for its region self-supervised action: it
collects exploration data, trains `tiny_learned`, then evaluates direct
start-to-goal control without injecting the task route. The current Johnson
Valley task can still need better exploration coverage and planning priors to
reach difficult goals reliably; see
`docs\reports\2026-05-29_region_self_supervised_blocker.md`.

Run a region/start/goal navigation loop. The collection stage uses
`expert_route`; the evaluation stage removes that route and keeps only
`start_pose + goal` in the task contract. BeamNG episodes terminate when the
vehicle enters the configured goal radius, and the summary reports both final
and minimum goal distance:

```powershell
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_johnson_valley_nav_001.yaml --algorithm local_lewm_cost --collect-steps 240 --eval-steps 520 --output-dir outputs\region_navigation\johnson_valley_nav_test_train
```

Johnson Valley now has a repeatable stock-terrain route task at
`configs\tasks\beamng_johnson_valley_nav_001.yaml`. The desktop GUI overview
page is now a guided demo launcher: choose a `Demo config`, click `Start demo`,
and review the result summary. The selected demo config owns the BeamNG region
task, saved world-model config, planner, and visible runtime defaults. The
default demo config points at this validated task and
`outputs\region_navigation\johnson_valley_nav_test_train_v2_validated\model\lewm_cost_object.ckpt`.
Clicking `Start demo` runs the selected task through
`stablewm_lewm + navigation_mpc + model_mpc` in BeamNG with default planner
settings `horizon=6`, `samples=32`, and `iterations=3`. The BeamNG Simulation
workbench keeps generic operations such as region editing, runtime checks, model
config selection, evaluation, and terrain draft export; one-off Johnson Valley
demo buttons have been removed. The GUI BeamNG Simulation workbench can preview
the selected region, start point, goal point, and route
markers before running the closed loop. The evaluation stage uses
`drive_mode=manual`, not
BeamNG `ai_line`; the default `model_mpc` agent generates candidate
steer/throttle/brake sequences, scores them through the LE-WM-compatible cost
adapter or a world-model rollout, adds goal progress and region-boundary costs,
and sends only the first action of the best sequence to BeamNG. `route_world_model`
remains available as a route-tracking baseline.

Run the standard demo acceptance from the command line:

```powershell
python scripts\demo_acceptance.py --demo-config johnson_valley_standard_demo --runs 1
```

The JSON report includes goal reached, collision count, final distance,
trajectory length, average speed, and whether recovery logic was triggered.
Use `--runs 2` or `--runs 3` to check repeatability.

```powershell
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_johnson_valley_nav_001.yaml --algorithm stablewm_lewm --algorithm-model-path outputs\region_navigation\johnson_valley_nav_test_train_v2_validated\model\lewm_cost_object.ckpt --eval-steps 520 --planner navigation_mpc --planner-horizon 6 --planner-samples 32 --planner-iterations 3 --keep-beamng-open
```

Local acceptance on 2026-05-21 passed the current Johnson Valley checkpoint
with manual model-guided control: 272 evaluation steps, final goal distance
of 11.217 m against a 12 m goal radius, zero collisions, and the final pose
inside the selected region. Runtime BeamNG evaluation now defaults to a high
rear follow camera so the vehicle remains visible instead of being hidden by
rear dust.

In the desktop GUI, open the BeamNG Simulation workbench and use `编辑/预览区域与起终点` to
edit a polygonal region, start point, goal point, and optional expert route
while refreshing a BeamNG preview from the same dialog. Enable realtime preview
to update the same BeamNG scene from a background worker with region,
start/goal, route marker, and top-down camera changes. Preview defaults to a high top-down camera,
and the camera mode/height are adjustable in the editor. The editor can also read
the current BeamNG vehicle world pose and apply it as a region point, start,
goal, or route waypoint, so operators can pick points from the real map view.
The region editor is a non-modal independent window, so the main GUI can be
minimized or moved away while BeamNG stays visible. Realtime preview
automatically loads the BeamNG-side `offroadSimBench/pointPicker` Lua extension.
With `BeamNG 窗口点击拾点` enabled, select the editor mode for region/start/goal
or route, then left-click or briefly hold on the BeamNG render window; the GUI
polls every 50 ms and consumes the `cameraMouseRayCast()` world coordinate
through Tech communication before writing it into the task draft. BeamNG preview
loading does not block the GUI; repeated drafts are coalesced so only the newest
pending preview runs after the active load finishes. BeamNG preview draws a
higher-contrast translucent region mask, closed outline, and region point
markers; region points are draggable in the editor, invalid regions only show
blocking warnings when saving, and the GUI canvas preserves the same world-axis
scale as the BeamNG map.

BeamNG training v1 is exposed as an explicit two-step GUI workflow on the
BeamNG Simulation page. Select a Johnson Valley region task, click
`采集训练数据` to run `region_explorer` rollouts and save a reusable
`region_training_collection.json`, then click `训练模型` to fit a
`tiny_learned` world model from those recorded episodes. The training step
writes `training_run.json`, registers the trained model as a selectable world
model config, and updates the BeamNG page so `开始评估` can immediately run the
same task with the trained model.

Saved tasks default to `evaluation_drive_mode: manual`, which means
the `OffroadAgent`/planner commands control the vehicle during evaluation.
`evaluation_drive_mode: ai_line` remains available only for BeamNG-native
visual smoke tests, because it follows a simulator line rather than proving
model navigation.
Navigation-region BeamNG tasks also default to `manual_control_is_adas: false`
for evaluation so model actions are sent as direct steering/throttle/brake
commands. The Johnson Valley demo keeps `steps_per_action` at 6 and the
collection `ai_line_speed` at 8.0 m/s; `model_mpc` also caps high-speed steering
and samples braking candidates so sharp turns slow down instead of sliding.
`start_pose.yaw` stays in OffroadSimBench's XY convention
(`yaw=0` faces +X); the task exporter converts it to BeamNG's vehicle
quaternion convention when spawning the vehicle.

The selected region is propagated into `Observation.info` during BeamNG runs.
Local CEM planners and the LE-WM-compatible cost checkpoint use it as a
trajectory constraint, penalizing candidates that leave the chosen polygon or
drive too close to the boundary.

Dataset-to-BeamNG map conversion notes live in
[`docs/dataset_to_beamng_map.md`](docs/dataset_to_beamng_map.md). The current
recommendation is to validate planner cost maps first, then export heightmap
drafts, and only later package full BeamNG levels.

List and inspect pluggable algorithm adapters:

```powershell
python -m offroad_sim.cli algorithms list
python -m offroad_sim.cli algorithms inspect local_lewm_cost --json
python -m offroad_sim.cli algorithms inspect stablewm_lewm --json
```

Third-party algorithms can be added under `algorithms/<name>/` with an
`algorithm.yaml` manifest and an `adapter.py` class. `local_lewm_cost` prepares
BeamNG episode data and trains the local LE-WM-compatible smoke checkpoint.
`stablewm_lewm` loads an existing stable-worldmodel / upstream LE-WM checkpoint
directly and exposes it as an action-cost scorer for `model_mpc`.

Train and validate the Johnson Valley region task saved from the GUI:

```powershell
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_johnson_valley_nav_001.yaml --algorithm local_lewm_cost --collect-steps 240 --eval-steps 520 --planner navigation_mpc --output-dir outputs\region_navigation\johnson_valley_nav_test_train
```

Use a real checkpoint without retraining the local smoke model:

```powershell
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_johnson_valley_nav_001.yaml --algorithm stablewm_lewm --algorithm-model-path D:\models\lewm\orfd\lewm_object.ckpt --eval-steps 520 --planner navigation_mpc --planner-horizon 6 --planner-samples 32 --planner-iterations 3 --keep-beamng-open
```

If the upstream checkpoint was downloaded as HuggingFace `weights.pt` +
`config.json`, convert it first:

```powershell
$env:LE_WM_HOME = "D:\programs\le-wm"
python scripts\convert_lewm_hf_checkpoint.py D:\models\lewm_hf\pusht D:\models\lewm\pusht
```

BeamNG checks and LE-WM CEM smoke:

```powershell
python examples\check_beamng_runtime.py
python examples\run_beamng_world_model.py --scenario configs\scenarios\beamng_orfd_eval.yaml --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

Visible BeamNG autonomous-driving demo:

```powershell
python scripts\run_beamng_visible_demo.py --dataset-root datasets\ORFD_Dataset_ICRA2022_ZIP --adapter orfd --sequence-id training/c2021_0228_1819 --world-model-type le_wm --world-model outputs\models\lewm_orfd_real_c2021_0228_1819 --planner le_wm_cem --scenario beamng_visible_autodrive --vehicle configs\vehicles\ugv_medium.yaml --max-steps 600
```

The visible demo script now waits for the BeamNG window, runs with human-visible pacing, and leaves BeamNG open when the episode ends. It defaults to Vulkan because the local BeamNG.tech 0.38.3 Direct3D11 auto-launch path can render a black window; add `--beamng-gfx dx11` to force Direct3D11. The current visual demo uses a stock BeamNG `gridmap_v2` offroad route with `drive_mode=ai_line`; it is not full ORFD scene reconstruction. Add `--close-beamng` to close it automatically or `--hold-open-sec 300` to keep the Python process attached longer.

### Phase 3 Acceptance

Default fixture run without launching BeamNG:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

Launch BeamNG for an LE-WM CEM planning smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -BeamNGConnect
```

### Phase 4 Acceptance

Run the configuration, interface, and GUI smoke checks without launching BeamNG:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1
```

Launch BeamNG for the ORFD + LE-WM-compatible checkpoint + `le_wm_cem` visible autodrive acceptance run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1 -BeamNGVisible -OrfdRoot datasets\ORFD_Dataset_ICRA2022_ZIP -SequenceId training/c2021_0228_1819 -WorldModelType le_wm -WorldModelPath outputs\models\lewm_orfd_real_c2021_0228_1819 -Planner le_wm_cem -MaxSteps 80
```

### LE-WM Boundary

The repository now has stable-worldmodel runtime dependencies, HDF5 export, LE-WM-compatible cost checkpoint training, `le_wm_cem` BeamNG inference/planning, and a visible BeamNG autonomous-driving demo wired into the system. The smoke checkpoint validates the system loop; it is not a full upstream LE-WM research model. Full upstream LE-WM training can reuse the current HDF5 boundary, `AutoCostModel` loading path, and `route_world_model` Agent entrypoint.

### Data And Large Files

`BeamNG/`, `outputs/`, real ORFD data, model checkpoints, `node_modules/`, and package/build artifacts are not committed to GitHub.

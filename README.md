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
- 算法可切换：`random`、`stop`、`rule_based`、`world_model`。
- 世界模型可切换：`simple_kinematic`、`tiny_learned`、`le_wm`。
- 路径规划可切换：`world_model_cem`、`le_wm_cem`。
- BeamNG 本地运行检测、连接 smoke、场景 reset/step、episode 记录。
- stable-worldmodel HDF5 导出、LE-WM-compatible cost checkpoint 训练、`AutoCostModel + CEMSolver` 推理规划。
- PySide6 桌面 GUI：运行配置、数据集检查、模型训练、HDF5 导出、LE-WM cost model 训练、episode 轨迹预览和日志。

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

将 BeamNG 记录 episode 导出为 stable-worldmodel HDF5：

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_orfd_eval_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_lewm_smoke.h5
```

BeamNG 检查和 LE-WM CEM smoke：

```powershell
python examples\check_beamng_runtime.py
python examples\run_beamng_world_model.py --scenario configs\scenarios\beamng_orfd_eval.yaml --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

### 阶段三验收

默认使用小 ORFD fixture，不启动 BeamNG：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

同时启动 BeamNG 做 LE-WM CEM 规划 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -BeamNGConnect
```

### LE-WM 边界

当前仓库已经打通 stable-worldmodel 运行依赖、HDF5 导出、LE-WM-compatible cost checkpoint 训练和 `le_wm_cem` BeamNG 推理规划。这个 smoke checkpoint 用于验证系统闭环，不等同于 upstream LE-WM 大模型训练结果。后续要接完整 upstream LE-WM 训练，可复用当前 HDF5 边界和 `AutoCostModel` 加载路径。

### 数据与大文件

`BeamNG/`、`outputs/`、真实 ORFD 数据、模型 checkpoint、`node_modules/` 和打包产物不提交到 GitHub。

## English

OffroadSimBench is a local off-road autonomous-driving simulation, dataset replay, world-model evaluation, and desktop GUI platform. It connects real off-road datasets, switchable algorithms/models, BeamNG simulator backends, and reproducible episode records into one local benchmark loop.

### Current Capabilities

- Multi-backend interface: `gym_heightmap`, `dataset_replay`, `beamng`, and `ue5`.
- Dynamic dataset ingestion with built-in `offroad_sim_v1` and ORFD adapters.
- Switchable agents: `random`, `stop`, `rule_based`, and `world_model`.
- Switchable world models: `simple_kinematic`, `tiny_learned`, and `le_wm`.
- Switchable planners: `world_model_cem` and `le_wm_cem`.
- BeamNG runtime detection, connection smoke tests, scenario reset/step, and episode recording.
- stable-worldmodel HDF5 export, LE-WM-compatible cost checkpoint training, and `AutoCostModel + CEMSolver` planning.
- PySide6 desktop GUI for run configuration, dataset inspection, model training, HDF5 export, LE-WM cost-model training, episode trajectory preview, and logs.

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

Export recorded BeamNG episodes to stable-worldmodel HDF5:

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_orfd_eval_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_lewm_smoke.h5
```

BeamNG checks and LE-WM CEM smoke:

```powershell
python examples\check_beamng_runtime.py
python examples\run_beamng_world_model.py --scenario configs\scenarios\beamng_orfd_eval.yaml --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

### Phase 3 Acceptance

Default fixture run without launching BeamNG:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

Launch BeamNG for an LE-WM CEM planning smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -BeamNGConnect
```

### LE-WM Boundary

The repository now has stable-worldmodel runtime dependencies, HDF5 export, LE-WM-compatible cost checkpoint training, and `le_wm_cem` BeamNG inference/planning wired into the system. The smoke checkpoint validates the system loop; it is not a full upstream LE-WM research model. Full upstream LE-WM training can reuse the current HDF5 boundary and `AutoCostModel` loading path.

### Data And Large Files

`BeamNG/`, `outputs/`, real ORFD data, model checkpoints, `node_modules/`, and package/build artifacts are not committed to GitHub.

# OffroadSimBench

<p align="center">
  <a href="#中文"><kbd>中文</kbd></a>
  <a href="#english"><kbd>English</kbd></a>
</p>

## 中文

OffroadSimBench 是一个面向越野自动驾驶的本地仿真、数据集回放、世界模型评测与可视化平台。项目目标是把真实越野数据、可切换算法/模型、BeamNG 等仿真后端和 dashboard 演示串成一条可复现的评测链路。

### 当前能力

- 多后端接口：`gym_heightmap`、`dataset_replay`、`beamng`、`ue5`。
- 数据集动态接入：内置 `offroad_sim_v1` 和 ORFD adapter。
- 算法可切换：`random`、`stop`、`rule_based`、`world_model` 通过 agent registry 选择。
- 世界模型可切换：`simple_kinematic`、`tiny_learned`、可选 `le_wm` wrapper。
- BeamNG 本地运行检测、连接 smoke、场景 reset/step、best-effort sensor payload。
- React/Vite dashboard：后端、场景、算法、世界模型、数据集路径、回放与指标可视化。
- 阶段验收脚本：Phase 2 dashboard/BeamNG demo，Phase 3 ORFD + world model + optional BeamNG run。

### 安装环境

推荐使用仓库内 BeamNG 附带的 micromamba：

```powershell
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" create -y -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" -f environment.yml
```

运行测试：

```powershell
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" run -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" python -m pytest
```

### 常用命令

列出后端、算法、数据集适配器和世界模型：

```powershell
python -m offroad_sim.cli list --kind all
```

创建一个小 ORFD fixture：

```powershell
python scripts\create_mock_orfd_dataset.py outputs\mock_orfd_phase3 --frames 8
```

检查 ORFD/真实数据集：

```powershell
python examples\inspect_dataset.py outputs\mock_orfd_phase3 --adapter orfd
```

训练小世界模型：

```powershell
python scripts\train_world_model.py outputs\mock_orfd_phase3 --adapter orfd --output outputs\models\phase3_tiny_world_model
```

用可切换世界模型跑 dataset replay：

```powershell
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --max-steps 3 --record
```

启用 CEM 路径规划：

```powershell
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --planner world_model_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

检查 BeamNG：

```powershell
python examples\check_beamng_runtime.py
python examples\check_beamng_runtime.py --connect --steps 1
```

启动 dashboard：

```powershell
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
cd dashboard/frontend
npm run dev
```

### 阶段三验收

默认使用小 ORFD fixture，不启动 BeamNG：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

使用真实 ORFD 数据集：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -OrfdRoot D:\datasets\ORFD
```

同时启动 BeamNG 做世界模型仿真：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -OrfdRoot D:\datasets\ORFD -BeamNGConnect
```

### LE-WM 集成边界

项目预留了 `le_wm` 世界模型入口，对应 upstream 仓库：

https://github.com/lucas-maes/le-wm

OffroadSimBench 不 vendoring LE-WM 源码；安装 upstream 依赖、设置 `LE_WM_HOME` 或提供 checkpoint 后，可以通过 `--world-model-type le_wm --planner le_wm_cem` 使用 stable-worldmodel 的 `AutoCostModel + CEMSolver` 做路径规划。checkpoint-specific 的输入键、归一化和图像尺寸可以在 `offroad_sim/planning/stablewm.py` 内调整，agent、CLI、dashboard 和 runner 不需要为换模型硬改代码。

规划运行时可以用 `python -m pip install -e .[lewm]` 安装；如果要按 upstream LE-WM 流程训练/评估，则使用更重的 `python -m pip install -e .[lewm-train]`。当前 bridge 会懒加载并缓存 `AutoCostModel + CEMSolver`，避免每个控制步重复读取 checkpoint。

### 数据与大文件

`BeamNG/`、`outputs/`、真实 ORFD 数据、模型 checkpoint、`node_modules/`、dashboard build artifacts 不提交到 GitHub。

## English

OffroadSimBench is a local off-road autonomous-driving simulation, dataset replay, world-model evaluation, and visualization platform. The goal is to connect real off-road datasets, switchable algorithms/models, BeamNG simulator backends, and a dashboard into one reproducible benchmark loop.

### Current Capabilities

- Multi-backend interface: `gym_heightmap`, `dataset_replay`, `beamng`, and `ue5`.
- Dynamic dataset ingestion with built-in `offroad_sim_v1` and ORFD adapters.
- Switchable algorithms through an agent registry: `random`, `stop`, `rule_based`, and `world_model`.
- Switchable world models: `simple_kinematic`, `tiny_learned`, and optional `le_wm` wrapper.
- BeamNG runtime detection, connection smoke test, scenario reset/step, and best-effort sensor payload mapping.
- React/Vite dashboard for backend, scenario, algorithm, world model, dataset path, replay, and metrics.
- Acceptance scripts for Phase 2 dashboard/BeamNG demo and Phase 3 ORFD + world model + optional BeamNG run.

### Environment

The recommended setup uses the micromamba runtime bundled with the local BeamNG install:

```powershell
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" create -y -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" -f environment.yml
```

Run tests:

```powershell
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" run -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" python -m pytest
```

### Common Commands

List backends, agents, dataset adapters, and world models:

```powershell
python -m offroad_sim.cli list --kind all
```

Create a tiny ORFD fixture:

```powershell
python scripts\create_mock_orfd_dataset.py outputs\mock_orfd_phase3 --frames 8
```

Inspect ORFD or a real dataset root:

```powershell
python examples\inspect_dataset.py outputs\mock_orfd_phase3 --adapter orfd
```

Train the small learned world model:

```powershell
python scripts\train_world_model.py outputs\mock_orfd_phase3 --adapter orfd --output outputs\models\phase3_tiny_world_model
```

Run dataset replay with a switchable world-model agent:

```powershell
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --max-steps 3 --record
```

Enable CEM path planning:

```powershell
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --planner world_model_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

Check BeamNG:

```powershell
python examples\check_beamng_runtime.py
python examples\check_beamng_runtime.py --connect --steps 1
```

Start the dashboard:

```powershell
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
cd dashboard/frontend
npm run dev
```

### Phase 3 Acceptance

Default fixture run without launching BeamNG:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

Use a real ORFD dataset root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -OrfdRoot D:\datasets\ORFD
```

Launch BeamNG for a world-model simulation run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -OrfdRoot D:\datasets\ORFD -BeamNGConnect
```

### LE-WM Boundary

The project reserves a `le_wm` world-model entry for the upstream repository:

https://github.com/lucas-maes/le-wm

OffroadSimBench does not vendor LE-WM. Install the upstream runtime, set `LE_WM_HOME` or provide a checkpoint, then use `--world-model-type le_wm --planner le_wm_cem` to run stable-worldmodel `AutoCostModel + CEMSolver` path planning. Checkpoint-specific input keys, normalization, and image size live in `offroad_sim/planning/stablewm.py`; agents, CLI commands, dashboard controls, and the runner do not need hard-coded model changes.

Install the planning runtime with `python -m pip install -e .[lewm]`. For upstream LE-WM training/evaluation workflows, use the heavier `python -m pip install -e .[lewm-train]`. The bridge lazy-loads and caches `AutoCostModel + CEMSolver` instead of reloading the checkpoint on every control step.

### Data And Large Files

`BeamNG/`, `outputs/`, real ORFD data, model checkpoints, `node_modules/`, and dashboard build artifacts are not committed to GitHub.

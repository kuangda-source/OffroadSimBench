# OffroadSimBench 第一阶段总验收与总结报告

验收日期：2026-05-14  
验收状态：通过  
对应阶段：第一阶段，本地可运行基座与多后端接口框架

## 1. 阶段目标

第一阶段的目标是把 OffroadSimBench 从项目骨架推进到一个可以本地运行、可测试、可扩展的仿真评测基座。重点不是一次性完成所有真实仿真器能力，而是把核心抽象、配置、数据集接入、轻量训练环境、世界模型接口、dashboard 和 CLI 打通，让后续 BeamNG/UE5 接入有稳定落点。

本阶段已经完成：

- `OffroadAgent` 与 `OffroadSimBackend` 两个核心接口。
- 车辆配置、场景配置、指标系统、episode 录制与回放。
- `GymHeightmapBackend` 本地轻量仿真后端。
- dataset adapter/registry 与 `DatasetReplayBackend`。
- 多后端 registry，统一管理 `gym_heightmap`、`dataset_replay`、`beamng`、`ue5`。
- BeamNG 可选后端骨架和运行状态检查。
- UE5 TCP JSON bridge 与 mock server。
- 世界模型接口、`SimpleKinematicWorldModel` 和 `WorldModelAgent`。
- Gymnasium wrapper。
- 共享 episode runner、CLI、FastAPI dashboard 后端和 React/Vite dashboard 前端。
- 第一阶段测试与 smoke test 矩阵。

## 2. 架构验收

核心架构符合第一阶段预期：

- Agent 代码通过 `offroad_sim.agents.OffroadAgent` 与后端解耦。
- 仿真器通过 `offroad_sim.backends.OffroadSimBackend` 接入。
- 后端创建通过 `default_backend_registry()` / `make_backend()` 进行，应用层无需直接绑定单个仿真器。
- 数据集格式通过 `DatasetAdapter` 隔离，`DatasetReplayBackend` 使用标准化后的 `DatasetFrame`。
- 本地训练/评测可以完全不依赖 BeamNG 或 UE5。
- BeamNG 和 UE5 都作为可选运行时接入，不影响包导入和测试。

## 3. 验收命令与结果

| 验收项 | 命令/动作 | 结果 |
| --- | --- | --- |
| Python 单元测试 | `python -m pytest -q` | 通过，`37 passed in 2.13s` |
| 前端构建 | `npm run build` | 通过，TypeScript + Vite build 成功 |
| 后端 registry | `python examples/check_backends.py` | 通过，`gym_heightmap`、`dataset_replay`、`ue5` 可用 |
| BeamNG 状态 | registry status | 接口可用，真实 runtime 未配置，提示缺 `beamngpy` 和 `BNG_HOME` |
| CLI catalog | `python -m offroad_sim.cli list --kind all` | 通过，列出 scenario、vehicle、agent、backend |
| CLI episode | `python -m offroad_sim.cli run --agent world_model --max-steps 30 --json` | 通过，返回 episode metrics |
| Gymnasium wrapper | `python examples/run_gymnasium_env.py --steps 20` | 通过，正常 reset/step/truncate |
| World-model agent | `python examples/run_world_model_agent.py --max-steps 40` | 通过，世界模型 agent 正常运行 |
| UE5 mock backend | `python examples/run_mock_ue5_backend.py` | 通过，TCP JSON bridge 与 mock server 正常交互 |
| Dataset mock 创建 | `python scripts/create_mock_dataset.py outputs/acceptance_phase1_dataset --frames 6` | 通过 |
| Dataset replay | `python examples/run_dataset_replay.py outputs/acceptance_phase1_dataset --sequence seq_0001 --load-assets` | 通过，6 帧播放完成 |
| Episode 录制 | `python examples/run_gym_demo.py --agent rule_based --max-steps 1200 --record outputs/episodes/phase1_acceptance` | 通过，128 步成功到达目标 |
| Episode 回放 | `python examples/replay_episode.py outputs/episodes/phase1_acceptance` | 通过，metadata/metrics/steps 可读取 |
| Dashboard API | `GET http://127.0.0.1:8000/health` | 通过，返回 `ok` |

## 4. 关键验收指标

`GymHeightmapBackend` + `rule_based` agent 的完整 episode 结果：

- `success`: `True`
- `done`: `True`
- `episode_length`: `128`
- `elapsed_time_sec`: `12.800`
- `path_length`: `95.472`
- `average_speed`: `7.459`
- `max_speed`: `9.097`
- `collision_count`: `0`
- `rollover`: `False`
- `average_terrain_risk`: `0.151`
- `distance_to_goal`: `4.528`

这说明第一阶段本地仿真闭环已经成立：场景加载、agent 决策、后端 step、指标累计、episode 保存和 replay 都能端到端跑通。

## 5. 当前代码模块状态

### Core / Config

- `offroad_sim.core` 定义统一数据结构：`Action`、`VehicleState`、`Observation`、`StepResult`、`EpisodeInfo`。
- `offroad_sim.scenarios` 与 `offroad_sim.vehicles` 支持 YAML 配置加载。

### Agents

- 已有 `RandomAgent`、`StopAgent`、`RuleBasedGoalAgent`、`KeyboardAgent` placeholder。
- 新增 `WorldModelAgent`，使用世界模型预测短时风险并调节 throttle/brake。

### Backends

- `GymHeightmapBackend`：第一阶段主力本地后端，可生成 2.5D terrain/risk/occupancy/traversability。
- `DatasetReplayBackend`：通过 adapter 层支持动态数据集接入。
- `BeamNGBackend`：接口和运行状态检查已就绪，真实 runtime 待 `beamngpy` 与 `BNG_HOME` 配置。
- `UE5Backend`：TCP JSON bridge 和 `MockUE5Server` 已可测。

### World Models / RL

- `BaseWorldModel` 和 `WorldModelPrediction` 已定义。
- `SimpleKinematicWorldModel` 可保存/加载并进行 kinematic rollout。
- `OffroadGymEnv` 提供 Gymnasium 接口，支持 RL 框架后续接入。

### Dashboard / CLI

- `offroad-sim` CLI 支持 `list`、`run`、`replay`。
- FastAPI dashboard 后端支持 catalog、backend status、episode run、episode/metrics 查询。
- React/Vite dashboard 前端可构建，具备实验控制、指标展示、轨迹预览和历史 episode 列表。

## 6. 已知限制

- BeamNG 真实运行尚未验收：当前机器缺 `beamngpy` Python 包和 `BNG_HOME` 环境变量。第一阶段只验收 import-safe、runtime status 和接口边界。
- UE5 当前使用 mock server 验收，真实 Unreal runtime 尚未接入。
- Dashboard 前端完成 build 和 HTTP/API 验证，浏览器自动化在当前工具连接中曾超时，因此本阶段以前端 build、API health 和服务可达作为验收依据。
- `outputs/`、`BeamNG/`、`node_modules/`、`dist/` 均不纳入 GitHub 提交。
- `package-lock.json` 未纳入本阶段提交；当前前端以 `package.json` 描述依赖，后续需要严格锁版本时可单独提交 lockfile。

## 7. 结论

第一阶段通过验收。

当前仓库已经具备一个稳定的本地 benchmark 闭环：配置 -> 后端 -> agent -> step -> metrics -> record -> replay -> CLI/dashboard。多后端组织方式已经成型，数据集动态接入方式已经成型，世界模型和 RL 的入口也已经放好。

下一阶段建议优先推进 BeamNG 真实 runtime 接入：

1. 安装并验证 `beamngpy`。
2. 设置 `BNG_HOME` 指向本地 BeamNG.tech 目录。
3. 完成 `VehicleConfig.sensors` 到 BeamNG sensors 的映射。
4. 跑通 BeamNG 场景加载、车辆 spawn、动作 step、传感器读取和 metrics 回传。
5. 把 BeamNG smoke test 纳入验收矩阵。

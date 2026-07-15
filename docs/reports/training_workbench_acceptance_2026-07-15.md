# 数据集与训练工作台最终验收

日期：2026-07-15

## 验收结论

P0-P6 的独立训练工作台闭环已实现：

```text
导入数据集 -> 检查/预览/划分 -> 选择 trainer manifest -> 配置参数
-> 异步训练与实时监控 -> 管理 artifact -> 推理预览
-> 对比实验 -> 导出报告
```

该流程不依赖 BeamNG，也不要求为新模型修改平台 Python 代码。

## 能力清单

- 数据集详情、质量扫描、确定性 train/validation/test 划分和 RGB/Depth/LiDAR 同步预览；划分文件会被训练、验证和推理实际消费并写入运行记录。
- 版本化 trainer manifest，支持 Python script/module/executable、Conda、环境变量、动态参数表单和输入兼容性检查。
- FIFO 训练队列、重新运行、进程树取消、stdout/stderr、events.jsonl、ETA、CPU/RAM 和可用时的 GPU/VRAM 监控。
- 多指标真实 step 对齐、train/validation loss 叠加、缩放、悬停、异常诊断及 JSON/CSV/PNG 导出。
- Artifact 自动发现、latest/best/favorite 标签、epoch/参数/指标摘要、引用保护删除和通用 inference 入口。
- 实验筛选、参数与指标排名、叠加曲线、复制复训、最佳标记、安全清理以及 Markdown/HTML/PNG 报告。

## 真实 ORFD 验收

- 数据集：`datasets/ORFD_Dataset_ICRA2022_ZIP`
- 数据集序列数：30；抽查序列 `training/c2021_0228_1819` 包含 449 帧
- 模态：RGB、dense depth、label、LiDAR
- 质量扫描：`ready`，0 个损坏资产，0 个质量问题
- 模型：`Tiny RGB 深度基线`
- 数据划分：按完整序列划分，train/validation/test 分别为 7872/2639/1687 帧
- 训练配置：从 train 和 validation 各读取 8 帧、每帧 256 像素、8 epochs

两组真实训练结果：

| Ridge | Validation RMSE |
|---:|---:|
| 0.0001 | 3.499036 m |
| 0.02 | 3.499613 m |

平台按 `validation_loss` 最小化方向选择 `ridge_1e4`。该 checkpoint 在 test split 的 4 个真实 ORFD 帧上的推理结果为：

- Depth RMSE：5.114240 m
- Depth MAE：3.301379 m
- 验收摘要：`outputs/training_acceptance/orfd_tiny_depth_split_aware/split_aware_acceptance.json`
- 预览：`outputs/training_acceptance/orfd_tiny_depth_split_aware/inference/depth_comparison.png`
- 对比报告：`outputs/training_acceptance/orfd_tiny_depth_split_aware/report/experiment_report.md`

可靠性复验在修复 split 泄漏、产物完整性和不可变快照后再次运行两组异步训练，指标与原验收一致。每个实验均生成 trainer/split 快照以及 trainer entrypoint 的 SHA-256，历史复跑会强制校验这些哈希；test split 推理仍为 4 帧、RMSE 5.114240 m、MAE 3.301379 m：

- 复验摘要：`outputs/training_acceptance/orfd_tiny_depth_reliability_20260715/reliability_acceptance.json`
- 推理预览：`outputs/training_acceptance/orfd_tiny_depth_reliability_20260715/inference/depth_comparison.png`
- 对比报告：`outputs/training_acceptance/orfd_tiny_depth_reliability_20260715/report/experiment_report.md`

Tiny RGB 深度模型用于验证训练和推理基础设施，不代表高精度深度估计效果。

## 自动验证

- `python -m pytest -q`：417 passed, 1 skipped
- `python examples/run_gym_demo.py --agent rule_based --max-steps 1200`：成功到达目标，0 碰撞
- `python -m offroad_sim.cli list`：成功
- PySide6 offscreen MainWindow：成功启动
- `git diff --check`：无补丁错误，仅 Windows LF/CRLF 提示

# Johnson Valley Alternate Start/Goal Support-Graph Validation - 2026-07-07

## Purpose

This run checks the same-map new start/goal requirement without changing Python code. The alternate Johnson Valley task uses a different start and goal within the same selected region. The workflow reuses the existing region training and evaluation services:

1. Convert the previous alternate self-supervised collection summary into a standard `region_training_collection.json`.
2. Train a fresh `mlp_dynamics` checkpoint from those BeamNG collection episodes.
3. Evaluate `world_model_direct` with `model_support_graph_subgoals`.
4. Compare it against the route-guided baseline in the same request.

The route-free evaluator receives no task route and no experience corridor.

## Artifacts

- Task: `configs/tasks/beamng_johnson_valley_nav_alt.yaml`
- Converted collection manifest: `outputs/beamng_training_acceptance/alt_support_graph_20260707_collect/region_training_collection.json`
- Model: `outputs/beamng_region_world_models/johnson_valley_alt_mlp_support_graph_20260707/model`
- Training summary: `outputs/beamng_region_world_models/johnson_valley_alt_mlp_support_graph_20260707/region_world_model_training_summary.json`
- Evaluation summary: `outputs/region_world_model_eval/johnson_valley_alt_mlp_support_graph_20260707_live/region_world_model_evaluation_summary.json`
- Trajectory plot: `outputs/region_world_model_eval/johnson_valley_alt_mlp_support_graph_20260707_live/region_world_model_trajectory.svg`
- GUI world-model config: `johnson_valley_alt_mlp_support_graph_20260707`

## Collection Quality

- Collection rollouts: 5
- Route coverage ratio: 1.0
- Goal zone coverage: 1.0
- Unique region cells: 4
- Collection minimum goal distance: 11.877480362664814 m
- Quality gate: passed

## Model Memory

- World model: `mlp_dynamics`
- Support routes: 5
- Support points: 811
- Support graph nodes: 732
- Support graph edges: 727
- Train RMSE: 0.01340104433141595
- Validation RMSE: 0.013653330250098227

## BeamNG Evaluation

Command:

```powershell
python scripts\run_region_world_model_evaluation.py configs\tasks\beamng_johnson_valley_nav_alt.yaml --world-model-type mlp_dynamics --world-model outputs\beamng_region_world_models\johnson_valley_alt_mlp_support_graph_20260707\model --output-dir outputs\region_world_model_eval\johnson_valley_alt_mlp_support_graph_20260707_live --eval-steps 900 --planner navigation_mpc --planner-horizon 6 --planner-samples 32 --planner-iterations 3 --evaluation-agent world_model_direct --evaluation-local-subgoal-distance-m 12 --evaluation-use-model-support-graph-subgoals --include-route-guided-baseline --beamng-gfx vk --step-delay-sec 0 --post-run-hold-sec 0
```

Route-free support-graph result:

- Goal success: true
- Final goal distance: 11.83696420862757 m
- Minimum goal distance: 11.83696420862757 m
- Goal radius: 12.0 m
- Collision count: 0
- Stuck recovery count: 0
- Reverse count: 0
- Distance traveled: 81.78044943932638 m
- Route waypoint count: 0
- Experience corridor: false
- `model_support_graph_subgoal_used`: true
- `model_support_subgoal_used`: false
- `model_support_field_subgoal_used`: false

Route-guided baseline:

- Goal success: true
- Final goal distance: 9.732468239444627 m
- Collision count: 0
- Stuck recovery count: 0
- Reverse count: 0
- Distance traveled: 86.75250883746224 m

## Interpretation

This validates the same-map new start/goal requirement for the current support-graph route-free mode: the task changed, a fresh checkpoint was trained from the alternate collection data, and the same evaluation code reached the goal without route waypoints, reverse recovery, collisions, or an experience corridor.

It is still a model-owned topology mode trained from route-aware collection. The next harder milestone remains strict direct control or a learned traversability/frontier policy that reduces dependence on route-aware rollout topology.

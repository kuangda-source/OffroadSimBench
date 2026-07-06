# Johnson Valley Route-Free Support-Graph Validation - 2026-07-07

## Purpose

This run validates a stricter route-free model-control mode for the Johnson Valley task. The evaluator receives only the task start, goal, and region. It does not inject the task expert route or an experience corridor. Local subgoals come from a sparse `support_graph` saved inside the trained `mlp_dynamics` checkpoint.

## Artifacts

- Task: `configs/tasks/beamng_johnson_valley_nav_test.yaml`
- Collection manifest: `outputs/beamng_training_acceptance/route_aware_strict_20260704_collect/region_training_collection.json`
- Model: `outputs/beamng_region_world_models/johnson_valley_mlp_support_graph_20260707/model`
- Training summary: `outputs/beamng_region_world_models/johnson_valley_mlp_support_graph_20260707/region_world_model_training_summary.json`
- Evaluation summary: `outputs/region_world_model_eval/johnson_valley_mlp_support_graph_20260707_live/region_world_model_evaluation_summary.json`
- Trajectory plot: `outputs/region_world_model_eval/johnson_valley_mlp_support_graph_20260707_live/region_world_model_trajectory.svg`
- GUI world-model config: `johnson_valley_mlp_support_graph_20260707`

## Model Memory

- World model: `mlp_dynamics`
- Support routes: 6
- Support points: 2038
- Support graph nodes: 954
- Support graph edges: 948
- Train RMSE: 0.020760254999373416
- Validation RMSE: 0.021026535191466925

## BeamNG Evaluation

Command:

```powershell
python scripts\run_region_world_model_evaluation.py configs\tasks\beamng_johnson_valley_nav_test.yaml --world-model-type mlp_dynamics --world-model outputs\beamng_region_world_models\johnson_valley_mlp_support_graph_20260707\model --output-dir outputs\region_world_model_eval\johnson_valley_mlp_support_graph_20260707_live --eval-steps 1200 --planner navigation_mpc --planner-horizon 6 --planner-samples 32 --planner-iterations 3 --evaluation-agent world_model_direct --evaluation-local-subgoal-distance-m 12 --evaluation-use-model-support-graph-subgoals --include-route-guided-baseline --beamng-gfx vk --step-delay-sec 0 --post-run-hold-sec 0
```

Route-free support-graph result:

- Goal success: true
- Final goal distance: 11.988179142481128 m
- Minimum goal distance: 11.988179142481128 m
- Goal radius: 12.0 m
- Collision count: 0
- Stuck recovery count: 0
- Reverse count: 0
- Distance traveled: 194.96474505190548 m
- Route waypoint count: 0
- Experience corridor: false
- `model_support_graph_subgoal_used`: true
- `model_support_subgoal_used`: false
- `model_support_field_subgoal_used`: false

Route-guided baseline:

- Goal success: true
- Final goal distance: 11.245425840767437 m
- Collision count: 0
- Stuck recovery count: 0
- Reverse count: 0
- Distance traveled: 183.4191504754298 m

## Interpretation

This is stronger than the previous support-route demo because the runtime local subgoal source is a model-owned sparse graph instead of directly following the stored support route sequence or the task expert route. It still uses route-aware collection data, so it is not a pure perception-only world model. The next step is to repeat this support-graph evaluation on a same-map alternate start/goal and then reduce dependence on route-aware collection by adding more diverse off-route exploration.

## Verification

- `python -m pytest -q`: 347 passed, 1 skipped
- `python examples/run_gym_demo.py --agent rule_based --max-steps 1200`: success true
- `python -m offroad_sim.cli list`: succeeded
- `python -m pytest tests/test_desktop_services.py -q`: 43 passed
- PySide6 offscreen smoke: printed `OffroadSimBench Desktop`

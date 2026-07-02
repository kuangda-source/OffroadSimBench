# Johnson Valley Route-Free P1 Probe - 2026-07-03

## Scope

Task: `configs/tasks/beamng_johnson_valley_nav_test.yaml`

Goal: quantify whether the current `tiny_learned + world_model_direct` stack can drive without an expert route, and separate map/vehicle feasibility from route-free planning quality.

## Baseline Evaluation

Output:

`outputs/region_world_model_eval/p1_existing_demo_ready_1200_20260703_071234/region_world_model_evaluation_summary.json`

Route-free direct baseline:

- Inputs: region, start, goal only.
- Expert route: removed.
- Experience corridor: disabled.
- Result: failed.
- Minimum goal distance: `109.825 m`.
- Final goal distance: `109.835 m`.
- Collisions: `0`.
- Distance traveled: `11.826 m`.
- Stuck recovery count: `1016`.
- Reverse count: `0`.

Route-guided baseline:

- Inputs: same region/start/goal plus expert route.
- Result: succeeded.
- Final goal distance: `11.752 m`.
- Collisions: `0`.
- Distance traveled: `182.846 m`.
- Stuck recovery count: `4`.
- Reverse count: `0`.

Conclusion: the map, vehicle, and expert route are drivable. The strict direct route-free controller is still not usable.

## Route-Aware Training Probe

Output:

`outputs/region_self_supervised/p1_route_aware_direct_20260703_072512/region_self_supervised_summary.json`

Collection and training:

- Collection strategy: route-aware multi-start.
- Rollouts: `4`.
- Steps per rollout: `220`.
- Route coverage ratio: `0.923`.
- Goal-zone coverage: `0.25`.
- Unique region cells: `6 / 8`.
- Validation RMSE: `0.0576`.
- Segment sample count: start `329`, middle `287`, goal `145`.

Strict direct route-free evaluation after training:

- Experience corridor: disabled.
- Result: failed.
- Minimum goal distance: `110.277 m`.
- Final goal distance: `111.252 m`.
- Collisions: `0`.
- Distance traveled: `41.302 m`.
- Stuck recovery count: `831`.
- Reverse count: `228`.

Conclusion: route-aware collection is now covering the route and goal region, but the current direct controller/planner still cannot convert the learned dynamics into useful navigation. The next bottleneck is P2: local subgoal selection, stuck recovery, reverse recovery limits, and planning costs.

## Follow-Up

- Keep direct route-free as the demo-ready gate.
- Treat experience-corridor success as an intermediate aid, not final acceptance.
- Default self-supervised evaluation should not enable reverse recovery unless explicitly requested.

# Region Self-Supervised BeamNG Probe

Date: 2026-06-27

## Command

```powershell
.\.conda\offroad-sim-bench\python.exe scripts\run_region_self_supervised_world_model.py configs\tasks\beamng_johnson_valley_nav_001.yaml --output-dir outputs\region_self_supervised\beamng_johnson_valley_nav_001_probe --collect-steps 80 --collect-rollouts 1 --eval-steps 80 --planner navigation_mpc --planner-horizon 4 --planner-samples 12 --planner-iterations 1 --evaluation-agent world_model_direct --evaluation-route-mode route_free --beamng-gfx vk --close-beamng --step-delay-sec 0 --post-run-hold-sec 0
```

## Result

- BeamNG runtime and connection worked.
- Collection, tiny learned model training, route-free evaluation, `training_run.json`, and summary writing all completed.
- The learned model did not reach the goal in the short probe.
- The run recorded `diagnostics.status = navigation_model_insufficient`.

Key evidence from the probe:

- `connected`: true
- `evaluation_agent`: `world_model_direct`
- `route_free`: true
- `model_controlled`: true
- `collision_count`: 0
- `goal_success`: false
- `min_goal_distance`: 110.97186830999544 m
- `goal_radius`: 12.0 m
- `collection_progress_ratio`: 0.08260335741249593
- `train_rmse`: 0.3336112568410361

## Interpretation

The current short self-supervised data is enough to prove the BeamNG loop and local dynamics training path, but it is not enough to prove route-free navigation. The immediate next technical step is to collect wider region coverage and add a traversability or cost-map learner before expecting the tiny dynamics model to solve start-to-goal navigation without a route.

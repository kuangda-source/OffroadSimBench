# Region Coverage Curriculum Probe

Date: 2026-06-28

This probe validated the new BeamNG region self-supervised collection coverage
curriculum on `configs/tasks/beamng_johnson_valley_nav_001.yaml`.

Command summary:

```powershell
.\.conda\offroad-sim-bench\python.exe scripts\run_region_self_supervised_world_model.py configs\tasks\beamng_johnson_valley_nav_001.yaml --output-dir outputs\region_self_supervised\coverage_switch_probe --collect-steps 40 --collect-rollouts 1 --collection-goal-bias-interval 0 --collection-goal-corridor-interval 0 --collection-coverage-grid-size 4 --collection-coverage-target-interval 1 --collection-max-target-steps 10 --eval-steps 10 --planner navigation_mpc --planner-horizon 3 --planner-samples 8 --planner-iterations 1 --evaluation-agent world_model_direct --evaluation-route-mode route_free --beamng-gfx vk --close-beamng --step-delay-sec 0 --post-run-hold-sec 0
```

Observed collection diagnostics:

- `connected`: true
- `target_source`: `coverage`
- `target_count`: 4
- `target_steps`: 10
- `coverage_target_count`: 8
- `target_in_region`: true
- `collision_count`: 0

Observed evaluation diagnostics:

- `model_controlled`: true
- `goal_success`: false
- `goal_reached`: false
- `final_goal_distance`: 115.361 m
- `min_goal_distance`: 115.319 m
- `collision_count`: 0
- diagnostic status: `navigation_model_insufficient`

Conclusion:

The coverage curriculum is active in real BeamNG collection and can switch
between multiple region targets within one rollout. This improves data
collection breadth, but it is not yet a complete start-to-goal autonomous
driving solution. The next bottleneck is still route-free navigation policy
quality: the tiny learned dynamics model remains local and does not yet provide
enough traversability or long-horizon search structure to reach the goal.

# Region Self-Supervised World Model Attempt Report

## What Was Implemented

- Added `region_explorer` for route-free BeamNG collection inside a navigation polygon.
- Added `world_model_direct` for direct start-to-goal world-model MPC without reading `obs.info["route"]`.
- Added `run_region_self_supervised_world_model()` and `scripts/run_region_self_supervised_world_model.py`.
- Added GUI access from the BeamNG workbench: "区域自监督训练 world model".
- Kept the model/planner boundary switchable through the existing registries, including future `le_wm_cem` or other planner adapters.

## Validation Result

Eight real BeamNG iterations were attempted on `configs/tasks/beamng_johnson_valley_nav_test.yaml`.

- Strict route-free collection and evaluation now run end-to-end.
- The backend no longer injects fallback start/goal route metadata when `draw_route=false` and `evaluation_route_mode=none`.
- The route-free direct controller is manual/model-controlled and does not use BeamNG `ai_line`.
- The current strict route-free run does not reach the goal. Best observed direct route-free evaluation remained around 105 m from the goal after 1200 steps.
- The same task is drivable: `model_mpc + simple_kinematic + task_route` reached the goal in 99 steps, with 0 collisions and final distance 11.546 m.
- `model_mpc + tiny_learned + task_route` did not reach the goal when the tiny model was trained only from the current self-supervised explorer data.

## Blocker

The blocker is not BeamNG connectivity or low-level actuator wiring. The blocker is the missing global traversability prior for the selected Johnson Valley region:

- The start orientation and direct start-to-goal line do not match the actual drivable path through the camp/terrain.
- A local world model trained from short, low-coverage exploration cannot infer an obstacle-free global path from only polygon bounds and a final goal.
- The current `tiny_learned` model is a low-dimensional dynamics model, not a visual traversability/world-state model. It can learn local motion deltas, but it does not learn where obstacles, roads, or terrain traps are.

## Next Executable Step

The next step should split the problem into a global planner and a local model controller:

1. Build a BeamNG traversability sampler for the selected region:
   - sample candidate ground points with raycast;
   - reject points with large slope, collision geometry, or non-ground hits;
   - connect nearby valid samples into a graph.
2. Plan start-to-goal over that graph with A*/Dijkstra.
3. Feed the generated path to `model_mpc` as a high-level route, while keeping the local controller model-switchable.
4. Continue using self-supervised data to train/improve the local world model, but do not require the local model to solve global navigation alone.
5. Only after graph-based traversability works should full LE-WM/Dreamer/TD-MPC-style visual models be used to replace or augment the traversability scorer.

This will keep the system honest: global path planning comes from map/traversability evidence, and the world model controls local vehicle behavior along that path.

# LE-WM Planning Integration

This project keeps LE-WM as an optional external runtime while exposing it
through the same agent/model/planner registries used by local models.

## Architecture

```text
WorldModelAgent
  -> WorldModelRegistry: le_wm
  -> PlannerRegistry: le_wm_cem
  -> stable_worldmodel.policy.AutoCostModel(checkpoint)
  -> stable_worldmodel.solver.CEMSolver
  -> Action sequence
```

For local smoke tests without a heavy LE-WM checkpoint, use:

```powershell
python scripts\export_lewm_hdf5.py outputs\mock_orfd_phase3 outputs\stablewm\mock_orfd_phase3.h5 --adapter orfd --image-size 32
python scripts\train_lewm_cost_model.py outputs\stablewm\mock_orfd_phase3.h5 --output outputs\models\lewm_cost_smoke
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type le_wm --world-model outputs\models\lewm_cost_smoke --planner le_wm_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
```

For a real LE-WM checkpoint:

```powershell
$env:LE_WM_HOME = "D:\programs\le-wm"
$env:STABLEWM_HOME = "D:\programs\OffroadSimBench\outputs\stablewm"

python -m offroad_sim.cli run --backend beamng --scenario configs\scenarios\beamng_orfd_eval.yaml --agent world_model --world-model-type le_wm --world-model D:\models\lewm\orfd --planner le_wm_cem --planner-horizon 10 --planner-samples 256 --planner-iterations 8 --max-steps 20 --record
```

To hand ORFD-style sequences to stable-worldmodel, export a flat HDF5 file:

```powershell
python scripts\export_lewm_hdf5.py D:\datasets\ORFD D:\stablewm-data\orfd.h5 --adapter orfd --image-size 64
```

Official ORFD ICRA 2022 ZIP releases can be used without expanding the whole
dataset. Select a sequence explicitly for smoke tests to avoid exporting every
ZIP file at once:

```powershell
python examples\inspect_dataset.py datasets\ORFD_Dataset_ICRA2022_ZIP --adapter orfd
python scripts\export_lewm_hdf5.py datasets\ORFD_Dataset_ICRA2022_ZIP outputs\stablewm\orfd_real_sample.h5 --adapter orfd --sequence-id training/c2021_0228_1819 --image-size 32
```

The export writes top-level `ep_len`, `ep_offset`, `state`, `action`,
`timestamp`, `goal`, and optional `pixels` arrays, matching stable-worldmodel's
`HDF5Dataset` boundary.

Recorded BeamNG episodes can be exported with:

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_orfd_eval_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_lewm_smoke.h5
```

For BeamNG expert/route recordings, derive training actions from observed
state deltas so the HDF5 reflects the motion that actually happened in the
simulator:

```powershell
python scripts\export_episodes_hdf5.py outputs\episodes\beamng_visible_autodrive_route_world_model_YYYYMMDDTHHMMSSZ outputs\stablewm\beamng_map_lewm.h5 --actions-from-state
```

The BeamNG map smoke loop can run the complete local cycle:

```powershell
python scripts\run_beamng_lewm_closed_loop.py --collect-steps 160 --eval-steps 120 --output-dir outputs\beamng_map_lewm\demo
```

That command records a BeamNG `gridmap_v2` route episode, exports it to
stable-worldmodel HDF5 with state-derived actions, trains an LE-WM-compatible
cost checkpoint, and evaluates the checkpoint through `route_world_model` with
`le_wm_cem`.

The same local LE-WM path is also exposed as a pluggable algorithm adapter:

```powershell
python -m offroad_sim.cli algorithms list
python -m offroad_sim.cli algorithms inspect local_lewm_cost --json
python -m offroad_sim.cli algorithms prepare-data local_lewm_cost --episode-root outputs\episodes\beamng_visible_autodrive_route_world_model_YYYYMMDDTHHMMSSZ --output-hdf5 outputs\stablewm\beamng_map_lewm.h5 --actions-from-state --json
python -m offroad_sim.cli algorithms train local_lewm_cost --input-hdf5 outputs\stablewm\beamng_map_lewm.h5 --output-dir outputs\models\beamng_map_lewm --json
```

External algorithm packages can follow the same `algorithm.yaml + adapter.py`
contract and declare whether they provide a full agent, world model, cost model,
or trajectory model.

Existing upstream/stable-worldmodel checkpoints can be used without retraining
the local smoke cost model:

```powershell
python -m offroad_sim.cli algorithms inspect stablewm_lewm --json
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_johnson_valley_nav_test.yaml --algorithm stablewm_lewm --algorithm-model-path D:\models\lewm\orfd\lewm_object.ckpt --eval-steps 520 --keep-beamng-open
```

`stablewm_lewm` accepts either a run directory containing `*_object.ckpt`, a
direct `*_object.ckpt` file, or a stable-worldmodel run name relative to
`STABLEWM_HOME`.

HuggingFace mirrors from the upstream LE-WM release contain `weights.pt` and
`config.json`. Convert those first:

```powershell
$env:LE_WM_HOME = "D:\programs\le-wm"
python scripts\convert_lewm_hf_checkpoint.py D:\models\lewm_hf\pusht D:\models\lewm\pusht
```

Region navigation tasks are the preferred path for start/goal experiments:

```powershell
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_johnson_valley_nav_test.yaml --algorithm local_lewm_cost --collect-steps 240 --eval-steps 520 --output-dir outputs\region_navigation\johnson_valley_nav_test_train
```

The task is defined as `navigation_region_v1`: region polygon, start pose, goal,
success radius, constraints, and an expert route used only during collection.
The evaluation scenario keeps only start and goal, then reports final/minimum
goal distance, first reached step, region membership, and success status.
For model-control experiments, set `beamng.evaluation_drive_mode: manual`; this
keeps BeamNG from taking over with `ai_line` so the selected agent/model/planner
actually produces vehicle commands. The GUI task editor saves manual evaluation
mode by default.

The selected region is now passed into BeamNG observations as
`observation.info["navigation_region"]`. `world_model_cem` adds an explicit
trajectory penalty for leaving the polygon or hugging the boundary, and the
local LE-WM-compatible cost checkpoint receives `region_polygon` so
`le_wm_cem` can penalize out-of-region rollouts as well.

## Required External Runtime

`le_wm_cem` requires:

- `stable-worldmodel`
- `torch`
- `gymnasium`
- a checkpoint or run directory containing a stable-worldmodel object with
  `get_cost(info_dict, action_candidates)`

Planning-only runtime can be installed with:

```powershell
python -m pip install -e .[lewm]
```

Training/evaluation workflows that follow the upstream LE-WM repository usually
need the heavier stable-worldmodel extras:

```powershell
python -m pip install -e .[lewm-train]
```

`LeWMCEMPlanner` builds an info dictionary with:

- `pixels`: current observation image, shape `(1, 1, H, W, 3)`
- `goal`: goal image, shape `(1, 1, H, W, 3)`
- `goal_state`: numeric target, shape `(1, 1, 2)`
- `state`: compact vehicle state, shape `(1, 1, 4)`

Checkpoint-specific preprocessing should be added inside
`offroad_sim/planning/stablewm.py` if the trained LE-WM run expects different
keys, history length, normalizers, or image size.

The planner lazy-loads and caches `AutoCostModel + CEMSolver`, so each episode
does not reload the checkpoint on every control step.

## ORFD Note

ORFD is primarily a freespace perception dataset. If a sequence does not include
vehicle actions or ego poses, the adapter marks poses as
`synthetic_index_order`. A production LE-WM path planner should train on data
that includes action-conditioned rollouts, such as BeamNG-recorded episodes or
ORFD synchronized with odometry/control logs.

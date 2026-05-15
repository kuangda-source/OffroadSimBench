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
python -m offroad_sim.cli run --backend dataset_replay --dataset-root outputs\mock_orfd_phase3 --adapter orfd --agent world_model --world-model-type tiny_learned --world-model outputs\models\phase3_tiny_world_model --planner world_model_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record
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

The export writes top-level `ep_len`, `ep_offset`, `state`, `action`,
`timestamp`, `goal`, and optional `pixels` arrays, matching stable-worldmodel's
`HDF5Dataset` boundary.

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

# Pluggable Model Adapter Layer Design

Date: 2026-05-17

## Goal

OffroadSimBench needs a stable way for different people to run their own
algorithms on maps designed by the project owner or uploaded by other users.
The adapter layer must support full agents, world models, cost models, and
trajectory models without forcing changes in core simulator, dataset, or GUI
code for each new algorithm.

The first local validation target is the current LE-WM-compatible cost-model
path:

```text
BeamNG episode -> StableWM HDF5 -> LE-WM cost checkpoint -> le_wm_cem -> BeamNG evaluation
```

## Design Choice

Use a capability-based algorithm adapter layer.

Each algorithm package declares what it can do in a manifest and implements a
small Python adapter class. The platform chooses the correct execution route
from declared capabilities:

- complete agent: `act(observation) -> Action`;
- world model: `predict(observation, action_sequence) -> rollout`;
- cost model: `score_actions(observation, action_candidates) -> costs`;
- trajectory model: `plan_trajectory(observation, task) -> trajectory`.

This preserves the existing core abstractions:

- `OffroadAgent` remains the runtime control entrypoint;
- `OffroadSimBackend` remains the simulator boundary;
- existing registries remain the way GUI and CLI discover runtime choices;
- simulator-specific logic stays under `offroad_sim/backends/`.

## Algorithm Package Layout

Algorithm packages live under a configurable search path. The default local
path is `algorithms/`.

```text
algorithms/
  local_lewm_cost/
    algorithm.yaml
    adapter.py
    README.md
    checkpoints/
```

Example manifest:

```yaml
algorithm_id: local_lewm_cost
display_name: Local LE-WM Cost Model
entrypoint: adapter:LocalLeWMCostAlgorithm
version: 0.1.0

capabilities:
  train: true
  infer: true
  act: false
  predict: false
  score_actions: true
  plan_trajectory: false

input_contract:
  observations: [state, goal, rgb_optional]
  actions: [steer, throttle, brake]
  task: navigation_region_v1

output_contract:
  mode: action_cost

runtime:
  device: cpu
  optional_dependencies: [torch, stable_worldmodel]
```

The manifest is intentionally metadata-only. Runtime code remains inside
`adapter.py`.

## Core Interfaces

Add a new package:

```text
offroad_sim/algorithms/
  base.py
  manifest.py
  registry.py
  builtins/
    local_lewm_cost.py
```

`AlgorithmAdapter` is the common optional-capability base class:

```python
class AlgorithmAdapter:
    algorithm_id: str

    def prepare_data(self, request: DataPrepRequest) -> DataPrepResult: ...
    def train(self, request: TrainRequest) -> TrainResult: ...
    def load(self, model_path: str | Path) -> None: ...
    def act(self, request: ActRequest) -> Action: ...
    def predict(self, request: PredictRequest) -> PredictionResult: ...
    def score_actions(self, request: ScoreActionsRequest) -> ScoreActionsResult: ...
    def plan_trajectory(self, request: TrajectoryPlanRequest) -> TrajectoryPlanResult: ...
```

Methods are optional. Calling an unsupported method raises
`UnsupportedCapabilityError` with a clear message that includes the algorithm
id and available capabilities.

## Task And Map Contract

Introduce a navigation task contract that is independent of BeamNG:

```yaml
task_type: navigation_region_v1
map_id: gridmap_v2_region_001
backend_targets: [beamng, gym_heightmap]
region:
  polygon:
    - [0.0, -160.0]
    - [30.0, -160.0]
    - [30.0, -260.0]
    - [0.0, -260.0]
start_pose:
  pos: [1.3, -167.0, 100.6]
  yaw: -1.57
goal:
  pos: [4.0, -240.0]
  radius: 6.0
expert_route:
  - [1.3, -167.0]
  - [1.3, -205.0]
  - [0.0, -232.0]
  - [4.0, -240.0]
constraints:
  max_steps: 300
  max_collision_count: 0
```

BeamNG-specific level names, spawn height, prefab settings, and native AI line
details remain in scenario metadata. The task contract only describes what the
algorithm must solve.

## Dataset And Episode Contract

Training data is exported through a unified episode schema:

```text
episode_id
map_id
task_id
timestep
observation.state
observation.goal
observation.assets
action
reward
done
info
```

Adapters may export this into model-specific formats. The built-in LE-WM cost
adapter exports StableWM HDF5 and can derive actions from BeamNG state deltas
for expert route demonstrations.

## Runtime Data Flow

Training flow:

```text
Task config
  -> backend collection policy or uploaded dataset
  -> unified episode records
  -> AlgorithmAdapter.prepare_data()
  -> AlgorithmAdapter.train()
  -> model artifact registry entry
```

Inference flow:

```text
Task config + selected map
  -> OffroadSimBackend.reset()
  -> AlgorithmRuntimeAgent
  -> selected AlgorithmAdapter capability
  -> action or trajectory
  -> backend.step()
  -> metrics + episode recording
```

Capability routing:

- if `act` is supported, call the adapter directly;
- else if `plan_trajectory` is supported, track the returned trajectory with a
  shared low-level controller;
- else if `score_actions` is supported, use CEM/MPC over the score function;
- else if `predict` is supported, use existing world-model planner;
- otherwise the algorithm is not runnable for closed-loop control.

## GUI And CLI Behavior

The GUI should show algorithms as first-class choices beside agent, world
model, and planner controls. A user can:

- import or select a map/task;
- select an algorithm package;
- inspect declared capabilities;
- run data preparation;
- train if supported;
- run inference/evaluation on BeamNG or another backend;
- open generated artifacts and episode summaries.

The CLI gets matching commands:

```powershell
python -m offroad_sim.cli algorithms list
python -m offroad_sim.cli algorithms inspect local_lewm_cost
python -m offroad_sim.cli algorithms train --algorithm local_lewm_cost --task configs\tasks\beamng_region_nav_001.yaml
python -m offroad_sim.cli algorithms run --algorithm local_lewm_cost --task configs\tasks\beamng_region_nav_001.yaml --backend beamng
```

## Error Handling

The adapter layer reports errors at the boundary where they occur:

- manifest parse errors include file path and field name;
- missing dependencies report install hints but do not break base imports;
- unsupported capabilities fail before launching a simulator;
- model artifact loading errors include algorithm id and path;
- backend runtime failures remain backend errors, not algorithm errors;
- every failed train/run writes a JSON failure summary under `outputs/`.

## Testing Strategy

Initial implementation tests should cover:

- manifest parsing and validation;
- registry discovery from built-in and local algorithm folders;
- unsupported capability errors;
- local LE-WM adapter data preparation using recorded BeamNG episodes;
- local LE-WM adapter training through the existing cost-model script;
- runtime routing from cost model to CEM planner;
- GUI smoke test showing the algorithm choice;
- full local validation without BeamNG;
- optional BeamNG smoke using the existing visible route and LE-WM checkpoint.

The acceptance target for the first implementation is:

```text
local_lewm_cost algorithm package
  -> discovers in registry
  -> prepares StableWM HDF5 from a BeamNG episode
  -> trains a checkpoint
  -> runs BeamNG evaluation through score_actions + le_wm_cem
  -> records metrics and summary JSON
```

## Non-Goals For First Implementation

- sandboxing untrusted third-party code;
- remote algorithm execution;
- packaging uploaded maps into full BeamNG levels;
- full upstream LE-WM visual latent training;
- automatic dependency installation from arbitrary uploads.

These can be added later after the local adapter protocol is stable.

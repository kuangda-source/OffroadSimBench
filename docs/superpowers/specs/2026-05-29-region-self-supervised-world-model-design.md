# Region Self-Supervised World Model Design

## Goal

Build the first route-free world-model control loop for BeamNG region tasks. The user should be able to mark a region, collect self-supervised interaction data inside it, train a switchable world model, and run start-to-goal navigation without passing an expert route to the evaluation agent.

## Model Choice

The first implementation uses the existing `tiny_learned` low-dimensional dynamics model because it is local, fast, testable, and already implements `BaseWorldModel.predict()`. I also checked the current open-source candidates:

- [TD-MPC2](https://github.com/nicklashansen/tdmpc2) is attractive for continuous control and has many released checkpoints, but those checkpoints target DMControl, Meta-World, ManiSkill2, and MyoSuite rather than BeamNG/off-road vehicle observations.
- [DreamerV3](https://github.com/danijar/dreamerv3) is a strong general world-model RL family, but integrating it here would mean a new policy-learning runtime and task wrapper before we can validate the BeamNG control loop.
- [LE-WM / stable-worldmodel](https://github.com/lucas-maes/le-wm) is already partially wired in this repository and matches the pixel-to-planning direction best. It also publishes pretrained checkpoints for its original environments, but the full visual model still needs a BeamNG-specific observation/action dataset and checkpoint glue before it can be the default controller.

So the MVP keeps `tiny_learned` as the validated self-supervised trainer and keeps TD-MPC2, DreamerV3, and full LE-WM as adapter targets. They remain compatible as long as they implement the same prediction boundary:

```text
Observation + action sequence -> predicted future states, risk, uncertainty metadata
```

This keeps the system model-agnostic while avoiding a large dependency jump before the control loop is proven.

## Workflow

1. Load a `navigation_region_v1` task.
2. Create a route-free BeamNG collection scenario from the same region/start/goal.
3. Run a `region_explorer` agent that samples temporary goals inside the polygon and records transitions.
4. Convert the recorded episode trace to `DatasetSequence`.
5. Train and save `TinyLearnedWorldModel`.
6. Evaluate with `world_model_direct`, which receives no expert route and plans directly toward the final goal with `NavigationMPCPlanner`.
7. Report acceptance using the existing goal, region, collision, and model-control metrics.

## Acceptance

- Evaluation scenario metadata must not include `beamng.route`.
- Evaluation agent must be `world_model_direct`.
- The saved model config should be usable from the existing world-model registry.
- Tests must prove the self-supervised service trains from recorded transitions and passes the trained model to route-free evaluation.

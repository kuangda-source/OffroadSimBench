# Johnson Valley Tiny vs MLP Route-Free Comparison

Date: 2026-07-04

## Purpose

Compare `tiny_learned` and `mlp_dynamics` on the same BeamNG Johnson Valley
collection to separate model-capacity effects from route-free control and data
coverage issues.

## Input

- Collection manifest:
  `outputs/beamng_training_acceptance/user_train_20260702_145625/collect/region_training_collection.json`
- Task:
  `configs/tasks/beamng_johnson_valley_nav_test.yaml`
- Evaluation:
  420 steps, `world_model_direct` route-free baseline plus `route_world_model`
  route-guided baseline, no experience corridor, no reverse recovery.
- Comparison output:
  `outputs/region_world_model_compare/johnson_valley_tiny_vs_mlp_420_20260704/region_world_model_comparison_summary.json`

## Result

| Model | Route-free min goal distance | Route-free final distance | Route-free stuck recoveries | Route-guided min distance | Route-guided final distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tiny_learned` | 112.856 m | 112.875 m | 265 | 59.652 m | 59.652 m |
| `mlp_dynamics` | 108.539 m | 108.571 m | 255 | 68.299 m | 68.299 m |

`mlp_dynamics` is slightly better on strict route-free minimum distance, but
both models remain near the original 113 m failure band and far from the first
gate of `<50 m`.

## Training Data Diagnosis

Both models were trained from 3,594 transitions, but all samples were in the
start segment:

```json
{
  "segment_sample_count": {
    "start": 3594,
    "middle": 0,
    "goal": 0
  }
}
```

The collection manifest was accepted by an older, weaker gate even though its
minimum goal distance stayed above 107 m. With the stricter gate now used by the
GUI and service options, the same collection fails because it does not make
enough progress and does not cover the route/goal zones.

## Conclusion

The immediate bottleneck is P1 data collection, not the difference between
`tiny_learned` and `mlp_dynamics`. The next useful BeamNG run should collect a
new route-aware, multi-start dataset with nonzero middle and goal segment
coverage before spending more iterations on P2 route-free control costs.


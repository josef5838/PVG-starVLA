# Cosmos Policy BEHAVIOR adapter checklist

This adapter belongs in the pinned Cosmos Policy checkout, not in StarVLA's
runtime environment.  StarVLA defines the contract it must satisfy.

## Required mapping

| BEHAVIOR value | Cosmos adapter input/output |
| --- | --- |
| head RGB | primary image |
| left/right wrist RGB | wrist-view representation; preserve both or document a strict two-camera protocol |
| `robot_r1::proprio` | 61-D proprioception, normalized only from training data statistics |
| `task_id` | natural-language task from `meta/tasks.jsonl` |
| predicted action chunk | unnormalized `(T, 23)` R1Pro action array |

## Conformance gate

Before a result is reportable, the adapter must pass one `turning_on_radio`
rollout with the following assertions:

1. It accepts exactly the benchmark camera ordering.
2. It rejects state vectors other than 61 dimensions.
3. It returns finite `(T, 23)` actions in the R1Pro action convention.
4. It resets all action/history state on a websocket `reset` request.
5. It logs inference latency without sending simulator-only information to the
   policy.

Record the Cosmos commit, model identifier, data manifest hash, image
preprocessing, action horizon, sampling settings, and adapter commit in
`provenance.json`.

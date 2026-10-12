# adjust_bottle: independent Omega camera-stream conditioning

Evaluation status: **completed**. Both training runs completed 10,000 updates and 80,000 sampled observations.

Direct keeps all 408 camera/register tokens; fused uses a two-layer transformer and 32 learned query tokens. Both use three independent eight-image Omega calls, a frozen backbone, learned camera/time/type embeddings and the same action decoder. No temporal RoPE.

## Training outcomes

| Conditioning | Last 1,000 updates: mean denoising loss | Best validation action MSE | Best update | Final validation MSE |
|---|---:|---:|---:|---:|
| direct | 0.00962 | 0.01650 | 9750 | 0.01659 |
| fused | 0.01092 | 0.03297 | 9000 | 0.03335 |

Both train losses converge. Direct reaches approximately half the validation error of fused. Best checkpoints are selected by the original 128-example held-out DDIM20 validation, not by rollout outcomes. Direct ran about 17h57m; fused about 17h46m. Peak allocated GPU memory was 11.91 and 12.05 GiB respectively.

![Training curves](training_curves.png)

## Recorded trajectory action prediction

Teacher forcing with recorded states and chronological per-camera histories: all 45 training episodes, eight equally spaced anchors each (360); all five held-out episodes, 16 anchors each (80). Predictions are matched by sample/batch order and diffusion noise. The held-out episodes were used for checkpoint selection. They are not an independent test set.

| Conditioning/checkpoint | Split | 16-action MSE | Valid-future MSE | First eight MSE | First action MSE | Active-arm joint RMSE (rad) | Gripper accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| direct/best | train | 0.01381 | 0.01455 | 0.01394 | 0.00387 | 0.1454 | 98.91% |
| direct/best | heldout_validation | 0.02389 | 0.02478 | 0.01773 | 0.00680 | 0.1790 | 97.30% |
| direct/final | train | 0.01391 | 0.01467 | 0.01379 | 0.00429 | 0.1470 | 98.85% |
| direct/final | heldout_validation | 0.02437 | 0.02530 | 0.01784 | 0.00690 | 0.1818 | 97.30% |
| fused/best | train | 0.02224 | 0.02275 | 0.01891 | 0.00590 | 0.1809 | 98.02% |
| fused/best | heldout_validation | 0.03651 | 0.03826 | 0.03043 | 0.01087 | 0.2244 | 96.76% |
| fused/final | train | 0.02419 | 0.02457 | 0.01820 | 0.00441 | 0.1887 | 98.28% |
| fused/final | heldout_validation | 0.03216 | 0.03326 | 0.02625 | 0.00796 | 0.2137 | 96.72% |

MSE uses training-normalized actions; lower is better. Valid-future metrics exclude episode-end target padding. Joint RMSE is the mean of per-anchor RMSEs after inverse normalization; gripper accuracy covers both grippers. A hold-current-state reference has valid-future MSE 0.10140 on train and 0.11761 on held-out examples. Noise seed 4000 + batch index differs from the training-time validation seed, and anchor selection differs, so these values need not equal the saved validation MSE.

![Paired trajectory errors and action horizon](offline_comparison.png)

Paired train: direct has lower valid-future MSE on 41/45 trajectories; fused minus direct mean 0.00820, trajectory-bootstrap 95% interval [0.00632, 0.01032].
Paired heldout_validation: direct has lower valid-future MSE on 4/5 trajectories; fused minus direct mean 0.01348, trajectory-bootstrap 95% interval [0.00524, 0.02172].

## Closed-loop task completion

Use each validation-selected best checkpoint. Execute eight of each 16-action prediction, replan with DDIM20, and update image history at every control observation. Resize simulator RGB to dataset 640x480 before unchanged official Omega preprocessing. Both models receive identical initial scenes and noise-seed schedules. Each case runs to success or the official 400-step limit.

| Conditioning | Seen training scenes | Held-out validation scenes | Fresh evaluation seeds |
|---|---:|---:|---:|
| direct | 4/5 | 2/5 | 2/5 |
| fused | 3/5 | 2/5 | 2/5 |

Direct predicts recorded actions more accurately and solves one additional training scene in this small rollout set. Both policies complete two of five held-out scenes and two of five fresh scenes; the 32-token fuser provides no observed success advantage here. Generalization remains weak. These results demonstrate that camera/register-only conditioning can drive the policy, but comparisons with a trained state-only policy and RGB/patch-token baselines are needed to establish the benefit of the geometric backbone.

On the 7 scenes that both policies solve, direct uses 36.6 control steps on average versus 75.3 for fused. This comparison is conditional on success; failures run to 400 steps.

Primary endpoint is the unchanged simulator criterion: bottle functional point above 0.9m and on the target side. Secondary signals are two-gripper-body contact with a closed gripper and any actor-origin lift >=4cm; they do not replace full success.

![Closed-loop bottle height](rollout_heights.png)

| Conditioning / split | Grasp contact | Lift >=4cm | Fresh-seed arm breakdown |
|---|---:|---:|---|
| direct/train | 4/5 | 4/5 | right-arm demonstrations |
| direct/heldout_validation | 2/5 | 2/5 | right-arm demonstrations |
| direct/unseen_eval | 2/5 | 3/5 | left: 0/2 success, right: 2/3 success |
| fused/train | 3/5 | 3/5 | right-arm demonstrations |
| fused/heldout_validation | 2/5 | 2/5 | right-arm demonstrations |
| fused/unseen_eval | 2/5 | 2/5 | left: 0/2 success, right: 2/3 success |

## Failure inspection

direct: 7 failed cases; 7 have no recorded closed two-finger contact; 4 finish with the bottle actor below 0.5m. Failed cases: train episode 22 / seed 80, heldout_validation episode 25 / seed 92, heldout_validation episode 39 / seed 18, heldout_validation episode 49 / seed 45, unseen_eval episode None / seed 200001, unseen_eval episode None / seed 300000, unseen_eval episode None / seed 500000.
fused: 8 failed cases; 8 have no recorded closed two-finger contact; 5 finish with the bottle actor below 0.5m. Failed cases: train episode 22 / seed 80, heldout_validation episode 25 / seed 92, train episode 35 / seed 31, heldout_validation episode 39 / seed 18, heldout_validation episode 49 / seed 45, unseen_eval episode None / seed 200001, unseen_eval episode None / seed 300000, unseen_eval episode None / seed 500000.

A representative direct failure is training episode 22 / source episode 24 / seed 80. Its trace records palm contact at control steps 13–15, first gripper closure at step 21 (recorded expert closure at command index 26), no closed two-finger contact and at most 2mm lift. The head video shows the bottle being knocked off the table. These are observed failure signals; they do not isolate the backbone, readout or controller as the cause.

![Direct training-scene failure](direct_train_failure.png)


## Expert execution control

Recorded raw expert actions pass through the same ModelClient chunk scheduling, action-order conversion and simulator controller. This oracle bypasses learned inference and its normalization; it validates scene recovery and execution.

- train: 5/5 expert replays succeed.
- heldout_validation: 5/5 expert replays succeed.

A second oracle uses the exact training action transform followed by inverse normalization. Its joint commands differ from recorded values by at most 1.8e-7 rad, while gripper commands become binary and differ by up to 0.4724. This checks the action-target execution path as well as the raw recording.

- train: 5/5 normalization-roundtrip expert replays succeed.
- heldout_validation: 5/5 normalization-roundtrip expert replays succeed.

## Scope and artifacts

- All 50 recorded demonstrations use the right arm. The six left-arm action dimensions have min=max=0; min/max inverse normalization therefore fixes them at zero regardless of predicted normalized values. Fresh left-arm cases are outside the demonstrated distribution and have this additional action-space restriction; arm-specific counts are reported above.
- Fresh seed starts are advanced only when setup is unstable, without expert-feasibility rejection. Accepted seeds are 100001, 200001, 300000, 400000, 500000. Their results are an unseen-scene diagnostic, not a full official benchmark submission.
- Rollout histories are sampled per control observation. TOPP determines variable physical elapsed time, so their eight-frame window is not guaranteed to match nominal training 15Hz.
- Five scenes per subset and one training seed give a small comparison. These runs alone cannot establish that geometry beats RGB/state alternatives.
- Converted-to-source mappings were checked against every recorded state and next action. Seen/held-out rollouts also require matching initial RGB, state and bottle model.
- GPU rendering used isolated driver libraries matching kernel 595.71.05 and isolated Warp 1.9.1; system installations were unchanged.
- [Paired video gallery](videos.html), [machine-readable summary](summary.json), [registered manifest](manifest.json), [training plots PDF](training_curves.pdf).
- [Per-trajectory action metrics CSV](trajectory_action_metrics.csv), [all rollout outcomes CSV](rollout_results.csv), [video validation](video_validation.json), [validation sampling audit](validation_sampling_audit.json).
- Raw action predictions, per-anchor metrics and per-step rollout action/state/object traces are under the direct and fused subdirectories. Videos were fully decoded before completion.
- The evaluation protocol and scripts are archived under code/.

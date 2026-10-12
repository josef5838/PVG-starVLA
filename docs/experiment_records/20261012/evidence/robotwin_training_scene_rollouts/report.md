# Training-scene rollouts

Exact source scenes recovered for training episodes 0, 1, 2 (source episodes 51, 60, 38; seeds 131, 162, 91). Both models used their step-10000 checkpoints and literal instruction `rotate_qrcode`. Initial robot states matched exactly; image MAE was 2.32–2.55/255.

| Variant | Task successes | Interpretation |
|---|---:|---|
| Linear | 1/3 | Episode 0 passed after 7 actions with both grippers open throughout: object pushed into accepted orientation, no successful grasp. |
| Residual | 0/3 | All episodes reached the 400-action limit. |

These failures occur on recovered training scenes, so the issue is not explained solely by generalization to unseen scenes. Teacher-action replay success validates controller execution, not learned-policy success.

The task success predicate checks orientation, height, and open grippers; it does not require a grasp or lift.

Videos:

- [linear_training_episode_0_seed_131.mp4](linear_training_episode_0_seed_131.mp4) (8 frames; task success: True)
- [linear_training_episode_1_seed_162.mp4](linear_training_episode_1_seed_162.mp4) (400 frames; task success: False)
- [linear_training_episode_2_seed_91.mp4](linear_training_episode_2_seed_91.mp4) (400 frames; task success: False)
- [residual_training_episode_0_seed_131.mp4](residual_training_episode_0_seed_131.mp4) (400 frames; task success: False)
- [residual_training_episode_1_seed_162.mp4](residual_training_episode_1_seed_162.mp4) (400 frames; task success: False)
- [residual_training_episode_2_seed_91.mp4](residual_training_episode_2_seed_91.mp4) (400 frames; task success: False)

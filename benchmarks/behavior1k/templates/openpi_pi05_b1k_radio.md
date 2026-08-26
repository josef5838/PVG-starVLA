# OpenPI `pi05_b1k_radio` configuration recipe

Apply this change in the separately pinned OpenPI `behavior` checkout before
using the `pi05` benchmark launcher. Do not copy OpenPI source into StarVLA.

In `src/openpi/training/config.py`, copy the existing `pi05_b1k` `TrainConfig`
and change its dataset identity and path:

```python
TrainConfig(
    name="pi05_b1k_radio",
    model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
    data=LeRobotB1KDataConfig(
        repo_id="turning_on_radio",
        base_config=DataConfig(
            data_cls=_lerobot_compat.LeRobotDataset,
            dataset_root="/absolute/path/behavior-1k-2026",
            prompt_from_task=True,
            dataset_kwargs={"tolerance_s": 5e-4},
        ),
        robot_config_name="b1k/R1Pro",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    save_interval=500,
    num_train_steps=2_000,
    assets_base_dir="./outputs/assets",
    checkpoint_base_dir="./outputs/checkpoints",
)
```

For a 24--48 GB RTX card, replace the model section and add the fields below
for LoRA training:

```python
model=pi0_config.Pi0Config(
    action_horizon=32,
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
),
freeze_filter=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter(),
ema_decay=None,
```

Use a unique config name for every distinct data regime and preserve the
matching `outputs/assets/<config>/turning_on_radio/norm_stats.json` directory
with the checkpoint when moving to the RTX evaluation machine.

# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

import logging
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    EmbodimentTag,
)

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return batch


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        print(
            f"Warning: DataConfig for robot_type={robot_type!r} has no embodiment_tag, using"
            f" {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"

    # Opt-in factory hook: a DataConfig may define ``make_dataset(dataset_name=..., **ds_kwargs)``
    # to swap in a custom dataset class (e.g. with per-task filtering / chunk stride).
    # When absent, fall through to the default LeRobotSingleDataset construction below.
    if hasattr(data_config, "make_dataset"):
        return data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
        )

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r in mixture_spec]}")
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    split_manifest = {}
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_cfg = data_cfg
        fraction = float(data_cfg.get("validation_fraction", 0.0))
        if fraction:
            from starVLA.dataloader.episode_split import split_episodes

            split_seed = int(data_cfg.get("validation_seed", seed))
            test_fraction = float(data_cfg.get("test_fraction", 0.0))
            test_ids = []
            if test_fraction:
                from starVLA.dataloader.episode_split import split_train_validation_test

                train_ids, val_ids, test_ids = split_train_validation_test(
                    Path(data_root_dir) / d_name, fraction, test_fraction, split_seed
                )
                if mode not in {"train", "val", "test"}:
                    raise ValueError("Three-way split requires train, val, or test mode")
            else:
                train_ids, val_ids = split_episodes(Path(data_root_dir) / d_name, fraction, split_seed)
            dataset_cfg = OmegaConf.merge(
                data_cfg,
                {
                    "episode_ids": train_ids if mode == "train" else test_ids if mode == "test" else val_ids,
                    "statistics_episode_ids": train_ids,
                },
            )
            split_manifest[d_name] = {
                "train_episode_ids": train_ids,
                "validation_episode_ids": val_ids,
                "seed": split_seed,
            }
            if test_fraction:
                split_manifest[d_name]["test_episode_ids"] = test_ids
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=dataset_cfg
                ),
                d_weight,
            )
        )
    manifest_dir = data_cfg.get("statistics_cache_dir")
    if split_manifest and manifest_dir:
        import json

        path = Path(manifest_dir) / "episode_split.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(split_manifest, indent=2))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/simBenchmarks/LIBERO/train_files/bar/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override data_root_dir from config")
    args = parser.parse_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = Path(vla_dataset_cfg.data_root_dir)
    if args.data_mix is not None:
        vla_dataset_cfg.data_mix = args.data_mix
    if args.data_root_dir is not None:
        vla_dataset_cfg.data_root_dir = Path(args.data_root_dir)

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for _batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 3:
            break
        count += 1
        pass

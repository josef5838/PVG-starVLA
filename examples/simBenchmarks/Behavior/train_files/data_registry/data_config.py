"""BEHAVIOR dataset mappings discovered by StarVLA's registry loader."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class Behavior2026R1ProDataConfig:
    """Official 2026 R1Pro LeRobot v3 RGB/proprio/action contract."""

    video_keys = ["video.head", "video.left_wrist", "video.right_wrist"]
    state_keys = ["state.proprio"]
    action_keys = ["action.control"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={"state.proprio": "min_max"},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.control": "min_max"},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "behavior_r1pro_2026": Behavior2026R1ProDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "behavior_r1pro_2026": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    # Retain the legacy mixture name without claiming that its 2025 adapter is
    # compatible with the 61-dimensional 2026 proprioception layout.
    "BEHAVIOR_challenge": [("BEHAVIOR_challenge", 1.0, "R1Pro")],
    "behavior_1k_2026": [("behavior-1k-2026", 1.0, "behavior_r1pro_2026")],
}

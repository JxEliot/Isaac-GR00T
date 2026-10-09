"""GR00T modality config for the SEER dual-arm parts-box-packing dataset."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


SEER_DUAL_ARM_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_image", "left_wrist_image", "right_wrist_image"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_joint_position",
            "right_arm_joint_position",
            "waist_position",
            "lift_position",
            "left_gripper_position",
            "right_gripper_position",
            "left_ee_pose",
            "right_ee_pose",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=[
            "left_ee_pose_target",
            "right_ee_pose_target",
            "left_gripper_command",
            "right_gripper_command",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
                state_key="left_ee_pose",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
                state_key="right_ee_pose",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(SEER_DUAL_ARM_CONFIG, EmbodimentTag.NEW_EMBODIMENT)

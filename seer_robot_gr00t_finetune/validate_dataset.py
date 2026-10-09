#!/usr/bin/env python3
"""Validate the metadata and file layout required by this training recipe."""

import argparse
import json
from pathlib import Path


EXPECTED_STATE = {
    "left_arm_joint_position": (0, 7),
    "right_arm_joint_position": (7, 14),
    "waist_position": (14, 15),
    "lift_position": (15, 16),
    "left_gripper_position": (62, 63),
    "right_gripper_position": (63, 64),
    "left_ee_pose": (48, 55),
    "right_ee_pose": (55, 62),
}
EXPECTED_ACTION = {
    "left_ee_pose_target": (0, 7),
    "right_ee_pose_target": (7, 14),
    "left_gripper_command": (14, 15),
    "right_gripper_command": (15, 16),
}
EXPECTED_VIDEO = {
    "head_image": "observation.images.head",
    "left_wrist_image": "observation.images.left_wrist",
    "right_wrist_image": "observation.images.right_wrist",
}


def load_json(path: Path):
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def slices(section: dict) -> dict[str, tuple[int, int]]:
    return {key: (value["start"], value["end"]) for key, value in section.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()

    info = load_json(root / "meta/info.json")
    modality = load_json(root / "meta/modality.json")
    load_json(root / "meta/stats.json")
    relative_stats = load_json(root / "meta/relative_stats.json")

    errors = []
    features = info.get("features", {})
    if features.get("observation.state", {}).get("shape") != [64]:
        errors.append("raw observation.state must have shape [64]")
    if features.get("action", {}).get("shape") != [16]:
        errors.append("raw action must have shape [16]")
    if slices(modality.get("state", {})) != EXPECTED_STATE:
        errors.append("state slices do not match the SEER dual-arm layout")
    if slices(modality.get("action", {})) != EXPECTED_ACTION:
        errors.append("action slices do not match the SEER dual-arm layout")

    actual_video = {
        key: value.get("original_key") for key, value in modality.get("video", {}).items()
    }
    if actual_video != EXPECTED_VIDEO:
        errors.append("video modalities must be head, left wrist, and right wrist")

    annotation = modality.get("annotation", {}).get("human.action.task_description", {})
    if annotation.get("original_key") != "task_index":
        errors.append("task description must map to task_index")

    episodes = int(info.get("total_episodes", 0))
    parquet_count = 0
    for parquet_path in (root / "data").glob("chunk-*/episode_*.parquet"):
        try:
            import pyarrow.parquet as pq

            if pq.ParquetFile(parquet_path).metadata.num_rows > 0:
                parquet_count += 1
        except Exception:
            errors.append(f"cannot read parquet: {parquet_path}")
    if parquet_count != episodes:
        errors.append(f"expected {episodes} parquet files, found {parquet_count}")

    valid_ids = {
        int(json.loads(line)["episode_index"])
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    video_count = sum(
        1
        for path in (root / "videos").glob("chunk-*/*/episode_*.mp4")
        if int(path.stem.split("_")[-1]) in valid_ids
    )
    expected_videos = episodes * len(EXPECTED_VIDEO)
    if video_count != expected_videos:
        errors.append(f"expected {expected_videos} videos, found {video_count}")

    task_file = root / "meta/tasks.jsonl"
    task_count = sum(1 for line in task_file.open(encoding="utf-8") if line.strip())
    if task_count != int(info.get("total_tasks", 0)):
        errors.append(f"expected {info.get('total_tasks')} tasks, found {task_count}")

    relative_keys = set(relative_stats) - {"__fingerprints__"}
    expected_relative = {"left_ee_pose_target", "right_ee_pose_target"}
    if relative_keys != expected_relative:
        errors.append(f"relative_stats keys must be {sorted(expected_relative)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"dataset: {root}")
    print(f"robot_type: {info.get('robot_type')}")
    print(f"episodes: {episodes}")
    print(f"frames: {info.get('total_frames')}")
    print(f"tasks: {task_count}")
    print(f"videos: {video_count}")
    print("raw state/action: 64/16; GR00T loader state/action: 30/14")
    print("validation: OK")


if __name__ == "__main__":
    main()

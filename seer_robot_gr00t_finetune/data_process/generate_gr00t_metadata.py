#!/usr/bin/env python3
"""Generate GR00T metadata for the SEER v2.1 quaternion dataset.

The parquet data stays untouched.  Relative statistics are computed in memory
from xyz+quaternion (xyzw) poses using the same T_reference^-1*T_target
convention used by GR00T, then expressed as xyz+rotvec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def stats(values: np.ndarray) -> dict:
    return {
        "max": np.max(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def relative_pose(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return target relative to reference as xyz + rotvec."""
    ref_r = Rotation.from_quat(reference[3:7])
    tgt_r = Rotation.from_quat(target[3:7])
    rel_r = ref_r.inv() * tgt_r
    rel_xyz = ref_r.inv().apply(target[:3] - reference[:3])
    return np.concatenate([rel_xyz, rel_r.as_rotvec()]).astype(np.float32)


def iter_episode_tables(root: Path):
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        yield path, pq.read_table(path, columns=["observation.state", "action"])


def modality() -> dict:
    return {
        "state": {
            "left_arm_joint_position": {"start": 0, "end": 7, "original_key": "observation.state"},
            "right_arm_joint_position": {
                "start": 7,
                "end": 14,
                "original_key": "observation.state",
            },
            "waist_position": {"start": 14, "end": 15, "original_key": "observation.state"},
            "lift_position": {"start": 15, "end": 16, "original_key": "observation.state"},
            "left_gripper_position": {"start": 62, "end": 63, "original_key": "observation.state"},
            "right_gripper_position": {"start": 63, "end": 64, "original_key": "observation.state"},
            "left_ee_pose": {"start": 48, "end": 55, "original_key": "observation.state"},
            "right_ee_pose": {"start": 55, "end": 62, "original_key": "observation.state"},
        },
        "action": {
            "left_ee_pose_target": {"start": 0, "end": 7, "original_key": "action"},
            "right_ee_pose_target": {"start": 7, "end": 14, "original_key": "action"},
            "left_gripper_command": {"start": 14, "end": 15, "original_key": "action"},
            "right_gripper_command": {"start": 15, "end": 16, "original_key": "action"},
        },
        "video": {
            "head_image": {"original_key": "observation.images.head"},
            "left_wrist_image": {"original_key": "observation.images.left_wrist"},
            "right_wrist_image": {"original_key": "observation.images.right_wrist"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--horizon", type=int, default=32)
    args = ap.parse_args()
    root = args.dataset.resolve()
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "modality.json").write_text(json.dumps(modality(), indent=2) + "\n")

    all_left, all_right = [], []
    usable = 0
    for path, table in iter_episode_tables(root):
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        n = min(len(states), len(actions)) - args.horizon + 1
        if n <= 0:
            continue
        for i in range(n):
            ref = states[i]
            targets = actions[i : i + args.horizon]
            all_left.extend(relative_pose(ref[48:55], x[0:7]) for x in targets)
            all_right.extend(relative_pose(ref[55:62], x[7:14]) for x in targets)
            usable += 1

    if not all_left or not all_right:
        raise RuntimeError("no usable episodes for relative statistics")

    # Match GR00T's cache fingerprint payload for the configured action keys.
    fingerprints = {}
    for key, state_key in (
        ("left_ee_pose_target", "left_ee_pose"),
        ("right_ee_pose_target", "right_ee_pose"),
    ):
        payload = {
            "embodiment_tag": "new_embodiment",
            "action_key": key,
            "action_delta_indices": list(range(args.horizon)),
            "state_delta_indices": [0],
            "rep": "RELATIVE",
            "type": "EEF",
            "format": "XYZ_ROTVEC",
            "state_key": state_key,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fingerprints[key] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    result = {
        "left_ee_pose_target": stats(np.asarray(all_left, dtype=np.float32)),
        "right_ee_pose_target": stats(np.asarray(all_right, dtype=np.float32)),
        "__fingerprints__": fingerprints,
    }
    (meta / "relative_stats.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {meta / 'modality.json'}")
    print(
        f"wrote {meta / 'relative_stats.json'}; usable reference frames={usable}, horizon={args.horizon}"
    )


if __name__ == "__main__":
    main()

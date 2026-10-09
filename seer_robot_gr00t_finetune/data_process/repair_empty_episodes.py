#!/usr/bin/env python3
"""Repair a LeRobot dataset manifest by excluding empty parquet episodes.

No data/video files are deleted. Metadata is backed up before replacement.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()
    root = args.dataset.resolve()
    meta = root / "meta"
    info_path = meta / "info.json"
    episodes_path = meta / "episodes.jsonl"
    stats_path = meta / "episodes_stats.jsonl"
    info = json.loads(info_path.read_text())
    episodes = [json.loads(x) for x in episodes_path.read_text().splitlines() if x.strip()]
    empty, valid = [], []
    for ep in episodes:
        i = int(ep["episode_index"])
        p = root / info["data_path"].format(
            episode_chunk=i // int(info["chunks_size"]), episode_index=i
        )
        rows = pq.ParquetFile(p).metadata.num_rows if p.exists() else 0
        (empty if rows == 0 else valid).append((ep, rows))
    if not empty:
        print("No empty episodes found; nothing changed")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = meta / f"manifest_backup_{stamp}"
    backup.mkdir()
    for p in (info_path, episodes_path, stats_path):
        if p.exists():
            shutil.copy2(p, backup / p.name)
    episodes_path.write_text("".join(json.dumps(ep, ensure_ascii=False) + "\n" for ep, _ in valid))
    if stats_path.exists():
        keep = {int(ep["episode_index"]) for ep, _ in valid}
        rows = [json.loads(x) for x in stats_path.read_text().splitlines() if x.strip()]
        stats_path.write_text(
            "".join(
                json.dumps(x, ensure_ascii=False) + "\n"
                for x in rows
                if int(x["episode_index"]) in keep
            )
        )
    info["total_episodes"] = len(valid)
    info["total_frames"] = sum(rows for _, rows in valid)
    info["total_videos"] = len(valid) * sum(
        1 for x in info.get("features", {}).values() if x.get("dtype") == "video"
    )
    info["total_chunks"] = (len(valid) + int(info["chunks_size"]) - 1) // int(info["chunks_size"])
    info["splits"] = {"train": f"0:{len(valid)}"}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"excluded empty episodes: {[int(ep['episode_index']) for ep, _ in empty]}")
    print(f"valid episodes: {len(valid)}, frames: {info['total_frames']}")
    print(f"metadata backup: {backup}")


if __name__ == "__main__":
    main()

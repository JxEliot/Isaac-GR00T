#!/usr/bin/env python3
"""Validate converted LeRobot v2.1 data and preserve DAgger/RECAP fields."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pyarrow.parquet as pq


REQUIRED = {
    "next.reward",
    "next.success",
    "next.terminated",
    "next.truncated",
    "info.is_intervention",
    "info.reset_state_id",
}


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} DATASET", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"ERROR: codebase_version is {info.get('codebase_version')!r}")
    data_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not data_files:
        raise SystemExit("ERROR: no v2.1 episode parquet files found")
    fields = set(pq.read_schema(data_files[0]).names)
    missing = sorted(REQUIRED - fields)
    if missing:
        raise SystemExit(f"ERROR: missing transition fields: {missing}")
    episodes = int(info.get("total_episodes", 0))
    videos = sorted((root / "videos").glob("chunk-*/*/episode_*.mp4"))
    print(f"dataset: {root}")
    print(f"version: {info.get('codebase_version')}")
    print(f"episodes: {episodes}")
    print(f"data files: {len(data_files)}")
    print(f"video files: {len(videos)}")
    print("preserved fields: " + ", ".join(sorted(REQUIRED)))
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pad truncated episode videos by cloning the final frame.

Default mode is report-only.  With --apply, only videos that are shorter than
their parquet episode length are rewritten.  Originals are moved to a backup
directory before replacement.  Internal PTS gaps are reported but are not
silently repaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,width,height,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    return stream


def frames(stream: dict) -> int:
    return int(stream.get("nb_frames") or 0)


def fps(stream: dict) -> str:
    return stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "20/1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--apply", action="store_true", help="rewrite videos; otherwise report only")
    ap.add_argument("--backup-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    root = args.dataset.resolve()
    meta = root / "meta"
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").open() if line.strip()]
    info = json.loads((meta / "info.json").read_text())
    video_keys = [k for k in info["features"] if k.startswith("observation.images.")]
    pattern = info["video_path"]
    backup = (args.backup_dir or root / ".video_pad_backups").resolve()
    rows = []

    for ep in episodes:
        eid = int(ep["episode_index"])
        expected = int(ep["length"])
        for feature in video_keys:
            path = root / pattern.format(
                episode_chunk=eid // int(info["chunks_size"]),
                video_key=feature,
                episode_index=eid,
            )
            item = {"episode": eid, "video": feature, "expected": expected, "path": str(path)}
            if not path.exists():
                item.update(status="missing", actual=0)
                rows.append(item)
                continue
            stream = probe(path)
            actual = frames(stream)
            item.update(
                actual=actual,
                missing=max(0, expected - actual),
                status="ok" if actual == expected else ("short" if actual < expected else "long"),
            )
            if args.apply and actual < expected and actual > 0:
                # Clone the final decoded frame. Re-encode to a temp file, then
                # atomically replace the original after the backup succeeds.
                backup_path = backup / path.relative_to(root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    suffix=".mp4", dir=path.parent, delete=False
                ) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    n = expected - actual
                    vf = f"tpad=stop_mode=clone:stop={n}"
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-vf",
                        vf,
                        "-frames:v",
                        str(expected),
                        "-r",
                        fps(stream),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        stream.get("pix_fmt") or "yuv420p",
                        str(tmp_path),
                    ]
                    subprocess.run(cmd, check=True)
                    actual_new = frames(probe(tmp_path))
                    if actual_new != expected:
                        raise RuntimeError(
                            f"padded output has {actual_new} frames, expected {expected}"
                        )
                    shutil.move(str(path), str(backup_path))
                    shutil.move(str(tmp_path), str(path))
                    item.update(status="padded", backup=str(backup_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            rows.append(item)

    report = args.report or (root / "meta" / "video_padding_report.json")
    report.write_text(
        json.dumps({"dataset": str(root), "applied": args.apply, "videos": rows}, indent=2) + "\n"
    )
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"report": str(report), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

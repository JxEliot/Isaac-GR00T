#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import pyarrow.parquet as pq


def serial(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [serial(x) for x in v]
    if isinstance(v, dict):
        return {k: serial(x) for k, x in v.items()}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    info = json.loads((src / "meta/info.json").read_text())
    eps = []
    for f in sorted((src / "meta/episodes").glob("chunk-*/file-*.parquet")):
        eps += pq.read_table(f).to_pylist()
    eps.sort(key=lambda x: int(x["episode_index"]))
    dst.mkdir(parents=True)
    (dst / "meta").mkdir()
    (dst / "data").mkdir()
    (dst / "videos").mkdir()
    outinfo = dict(info)
    outinfo["codebase_version"] = "v2.1"
    outinfo["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    outinfo["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    outinfo.pop("data_files_size_in_mb", None)
    outinfo.pop("video_files_size_in_mb", None)
    outinfo["total_chunks"] = math.ceil(len(eps) / int(info.get("chunks_size", 1000)))
    outinfo["total_videos"] = len(eps) * sum(
        1 for x in info["features"].values() if x.get("dtype") == "video"
    )
    (dst / "meta/info.json").write_text(json.dumps(outinfo, indent=2) + "\n")
    for name in ["stats.json", "seer_collection.json"]:
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, dst / "meta" / name)
    tasks = pq.read_table(src / "meta/tasks.parquet").to_pylist()
    with (dst / "meta/tasks.jsonl").open("w") as f:
        for row in tasks:
            f.write(
                json.dumps(
                    {"task_index": int(row["task_index"]), "task": row["task"]}, ensure_ascii=False
                )
                + "\n"
            )
    with (
        (dst / "meta/episodes.jsonl").open("w") as ef,
        (dst / "meta/episodes_stats.jsonl").open("w") as sf,
    ):
        for rec in eps:
            legacy = {
                k: serial(v)
                for k, v in rec.items()
                if not k.startswith(("data/", "videos/", "stats/", "meta/"))
                and k not in ("dataset_from_index", "dataset_to_index")
            }
            legacy.setdefault(
                "length", int(rec["dataset_to_index"]) - int(rec["dataset_from_index"])
            )
            ef.write(json.dumps(legacy, ensure_ascii=False) + "\n")
            stats = {k[6:]: serial(v) for k, v in rec.items() if k.startswith("stats/")}
            sf.write(
                json.dumps(
                    {"episode_index": int(rec["episode_index"]), "stats": stats}, ensure_ascii=False
                )
                + "\n"
            )
    data_cache = {}
    source_offsets = {}
    for rec in eps:
        ei = int(rec["episode_index"])
        start = int(rec["dataset_from_index"])
        end = int(rec["dataset_to_index"])
        chunk = int(rec.get("data/chunk_index", 0))
        fi = int(rec.get("data/file_index", 0))
        key = (chunk, fi)
        if key not in data_cache:
            source_path = src / f"data/chunk-{chunk:03d}/file-{fi:03d}.parquet"
            data_cache[key] = pq.read_table(source_path)
            # episode indices use dataset-global offsets, while each source parquet
            # starts at its own local row zero.  Build that file's global base from the
            # episode manifest instead of passing the global index to Table.slice().
            source_offsets[key] = min(
                int(x["dataset_from_index"])
                for x in eps
                if int(x.get("data/chunk_index", 0)) == chunk
                and int(x.get("data/file_index", 0)) == fi
            )
        local_start = start - source_offsets[key]
        local_end = end - source_offsets[key]
        if local_start < 0 or local_end > data_cache[key].num_rows or local_end < local_start:
            raise RuntimeError(
                f"episode {ei}: invalid local slice [{local_start}:{local_end}] for {key} with {data_cache[key].num_rows} rows"
            )
        table = data_cache[key].slice(local_start, local_end - local_start)
        if table.num_rows != end - start:
            raise RuntimeError(f"episode {ei}: wrote {table.num_rows} rows, expected {end - start}")
        out = (
            dst
            / f"data/chunk-{ei // int(info.get('chunks_size', 1000)):03d}/episode_{ei:06d}.parquet"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        for vkey, ft in info["features"].items():
            if ft.get("dtype") != "video":
                continue
            base = f"videos/{vkey}"
            vc = int(rec[f"{base}/chunk_index"])
            vf = int(rec[f"{base}/file_index"])
            vs = float(rec[f"{base}/from_timestamp"])
            ve = float(rec[f"{base}/to_timestamp"])
            source = src / f"videos/{vkey}/chunk-{vc:03d}/file-{vf:03d}.mp4"
            target = (
                dst
                / f"videos/chunk-{ei // int(info.get('chunks_size', 1000)):03d}/{vkey}/episode_{ei:06d}.mp4"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{vs:.6f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{max(ve - vs, 1e-6):.6f}",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "1",
                    "-y",
                    str(target),
                ],
                check=True,
            )
    print(f"converted episodes={len(eps)} dst={dst}")


if __name__ == "__main__":
    main()

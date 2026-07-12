#!/usr/bin/env python3
"""Export an ImageNet-256 subset from HF parquet shards.

Input repo expected:
  benjamin-paine/imagenet-1k-256x256

The parquet images are already 256x256 JPEGs. This script writes the image
bytes into an ImageFolder-style layout and records a JSONL manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def load_label_to_wnid(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "label_to_wnid" in data:
        data = data["label_to_wnid"]
    return {int(k): str(v) for k, v in data.items()}


def write_manifest_line(handle, digest, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    handle.write(line + "\n")
    digest.update((line + "\n").encode("utf-8"))


def export(args: argparse.Namespace) -> None:
    raw_data = args.raw_dir / "data"
    out_root = args.output_dir
    extracted = out_root / "extracted"
    metadata = out_root / "metadata"
    derived_raw = out_root / "raw"
    if args.overwrite and extracted.exists():
        import shutil

        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    derived_raw.mkdir(parents=True, exist_ok=True)

    label_to_wnid = load_label_to_wnid(args.label_to_wnid)
    (metadata / "label_to_wnid.json").write_text(
        json.dumps({"label_to_wnid": {str(k): v for k, v in sorted(label_to_wnid.items())}}, indent=2) + "\n",
        encoding="utf-8",
    )
    (derived_raw / "SOURCE_HF_IMAGENET_1K_256X256.md").write_text(
        "# ImageNet-256 HF Raw Source\n\n"
        "Derived from server-local HF parquet raw files:\n\n"
        "```text\n"
        f"{args.raw_dir.as_posix()}\n"
        "```\n\n"
        "Source repo: `benjamin-paine/imagenet-1k-256x256`.\n",
        encoding="utf-8",
    )

    manifest_path = metadata / "image_manifest.jsonl"
    manifest_sha = hashlib.sha256()
    split_counts = Counter()
    class_counts = Counter()
    label_seen = defaultdict(int)
    total = 0
    start = time.time()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for shard in sorted(raw_data.glob("train-*.parquet")):
            table = pq.read_table(shard, columns=["image", "label"])
            rows = table.to_pylist()
            shard_written = 0
            for row_idx, row in enumerate(rows):
                label = int(row["label"])
                class_index = label_seen[label]
                label_seen[label] += 1
                if class_index % args.train_stride != args.train_offset:
                    continue
                wnid = label_to_wnid[label]
                img = row["image"]
                source_path = img.get("path") or f"{shard.stem}_{row_idx:06d}.jpg"
                stem = Path(source_path).stem or f"{shard.stem}_{row_idx:06d}"
                out_rel = Path("train") / wnid / f"{stem}.jpg"
                out_path = extracted / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(img["bytes"])
                rec = {
                    "split": "train",
                    "path": str(Path("extracted") / out_rel).replace("\\", "/"),
                    "label": label,
                    "wnid": wnid,
                    "source_shard": shard.name,
                    "source_row": row_idx,
                    "source_path": source_path,
                    "width": args.image_size,
                    "height": args.image_size,
                    "sampling": f"class_index_mod_{args.train_stride}_eq_{args.train_offset}",
                }
                write_manifest_line(manifest, manifest_sha, rec)
                split_counts["train"] += 1
                class_counts[f"train/{wnid}"] += 1
                shard_written += 1
                total += 1
                if total % args.progress_every == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    print(f"written={total} elapsed={elapsed/60:.1f}m rate={total/elapsed:.1f}/s", flush=True)
            print(f"train {shard.name}: wrote={shard_written}", flush=True)

        for shard in sorted(raw_data.glob("validation-*.parquet")):
            table = pq.read_table(shard, columns=["image", "label"])
            rows = table.to_pylist()
            shard_written = 0
            for row_idx, row in enumerate(rows):
                label = int(row["label"])
                wnid = label_to_wnid[label]
                img = row["image"]
                source_path = img.get("path") or f"{shard.stem}_{row_idx:06d}.jpg"
                stem = Path(source_path).stem or f"{shard.stem}_{row_idx:06d}"
                out_rel = Path("val") / wnid / f"{stem}.jpg"
                out_path = extracted / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(img["bytes"])
                rec = {
                    "split": "val",
                    "path": str(Path("extracted") / out_rel).replace("\\", "/"),
                    "label": label,
                    "wnid": wnid,
                    "source_shard": shard.name,
                    "source_row": row_idx,
                    "source_path": source_path,
                    "width": args.image_size,
                    "height": args.image_size,
                    "sampling": "full_validation",
                }
                write_manifest_line(manifest, manifest_sha, rec)
                split_counts["val"] += 1
                class_counts[f"val/{wnid}"] += 1
                shard_written += 1
                total += 1
                if total % args.progress_every == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    print(f"written={total} elapsed={elapsed/60:.1f}m rate={total/elapsed:.1f}/s", flush=True)
            print(f"val {shard.name}: wrote={shard_written}", flush=True)

    summary = {
        "dataset_alias": args.alias,
        "source_alias": "imagenet_1k_256x256_hf",
        "source_raw_dir": str(args.raw_dir),
        "image_size": args.image_size,
        "preprocessing": "HF parquet images copied as 256x256 JPEG bytes; no resize or recoding",
        "train_sampling": {
            "stride": args.train_stride,
            "offset": args.train_offset,
            "description": f"keep per-label image index % {args.train_stride} == {args.train_offset}",
        },
        "split_counts": dict(sorted(split_counts.items())),
        "class_split_entries": len(class_counts),
        "manifest_sha256": manifest_sha.hexdigest(),
        "elapsed_seconds": time.time() - start,
    }
    (metadata / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="imagenet_256_10pct")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--label-to-wnid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())


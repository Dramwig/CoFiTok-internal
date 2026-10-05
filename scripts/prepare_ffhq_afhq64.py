#!/usr/bin/env python3
"""Prepare the documented FFHQ-64 and AFHQv2-64 ImageFolder datasets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


AFHQ_LABELS = ("cat", "dog", "wild")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest_line(handle: Any, digest: Any, record: dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True)
    handle.write(line + "\n")
    digest.update((line + "\n").encode("utf-8"))


def _reset_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _verify_image(payload: bytes, expected_size: int) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
        if image.size != (expected_size, expected_size):
            raise ValueError(f"Expected {expected_size}x{expected_size}, found {image.size}")
        return image.copy()


def prepare_ffhq(root: Path, overwrite: bool) -> dict[str, Any]:
    archive = root / "raw" / "ffhq-64x64.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing FFHQ archive: {archive}")
    extracted = root / "extracted"
    _reset_output(extracted, overwrite)
    images = extracted / "images"
    images.mkdir()
    metadata = root / "metadata"
    _reset_output(metadata, overwrite)

    manifest_path = metadata / "image_manifest.jsonl"
    digest = hashlib.sha256()
    count = 0
    with zipfile.ZipFile(archive) as source, manifest_path.open("w", encoding="utf-8") as manifest:
        for member in sorted(source.infolist(), key=lambda item: item.filename):
            if member.is_dir() or Path(member.filename).suffix.lower() != ".png":
                continue
            filename = Path(member.filename).name
            target = images / filename
            if target.exists():
                raise ValueError(f"Duplicate FFHQ image name in archive: {filename}")
            payload = source.read(member)
            _verify_image(payload, expected_size=64)
            target.write_bytes(payload)
            _write_manifest_line(
                manifest,
                digest,
                {
                    "relative_path": str(target.relative_to(root)).replace("\\", "/"),
                    "class_name": "unlabeled",
                    "source_member": member.filename,
                    "width": 64,
                    "height": 64,
                    "mode": "RGB",
                },
            )
            count += 1

    if count != 70_000:
        raise ValueError(f"Expected 70,000 FFHQ images, found {count}")
    summary = {
        "dataset_alias": "ffhq_64",
        "split_counts": {"train": count},
        "image_size": 64,
        "preprocessing": "ZIP extraction only; source PNG bytes retained without crop, resize, or recoding",
        "raw_sha256": _sha256(archive),
        "manifest_sha256": digest.hexdigest(),
    }
    (metadata / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _image_bytes(image_value: Any) -> bytes:
    if isinstance(image_value, dict):
        payload = image_value.get("bytes")
        if isinstance(payload, bytes):
            return payload
    if isinstance(image_value, bytes):
        return image_value
    raise TypeError("AFHQ parquet image field does not contain image bytes")


def _label_name(value: Any) -> str:
    if isinstance(value, str):
        if value in AFHQ_LABELS:
            return value
    else:
        label = int(value)
        if 0 <= label < len(AFHQ_LABELS):
            return AFHQ_LABELS[label]
    raise ValueError(f"Unexpected AFHQ label: {value!r}")


def prepare_afhq(root: Path, overwrite: bool) -> dict[str, Any]:
    raw_data = root / "raw" / "data"
    shards = sorted(raw_data.glob("train-*-of-00013.parquet"))
    if len(shards) != 13:
        raise FileNotFoundError(f"Expected 13 AFHQ parquet shards under {raw_data}, found {len(shards)}")
    extracted = root / "extracted"
    _reset_output(extracted, overwrite)
    train = extracted / "train"
    for label in AFHQ_LABELS:
        (train / label).mkdir(parents=True)
    metadata = root / "metadata"
    _reset_output(metadata, overwrite)

    manifest_path = metadata / "image_manifest.jsonl"
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            row_index = 0
            for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
                for row in batch.to_pylist():
                    label = _label_name(row["label"])
                    with Image.open(io.BytesIO(_image_bytes(row["image"]))) as opened:
                        image = opened.convert("RGB").resize((64, 64), resample=resampling)
                    filename = f"{shard.stem}_{row_index:06d}.png"
                    target = train / label / filename
                    image.save(target, format="PNG")
                    _write_manifest_line(
                        manifest,
                        digest,
                        {
                            "relative_path": str(target.relative_to(root)).replace("\\", "/"),
                            "class_name": label,
                            "source_shard": shard.name,
                            "source_row": row_index,
                            "width": 64,
                            "height": 64,
                            "mode": "RGB",
                        },
                    )
                    counts[label] += 1
                    row_index += 1

    total = sum(counts.values())
    expected = {"cat": 5558, "dog": 5169, "wild": 5076}
    if total != 15_803 or dict(counts) != expected:
        raise ValueError(f"Unexpected AFHQ class counts: {dict(counts)}")
    raw_hashes = {str(path.relative_to(root / "raw")): _sha256(path) for path in shards}
    summary = {
        "dataset_alias": "afhqv2_64",
        "split_counts": {"train": total},
        "class_counts": dict(sorted(counts.items())),
        "image_size": 64,
        "preprocessing": "Parquet 512x512 images converted to RGB and resized to 64x64 with PIL LANCZOS",
        "raw_sha256": raw_hashes,
        "manifest_sha256": digest.hexdigest(),
    }
    (metadata / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ffhq_64", "afhqv2_64", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {}
    if args.dataset in {"ffhq_64", "all"}:
        results["ffhq_64"] = prepare_ffhq(args.data_root / "ffhq_64", args.overwrite)
    if args.dataset in {"afhqv2_64", "all"}:
        results["afhqv2_64"] = prepare_afhq(args.data_root / "afhqv2_64", args.overwrite)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

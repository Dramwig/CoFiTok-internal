#!/usr/bin/env python3
"""Prepare an ImageNet-256 derived subset from official ILSVRC2012 raw tar files.

This script is intentionally local-raw friendly: it reads the official ImageNet-1K
raw archives in place and writes a derived ImageFolder-style dataset plus metadata.
It does not modify the official raw files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def center_crop_resize_jpeg(data: bytes, out_path: Path, image_size: int, quality: int) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        orig_w, orig_h = im.size
        im = im.convert("RGB")
        side = min(im.size)
        left = (im.width - side) // 2
        top = (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="JPEG", quality=quality, optimize=True)
    return orig_w, orig_h


def load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"id_to_wnid", "wnid_to_words", "validation_wnids"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"mapping file missing keys: {sorted(missing)}")
    return data


def iter_train_members(train_tar: Path) -> Iterable[tarfile.TarInfo]:
    with tarfile.open(train_tar, "r") as outer:
        members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar")]
        for member in sorted(members, key=lambda m: m.name):
            yield member


def write_source_note(raw_dir: Path, source_raw_dir: Path, train_tar: Path, val_tar: Path, devkit_tar: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    note = f"""# ImageNet-256 10pct Raw Source

This derived dataset was generated from the official ImageNet-1K / ILSVRC2012 raw archives kept in the local raw hub.

Local source raw directory:

```text
{source_raw_dir.as_posix()}
```

Source files:

```text
{train_tar.as_posix()}
{val_tar.as_posix()}
{devkit_tar.as_posix()}
```

The full official raw archives are intentionally not duplicated under this derived alias.
"""
    (raw_dir / "SOURCE_LOCAL_IMAGENET1K.md").write_text(note, encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir
    out_root = args.output_dir
    extracted = out_root / "extracted"
    metadata = out_root / "metadata"
    derived_raw = out_root / "raw"
    train_tar = raw_dir / "ILSVRC2012_img_train.tar"
    val_tar = raw_dir / "ILSVRC2012_img_val.tar"
    devkit_tar = raw_dir / "ILSVRC2012_devkit_t12.tar.gz"
    for p in [train_tar, val_tar, devkit_tar, args.mapping_json]:
        if not p.exists():
            raise FileNotFoundError(p)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if extracted.exists():
        raise FileExistsError(f"{extracted} already exists; pass --overwrite to rebuild")
    extracted.mkdir(parents=True)
    metadata.mkdir(parents=True)
    derived_raw.mkdir(parents=True)

    mapping = load_mapping(args.mapping_json)
    validation_wnids = mapping["validation_wnids"]
    wnid_to_words = mapping["wnid_to_words"]

    write_source_note(derived_raw, raw_dir, train_tar, val_tar, devkit_tar)
    shutil.copy2(devkit_tar, derived_raw / devkit_tar.name)

    raw_checksums = {
        devkit_tar.name: sha256_file(devkit_tar),
    }
    with (derived_raw / "checksums.sha256").open("w", encoding="utf-8") as f:
        for name, digest in raw_checksums.items():
            f.write(f"{digest}  {name}\n")

    manifest_path = metadata / "image_manifest.jsonl"
    manifest_hash = hashlib.sha256()
    split_counts = Counter()
    class_counts = Counter()
    original_size_counts = Counter()
    start = time.time()
    total_written = 0

    def write_manifest(record: dict) -> None:
        nonlocal total_written
        line = json.dumps(record, sort_keys=True)
        manifest.write(line + "\n")
        manifest_hash.update((line + "\n").encode("utf-8"))
        split_counts[record["split"]] += 1
        class_counts[f'{record["split"]}/{record["wnid"]}'] += 1
        original_size_counts[f'{record["original_width"]}x{record["original_height"]}'] += 1
        total_written += 1
        if total_written % args.progress_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            rate = total_written / elapsed
            print(f"written={total_written} elapsed={elapsed/60:.1f}m rate={rate:.1f}/s", flush=True)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        with tarfile.open(train_tar, "r") as outer:
            train_members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar")]
            for class_number, class_member in enumerate(sorted(train_members, key=lambda m: m.name), start=1):
                if args.limit_train_classes and class_number > args.limit_train_classes:
                    break
                wnid = Path(class_member.name).stem
                class_fileobj = outer.extractfile(class_member)
                if class_fileobj is None:
                    continue
                class_index = 0
                selected = 0
                with tarfile.open(fileobj=class_fileobj, mode="r|*") as inner:
                    for img_member in inner:
                        if not img_member.isfile():
                            continue
                        if not img_member.name.lower().endswith((".jpeg", ".jpg")):
                            continue
                        keep = (class_index % args.train_stride) == args.train_offset
                        class_index += 1
                        if not keep:
                            continue
                        img_file = inner.extractfile(img_member)
                        if img_file is None:
                            continue
                        img_bytes = img_file.read()
                        stem = Path(img_member.name).stem
                        out_rel = Path("train") / wnid / f"{stem}.jpg"
                        out_path = extracted / out_rel
                        orig_w, orig_h = center_crop_resize_jpeg(img_bytes, out_path, args.image_size, args.jpeg_quality)
                        selected += 1
                        write_manifest(
                            {
                                "split": "train",
                                "path": str(Path("extracted") / out_rel).replace("\\", "/"),
                                "wnid": wnid,
                                "words": wnid_to_words.get(wnid, ""),
                                "source_archive": class_member.name,
                                "source_member": img_member.name,
                                "original_width": orig_w,
                                "original_height": orig_h,
                                "width": args.image_size,
                                "height": args.image_size,
                                "sampling": f"class_index_mod_{args.train_stride}_eq_{args.train_offset}",
                            }
                        )
                print(f"train {wnid}: selected={selected} seen={class_index}", flush=True)

        with tarfile.open(val_tar, "r") as val_outer:
            val_members = [m for m in val_outer.getmembers() if m.isfile() and m.name.endswith(".JPEG")]
            for val_written, img_member in enumerate(sorted(val_members, key=lambda m: m.name), start=1):
                if args.limit_val_images and val_written > args.limit_val_images:
                    break
                match = VAL_RE.search(Path(img_member.name).name)
                if not match:
                    raise ValueError(f"unexpected validation member name: {img_member.name}")
                val_index = int(match.group(1))
                wnid = validation_wnids[val_index - 1]
                img_file = val_outer.extractfile(img_member)
                if img_file is None:
                    continue
                img_bytes = img_file.read()
                stem = Path(img_member.name).stem
                out_rel = Path("val") / wnid / f"{stem}.jpg"
                out_path = extracted / out_rel
                orig_w, orig_h = center_crop_resize_jpeg(img_bytes, out_path, args.image_size, args.jpeg_quality)
                write_manifest(
                    {
                        "split": "val",
                        "path": str(Path("extracted") / out_rel).replace("\\", "/"),
                        "wnid": wnid,
                        "words": wnid_to_words.get(wnid, ""),
                        "source_archive": val_tar.name,
                        "source_member": img_member.name,
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "width": args.image_size,
                        "height": args.image_size,
                        "sampling": "full_validation",
                    }
                )

    elapsed = time.time() - start
    summary = {
        "dataset_alias": args.alias,
        "source_alias": "imagenet1k",
        "source_raw_dir": str(raw_dir).replace("\\", "/"),
        "image_size": args.image_size,
        "preprocessing": "EXIF transpose, RGB convert, center crop to square, LANCZOS resize, JPEG save",
        "jpeg_quality": args.jpeg_quality,
        "train_sampling": {
            "stride": args.train_stride,
            "offset": args.train_offset,
            "description": f"keep per-class image index % {args.train_stride} == {args.train_offset}",
        },
        "split_counts": dict(sorted(split_counts.items())),
        "class_split_entries": len(class_counts),
        "original_size_counts_top20": dict(original_size_counts.most_common(20)),
        "manifest_sha256": manifest_hash.hexdigest(),
        "elapsed_seconds": elapsed,
    }
    (metadata / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="imagenet_256_10pct")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--limit-train-classes", type=int, default=0)
    parser.add_argument("--limit-val-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_stride <= 0:
        raise ValueError("--train-stride must be positive")
    if not (0 <= args.train_offset < args.train_stride):
        raise ValueError("--train-offset must be in [0, train_stride)")
    return args


if __name__ == "__main__":
    prepare(parse_args())

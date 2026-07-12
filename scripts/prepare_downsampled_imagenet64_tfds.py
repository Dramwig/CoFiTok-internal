from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TFDS downsampled_imagenet/64x64 to a prepared image-folder layout."
    )
    parser.add_argument("--dataset-name", default="downsampled_imagenet/64x64")
    parser.add_argument(
        "--tfds-data-dir",
        default="/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/tfds",
        help="TFDS cache/download directory.",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted",
        help="Prepared output root containing train/validation split directories.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--unlabeled-class", default="unlabeled")
    parser.add_argument("--format", choices=["png", "jpeg"], default="png")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of images per nested shard directory under each class. Zero disables sharding.",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=0,
        help="Limit each split for a loader smoke export. Zero means export all examples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing exported image files and record their hashes.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Write TFDS builder metadata without downloading or exporting examples.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON path for --inspect-only metadata.",
    )
    return parser.parse_args()


def _import_tfds() -> Any:
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise SystemExit(
            "tensorflow_datasets is required only for this export script. "
            "Install it in the chosen server environment, for example: "
            "python -m pip install tensorflow-datasets"
        ) from error
    return tfds


def _to_uint8_image_array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], repeats=3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"Unsupported image array shape: {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, repeats=3, axis=2)
    return array


def _scalar_or_none(value: Any) -> int | str | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, (int, np.integer)):
            return int(item)
        return str(item)
    return str(array.tolist())


def _encode_image(array: np.ndarray, image_format: str) -> tuple[bytes, int, int]:
    image = Image.fromarray(array).convert("RGB")
    buffer = BytesIO()
    if image_format == "png":
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), image.width, image.height


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_total(info: Any, split: str, limit: int) -> int | None:
    try:
        total = int(info.splits[split].num_examples)
    except Exception:
        return limit if limit > 0 else None
    if limit > 0:
        return min(total, limit)
    return total


def _builder_metadata(tfds: Any, builder: Any) -> dict[str, Any]:
    info = builder.info
    splits = {
        name: int(split_info.num_examples)
        for name, split_info in info.splits.items()
    }
    return {
        "dataset_alias": "downsampled_imagenet_64",
        "source_loader": builder.name,
        "tfds_version": getattr(tfds, "__version__", "unknown"),
        "builder_version": str(info.version),
        "features": str(info.features),
        "supervised_keys": str(info.supervised_keys),
        "splits": splits,
        "download_size": str(info.download_size),
        "dataset_size": str(info.dataset_size),
    }


def _class_name(example: dict[str, Any], label_key: str, unlabeled_class: str) -> tuple[str, int | str | None]:
    label = _scalar_or_none(example.get(label_key))
    if label is None:
        return unlabeled_class, None
    if isinstance(label, int):
        return f"{label:04d}", label
    clean = str(label).replace("/", "_").replace("\\", "_")
    return clean, label


def _export_split(
    *,
    tfds: Any,
    builder: Any,
    info: Any,
    split: str,
    output_root: Path,
    image_key: str,
    label_key: str,
    unlabeled_class: str,
    image_format: str,
    shard_size: int,
    limit: int,
    skip_existing: bool,
) -> dict[str, Any]:
    dataset = builder.as_dataset(split=split, shuffle_files=False)
    if limit > 0:
        dataset = dataset.take(limit)

    extension = "jpg" if image_format == "jpeg" else image_format
    manifest_path = output_root / f"{split}_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "split": split,
        "examples": 0,
        "classes": {},
        "manifest_path": str(manifest_path),
    }

    total = _split_total(info, split, limit)
    iterator = tfds.as_numpy(dataset)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, example in enumerate(tqdm(iterator, total=total, desc=f"export {split}")):
            if image_key not in example:
                raise KeyError(f"Example does not contain image key {image_key!r}: {sorted(example)}")
            class_name, label = _class_name(example, label_key, unlabeled_class)
            class_dir = output_root / split / class_name
            if shard_size > 0:
                class_dir = class_dir / f"shard_{index // shard_size:05d}"
            image_path = class_dir / f"{index:08d}.{extension}"
            image_path.parent.mkdir(parents=True, exist_ok=True)

            if image_path.exists() and skip_existing:
                data = image_path.read_bytes()
                with Image.open(image_path) as image:
                    width, height = image.size
            else:
                data, width, height = _encode_image(
                    _to_uint8_image_array(example[image_key]),
                    image_format=image_format,
                )
                image_path.write_bytes(data)

            sha256 = _sha256_bytes(data)
            summary["examples"] += 1
            summary["classes"][class_name] = summary["classes"].get(class_name, 0) + 1
            record = {
                "split": split,
                "index": index,
                "relative_path": str(image_path.relative_to(output_root.parent)).replace("\\", "/"),
                "class_name": class_name,
                "label": label,
                "width": width,
                "height": height,
                "sha256": sha256,
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")

    return summary


def main() -> None:
    args = parse_args()
    tfds = _import_tfds()
    output_root = Path(args.output_root)
    tfds_data_dir = Path(args.tfds_data_dir)

    builder = tfds.builder(args.dataset_name, data_dir=str(tfds_data_dir))
    if args.inspect_only:
        metadata = _builder_metadata(tfds, builder)
        body = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(body, encoding="utf-8")
            print(f"wrote {summary_path}")
        else:
            print(body)
        return

    builder.download_and_prepare()
    info = builder.info

    split_summaries = [
        _export_split(
            tfds=tfds,
            builder=builder,
            info=info,
            split=split,
            output_root=output_root,
            image_key=args.image_key,
            label_key=args.label_key,
            unlabeled_class=args.unlabeled_class,
            image_format=args.format,
            shard_size=args.shard_size,
            limit=args.max_examples_per_split,
            skip_existing=args.skip_existing,
        )
        for split in args.splits
    ]
    summary = {
        "dataset_alias": "downsampled_imagenet_64",
        "source_loader": args.dataset_name,
        "tfds_data_dir": str(tfds_data_dir),
        "output_root": str(output_root),
        "image_key": args.image_key,
        "label_key": args.label_key,
        "unlabeled_class": args.unlabeled_class,
        "image_format": args.format,
        "shard_size": args.shard_size,
        "max_examples_per_split": args.max_examples_per_split,
        "splits": split_summaries,
        "note": "TFDS downsampled_imagenet is treated as unlabeled for CoFiTok unconditional validation.",
    }
    summary_path = output_root / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

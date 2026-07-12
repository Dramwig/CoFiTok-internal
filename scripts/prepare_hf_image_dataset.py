from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Hugging Face image dataset to prepared folders.")
    parser.add_argument("--dataset-id", default="benjamin-paine/imagenet-1k-64x64")
    parser.add_argument("--config", default="default")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/extracted",
    )
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--unlabeled-class", default="unlabeled")
    parser.add_argument("--format", choices=["png", "jpeg"], default="png")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--max-examples-per-split", type=int, default=0)
    parser.add_argument("--no-streaming", action="store_true", help="Load regular datasets instead of streaming.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _import_datasets() -> Any:
    try:
        import datasets
    except ImportError as error:
        raise SystemExit(
            "datasets is required for this export script. "
            "Install it in the selected environment with python -m pip install datasets."
        ) from error
    return datasets


def _to_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], repeats=3, axis=2)
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, repeats=3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Unsupported image value shape: {array.shape}")
    return Image.fromarray(array).convert("RGB")


def _encode_image(image: Image.Image, image_format: str) -> tuple[bytes, int, int]:
    image = image.convert("RGB")
    buffer = BytesIO()
    if image_format == "png":
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), image.width, image.height


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _class_name(example: dict[str, Any], label_key: str, unlabeled_class: str) -> tuple[str, int | str | None]:
    label = example.get(label_key)
    if label is None:
        return unlabeled_class, None
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, int):
        return f"{label:04d}", int(label)
    clean = str(label).replace("/", "_").replace("\\", "_")
    return clean, str(label)


def _limited_rows(rows: Iterable[dict[str, Any]], limit: int) -> Iterable[dict[str, Any]]:
    if limit > 0:
        return islice(rows, limit)
    return rows


def _export_split(
    *,
    rows: Iterable[dict[str, Any]],
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
    extension = "jpg" if image_format == "jpeg" else image_format
    manifest_path = output_root / f"{split}_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "split": split,
        "examples": 0,
        "classes": {},
        "manifest_path": str(manifest_path),
    }

    total = limit if limit > 0 else None
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, example in enumerate(tqdm(_limited_rows(rows, limit), total=total, desc=f"export {split}")):
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
                data, width, height = _encode_image(_to_pil_image(example[image_key]), image_format)
                image_path.write_bytes(data)

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
                "sha256": _sha256_bytes(data),
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    datasets = _import_datasets()
    output_root = Path(args.output_root)
    split_summaries = []

    for split in args.splits:
        rows = datasets.load_dataset(
            args.dataset_id,
            name=args.config if args.config else None,
            split=split,
            streaming=not args.no_streaming,
        )
        split_summaries.append(
            _export_split(
                rows=rows,
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
        )

    summary = {
        "dataset_alias": output_root.parent.name,
        "source_loader": args.dataset_id,
        "config": args.config,
        "output_root": str(output_root),
        "image_key": args.image_key,
        "label_key": args.label_key,
        "image_format": args.format,
        "shard_size": args.shard_size,
        "streaming": not args.no_streaming,
        "max_examples_per_split": args.max_examples_per_split,
        "splits": split_summaries,
    }
    summary_path = output_root / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

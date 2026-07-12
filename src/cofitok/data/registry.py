from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from cofitok.configs import DataConfig

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_PREPARED_IMAGE_DATASETS = {
    "downsampled_imagenet_64",
    "imagenet_1k_64x64_hf",
    "ffhq_64",
    "afhqv2_64",
    "imagenet_256",
    "imagenet_256_10pct",
}
_UNSPLIT_PREPARED_DATASETS = {"ffhq_64", "afhqv2_64"}


class RandomImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, length: int, channels: int, image_size: int, seed: int = 13) -> None:
        self.length = length
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(length, channels, image_size, image_size, generator=generator)
        self.labels = torch.zeros(length, dtype=torch.long)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


class TinyImageNetDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, split: str, transform: transforms.Compose) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.wnids = self._read_lines(root / "wnids.txt")
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.samples = self._build_samples(split)

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _build_samples(self, split: str) -> list[tuple[Path, int]]:
        if split == "train":
            samples: list[tuple[Path, int]] = []
            for wnid in self.wnids:
                image_dir = self.root / "train" / wnid / "images"
                samples.extend((path, self.class_to_idx[wnid]) for path in sorted(image_dir.glob("*.JPEG")))
            return samples

        if split in {"val", "validation"}:
            annotation_path = self.root / "val" / "val_annotations.txt"
            image_dir = self.root / "val" / "images"
            samples = []
            for line in self._read_lines(annotation_path):
                fields = line.split("\t")
                filename, wnid = fields[0], fields[1]
                samples.append((image_dir / filename, self.class_to_idx[wnid]))
            return samples

        if split == "test":
            image_dir = self.root / "test" / "images"
            return [(path, -1) for path in sorted(image_dir.glob("*.JPEG"))]

        raise ValueError(f"Unsupported Tiny ImageNet split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


class PreparedImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read an already-prepared split directory with class folders or flat images."""

    def __init__(
        self,
        split_root: Path,
        transform: transforms.Compose,
        manifest_path: Path | None = None,
        dataset_root: Path | None = None,
        split: str | None = None,
    ) -> None:
        self.split_root = split_root
        self.transform = transform
        self.classes, self.class_to_idx, self.samples = self._build_samples(
            split_root,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            split=split,
        )

    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS

    @classmethod
    def _build_samples(
        cls,
        split_root: Path,
        manifest_path: Path | None = None,
        dataset_root: Path | None = None,
        split: str | None = None,
    ) -> tuple[list[str], dict[str, int], list[tuple[Path, int]]]:
        if not split_root.exists():
            raise FileNotFoundError(f"Prepared image split does not exist: {split_root}")

        split_manifest_path = split_root.parent / f"{split_root.name}_manifest.jsonl"
        if manifest_path is not None and manifest_path.exists():
            return cls._build_samples_from_manifest(
                split_root,
                manifest_path,
                dataset_root=dataset_root,
                split=split,
            )
        if split_manifest_path.exists():
            return cls._build_samples_from_manifest(
                split_root,
                split_manifest_path,
                dataset_root=dataset_root,
                split=split,
            )

        class_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
        if class_dirs:
            classes = [path.name for path in class_dirs]
            class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
            samples = [
                (image_path, class_to_idx[class_dir.name])
                for class_dir in class_dirs
                for image_path in sorted(class_dir.rglob("*"))
                if cls._is_image(image_path)
            ]
        else:
            classes = ["unlabeled"]
            class_to_idx = {"unlabeled": 0}
            samples = [
                (image_path, 0)
                for image_path in sorted(split_root.iterdir())
                if cls._is_image(image_path)
            ]

        if not samples:
            raise FileNotFoundError(f"No image files found in prepared split: {split_root}")

        return classes, class_to_idx, samples

    @classmethod
    def _build_samples_from_manifest(
        cls,
        split_root: Path,
        manifest_path: Path,
        dataset_root: Path | None = None,
        split: str | None = None,
    ) -> tuple[list[str], dict[str, int], list[tuple[Path, int]]]:
        if dataset_root is None:
            dataset_root = split_root.parent.parent
        normalized_split = _normalize_split(split) if split is not None else None
        records: list[tuple[Path, str]] = []
        class_names: set[str] = set()
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_split = record.get("split")
                if record_split is not None and normalized_split is not None:
                    if _normalize_split(str(record_split)) != normalized_split:
                        continue
                relative_path = record.get("relative_path") or record.get("path")
                if not relative_path:
                    raise ValueError(f"Manifest record missing image path: {manifest_path}")
                path = dataset_root / relative_path
                if record_split is None and not _is_relative_to(path, split_root):
                    continue
                class_name = _record_class_name(record)
                records.append((path, class_name))
                class_names.add(class_name)

        if not records:
            raise FileNotFoundError(f"No image records found in manifest: {manifest_path}")

        classes = sorted(class_names)
        class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
        samples = [(path, class_to_idx[class_name]) for path, class_name in records]
        return classes, class_to_idx, samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def _image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _record_class_name(record: dict[str, Any]) -> str:
    for key in ("class_name", "label_name", "wnid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    label = record.get("label")
    if label is not None:
        return str(label)
    return "unlabeled"


def _normalize_split(split: str) -> str:
    return "val" if split in {"val", "validation", "valid"} else split


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _prepared_split_root(dataset_root: Path, split: str, allow_unsplit: bool = False) -> Path:
    if split == "train":
        candidates = ["train", "train_64x64"]
    elif split in {"val", "validation"}:
        candidates = ["validation", "val", "valid_64x64"]
    elif split == "test":
        candidates = ["test"]
    else:
        raise ValueError(f"Unsupported prepared-image split: {split}")

    if allow_unsplit:
        candidates = [*candidates, "images", "all", "train"]

    for candidate in candidates:
        path = dataset_root / candidate
        if path.exists():
            return path

    expected = ", ".join(str(dataset_root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Prepared image split not found. Expected one of: {expected}")


def build_dataset(config: DataConfig, split: str = "train") -> Dataset[Any]:
    dataset = config.dataset.lower()
    root = Path(config.root)
    transform = _image_transform(config.image_size)

    if dataset == "random":
        return RandomImageDataset(
            length=max(config.batch_size * 4, 16),
            channels=config.channels,
            image_size=config.image_size,
        )

    if dataset == "cifar10":
        train = split == "train"
        return datasets.CIFAR10(
            root=str(root / "cifar10" / "extracted"),
            train=train,
            transform=transform,
            download=False,
        )

    if dataset == "tiny_imagenet_200":
        split_dir = root / "tiny_imagenet_200" / "extracted" / "tiny-imagenet-200"
        return TinyImageNetDataset(split_dir, split=split, transform=transform)

    if dataset in _PREPARED_IMAGE_DATASETS:
        dataset_root = root / dataset
        split_dir = _prepared_split_root(
            dataset_root / "extracted",
            split,
            allow_unsplit=dataset in _UNSPLIT_PREPARED_DATASETS,
        )
        manifest_path = dataset_root / "metadata" / "image_manifest.jsonl"
        return PreparedImageDataset(
            split_dir,
            transform=transform,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            split=split,
        )

    raise ValueError(f"Unknown dataset: {config.dataset}")


def build_dataloader(
    config: DataConfig,
    split: str = "train",
    drop_last: bool | None = None,
) -> DataLoader[Any]:
    dataset = build_dataset(config, split=split)
    if drop_last is None:
        drop_last = split == "train"
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=split == "train",
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )

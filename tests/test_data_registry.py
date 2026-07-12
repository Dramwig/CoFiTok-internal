from pathlib import Path

from PIL import Image

from cofitok.configs import DataConfig
from cofitok.data import registry


def test_cifar10_uses_extracted_root(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    class FakeCIFAR10:
        def __init__(self, root: str, train: bool, transform: object, download: bool) -> None:
            calls["root"] = root
            calls["train"] = train
            calls["download"] = download

    monkeypatch.setattr(registry.datasets, "CIFAR10", FakeCIFAR10)
    config = DataConfig(dataset="cifar10", root=str(tmp_path), image_size=32)

    registry.build_dataset(config, split="val")

    assert calls == {
        "root": str(tmp_path / "cifar10" / "extracted"),
        "train": False,
        "download": False,
    }


def test_tiny_imagenet_val_uses_annotation_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tiny-imagenet-200"
    image_dir = dataset_root / "val" / "images"
    image_dir.mkdir(parents=True)
    (dataset_root / "wnids.txt").write_text("n00000001\nn00000002\n", encoding="utf-8")
    (dataset_root / "val" / "val_annotations.txt").write_text(
        "val_0.JPEG\tn00000002\t0\t0\t64\t64\n",
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64), color=(128, 64, 32)).save(image_dir / "val_0.JPEG")

    dataset = registry.TinyImageNetDataset(
        dataset_root,
        split="val",
        transform=registry._image_transform(64),
    )
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 1


def test_downsampled_imagenet_64_uses_validation_split(tmp_path: Path) -> None:
    image_dir = tmp_path / "downsampled_imagenet_64" / "extracted" / "validation" / "unlabeled"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(16, 32, 48)).save(image_dir / "val_00000000.png")

    config = DataConfig(dataset="downsampled_imagenet_64", root=str(tmp_path), image_size=64)
    dataset = registry.build_dataset(config, split="val")
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.classes == ["unlabeled"]


def test_downsampled_imagenet_64_uses_strict_torrent_split_names(tmp_path: Path) -> None:
    image_dir = tmp_path / "downsampled_imagenet_64" / "extracted" / "valid_64x64"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(16, 32, 48)).save(image_dir / "00001.png")

    config = DataConfig(dataset="downsampled_imagenet_64", root=str(tmp_path), image_size=64)
    dataset = registry.build_dataset(config, split="validation")
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.classes == ["unlabeled"]


def test_prepared_image_dataset_accepts_flat_unlabeled_split(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    split_root.mkdir()
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(split_root / "image_00000000.png")

    dataset = registry.PreparedImageDataset(split_root, transform=registry._image_transform(32))
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 32, 32)
    assert label == 0
    assert dataset.class_to_idx == {"unlabeled": 0}


def test_prepared_image_dataset_prefers_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "downsampled_imagenet_64"
    image_dir = dataset_root / "extracted" / "train" / "unlabeled" / "shard_00000"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "00000000.png"
    Image.new("RGB", (64, 64), color=(8, 16, 24)).save(image_path)
    (dataset_root / "extracted" / "train_manifest.jsonl").write_text(
        (
            '{"class_name":"unlabeled","index":0,"label":null,'
            '"relative_path":"extracted/train/unlabeled/shard_00000/00000000.png",'
            '"sha256":"unused","split":"train","width":64,"height":64}\n'
        ),
        encoding="utf-8",
    )

    dataset = registry.PreparedImageDataset(
        dataset_root / "extracted" / "train",
        transform=registry._image_transform(64),
    )
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.samples[0][0] == image_path


def test_imagenet_1k_64x64_hf_alias_uses_prepared_layout(tmp_path: Path) -> None:
    image_dir = tmp_path / "imagenet_1k_64x64_hf" / "extracted" / "validation" / "0000" / "shard_00000"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(24, 48, 96)).save(image_dir / "00000000.png")

    config = DataConfig(dataset="imagenet_1k_64x64_hf", root=str(tmp_path), image_size=64)
    dataset = registry.build_dataset(config, split="validation")
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.classes == ["0000"]


def test_ffhq_64_uses_metadata_manifest_for_unlabeled_shards(tmp_path: Path) -> None:
    image_dir = tmp_path / "ffhq_64" / "extracted" / "images" / "00000"
    metadata_dir = tmp_path / "ffhq_64" / "metadata"
    image_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    image_path = image_dir / "img00000000.png"
    Image.new("RGB", (64, 64), color=(64, 64, 64)).save(image_path)
    (metadata_dir / "image_manifest.jsonl").write_text(
        '{"height":64,"index":0,"mode":"RGB","path":"extracted/images/00000/img00000000.png","width":64}\n',
        encoding="utf-8",
    )

    config = DataConfig(dataset="ffhq_64", root=str(tmp_path), image_size=64)
    dataset = registry.build_dataset(config, split="val")
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.classes == ["unlabeled"]


def test_afhqv2_64_uses_label_name_from_metadata_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "afhqv2_64" / "extracted" / "train" / "dog"
    metadata_dir = tmp_path / "afhqv2_64" / "metadata"
    image_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    image_path = image_dir / "000000.png"
    Image.new("RGB", (64, 64), color=(32, 64, 96)).save(image_path)
    (metadata_dir / "image_manifest.jsonl").write_text(
        (
            '{"height":64,"index":0,"label":1,"label_name":"dog",'
            '"path":"extracted/train/dog/000000.png","width":64}\n'
        ),
        encoding="utf-8",
    )

    config = DataConfig(dataset="afhqv2_64", root=str(tmp_path), image_size=64)
    dataset = registry.build_dataset(config, split="val")
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.classes == ["dog"]


def test_imagenet_256_manifest_filters_split(tmp_path: Path) -> None:
    train_dir = tmp_path / "imagenet_256_10pct" / "extracted" / "train" / "n01440764"
    val_dir = tmp_path / "imagenet_256_10pct" / "extracted" / "val" / "n01440764"
    metadata_dir = tmp_path / "imagenet_256_10pct" / "metadata"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    Image.new("RGB", (256, 256), color=(10, 10, 10)).save(train_dir / "train.jpg")
    Image.new("RGB", (256, 256), color=(20, 20, 20)).save(val_dir / "val.jpg")
    (metadata_dir / "image_manifest.jsonl").write_text(
        (
            '{"height":256,"label":0,"path":"extracted/train/n01440764/train.jpg",'
            '"split":"train","width":256,"wnid":"n01440764"}\n'
            '{"height":256,"label":0,"path":"extracted/val/n01440764/val.jpg",'
            '"split":"val","width":256,"wnid":"n01440764"}\n'
        ),
        encoding="utf-8",
    )

    config = DataConfig(dataset="imagenet_256_10pct", root=str(tmp_path), image_size=64)
    dataset = registry.build_dataset(config, split="validation")
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
    assert dataset.samples[0][0] == val_dir / "val.jpg"


def _write_flat_images(split_root: Path, count: int) -> None:
    split_root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (8, 8), color=(index, index, index)).save(split_root / f"{index:04d}.png")


def test_build_dataloader_keeps_eval_tail_batch_by_default(tmp_path: Path) -> None:
    _write_flat_images(
        tmp_path / "downsampled_imagenet_64" / "extracted" / "train" / "unlabeled",
        13,
    )
    _write_flat_images(
        tmp_path / "downsampled_imagenet_64" / "extracted" / "validation" / "unlabeled",
        13,
    )
    config = DataConfig(dataset="downsampled_imagenet_64", root=str(tmp_path), batch_size=10, image_size=8)
    train_loader = registry.build_dataloader(config, split="train")
    val_loader = registry.build_dataloader(config, split="val")

    assert len(train_loader.dataset) == 13
    assert len(train_loader) == 1
    assert train_loader.drop_last is True
    assert len(val_loader.dataset) == 13
    assert len(val_loader) == 2
    assert val_loader.drop_last is False


def test_build_dataloader_can_force_eval_drop_last(tmp_path: Path) -> None:
    _write_flat_images(
        tmp_path / "downsampled_imagenet_64" / "extracted" / "validation" / "unlabeled",
        13,
    )
    config = DataConfig(dataset="downsampled_imagenet_64", root=str(tmp_path), batch_size=10, image_size=8)

    val_loader = registry.build_dataloader(config, split="val", drop_last=True)

    assert len(val_loader.dataset) == 13
    assert len(val_loader) == 1
    assert val_loader.drop_last is True

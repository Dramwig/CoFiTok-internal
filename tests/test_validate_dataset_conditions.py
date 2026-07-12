import json
from pathlib import Path

import pytest

from scripts.validate_dataset_conditions import validate_dataset_conditions


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_conditions(root: Path) -> None:
    _write(
        root / "datasets_2026-07-07.md",
        """
cifar10
Source:
cifar-10-python.tar.gz
sha256
Train: 50,000 images
Test: 10,000 images
Resolution: 32x32 RGB
Labels: class labels are available
Raw preprocessing: none
Server dataset directory size
/root/autodl-tmp/CoFiTok/datasets/cifar10
/root/autodl-tmp/CoFiTok/checkpoints

tiny_imagenet_200
Source:
tiny-imagenet-200.zip
sha256
Train: 100,000 images
Validation: 10,000 images
Test: 10,000 images
Resolution: 64x64 RGB
Labels: class labels are available
Raw preprocessing: none
Server dataset directory size
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200
/root/autodl-tmp/CoFiTok/checkpoints
""",
    )
    _write(
        root / "imagenet_1k_64x64_hf_plan_2026-07-08.md",
        """
dataset alias: imagenet_1k_64x64_hf
Hugging Face repo: benjamin-paine/imagenet-1k-64x64
repo URL: https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64
columns: image, label
class labels: ImageNet-1K labels
train_manifest rows: 1,281,167
validation_manifest rows: 50,000
dataset dir size: 14G
Keep its source and preprocessing distinct from `downsampled_imagenet_64`
""",
    )
    _write_json(
        root / "imagenet_1k_64x64_hf_manifest_summary_2026-07-08.json",
        {
            "dataset_alias": "imagenet_1k_64x64_hf",
            "source_loader": "benjamin-paine/imagenet-1k-64x64",
            "splits": [{"split": "train", "examples": 1281167}, {"split": "validation", "examples": 50000}],
        },
    )
    _write(
        root / "downsampled_imagenet_64_plan_2026-07-08.md",
        """
alias: downsampled_imagenet_64
preferred public loader: TensorFlow Datasets downsampled_imagenet/64x64
result:
failed before downloading data
HTTP code: 404
Academic Torrents fallback attempt
imagenet_1k_64x64_hf
This fallback is ImageNet-1K 64x64 repack/resized data, not the exact TFDS
""",
    )
    _write_json(
        root / "downsampled_imagenet_64_tfds_inspect_2026-07-08.json",
        {
            "dataset_alias": "downsampled_imagenet_64",
            "source_loader": "downsampled_imagenet",
            "tfds_version": "4.9.10",
        },
    )
    _write(
        root / "downsampled_imagenet_64_2026-07-09.md",
        """
# downsampled_imagenet_64 Dataset Record
Completed and verified from the strict Academic Torrents source.
https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b
96816a530ee002254d29bf7a61c0c158d3dedc3b
aria2c magnet+DHT on pro6000
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/train_64x64.tar
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/valid_64x64.tar
e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7
eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd
1,281,149
49,999
No crop, resize, normalization
""",
    )
    _write_json(
        root / "downsampled_imagenet_64_manifest_summary_2026-07-09.json",
        {
            "alias": "downsampled_imagenet_64",
            "status": "completed",
            "source": {"info_hash": "96816a530ee002254d29bf7a61c0c158d3dedc3b"},
            "raw_files": [
                {
                    "name": "96816a530ee002254d29bf7a61c0c158d3dedc3b.torrent",
                    "bytes": 240592,
                    "sha256": "1104077da8c88db149d74ffb6d1f9fe17b32abf535880966ee1684559820cdfa",
                },
                {
                    "name": "train_64x64.tar",
                    "bytes": 12112588800,
                    "sha256": "e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7",
                },
                {
                    "name": "valid_64x64.tar",
                    "bytes": 477255680,
                    "sha256": "eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd",
                },
            ],
            "splits": {"train": {"png_files": 1281149}, "valid": {"png_files": 49999}},
            "sample_png_verification": {
                "train": [{"name": "0000001.png", "size": [64, 64]}],
                "valid": [{"name": "00001.png", "size": [64, 64]}],
            },
        },
    )


def test_validate_dataset_conditions_accepts_complete_records(tmp_path: Path) -> None:
    _complete_conditions(tmp_path)

    result = validate_dataset_conditions(tmp_path)

    assert result["status"] == "ok"
    assert result["check_count"] == 4


def test_validate_dataset_conditions_reports_missing_manifest_rows(tmp_path: Path) -> None:
    _complete_conditions(tmp_path)
    _write_json(
        tmp_path / "imagenet_1k_64x64_hf_manifest_summary_2026-07-08.json",
        {
            "dataset_alias": "imagenet_1k_64x64_hf",
            "source_loader": "benjamin-paine/imagenet-1k-64x64",
            "splits": [{"split": "train", "examples": 10}],
        },
    )

    result = validate_dataset_conditions(tmp_path)

    assert result["status"] == "missing"
    hf_check = next(check for check in result["checks"] if check["name"] == "imagenet_hf_records")
    assert any("train examples" in item for item in hf_check["missing"])

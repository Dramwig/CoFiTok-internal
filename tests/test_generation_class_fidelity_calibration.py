from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.calibrate_generation_class_fidelity_classifier import (
    CLAIM_BOUNDARY,
    build_checks,
    class_order_sha256,
    load_class_order,
    select_balanced_real_images,
    selection_sha256,
)


def _write_order_sources(root: Path, order: list[str]) -> tuple[Path, Path, Path]:
    mapping = root / "mapping.json"
    timm = root / "imagenet_synsets.txt"
    categories = root / "imagenet.categories"
    mapping.write_text(
        json.dumps({"label_to_wnid": {str(i): wnid for i, wnid in enumerate(order)}}),
        encoding="utf-8",
    )
    timm.write_text("\n".join(order) + "\n", encoding="utf-8")
    categories.write_text(
        "\n".join(f"class {index},{wnid}" for index, wnid in enumerate(order)) + "\n",
        encoding="utf-8",
    )
    return mapping, timm, categories


def test_load_class_order_requires_exact_three_way_match(tmp_path: Path) -> None:
    order = [f"n{index:08d}" for index in range(5)]
    mapping, timm, categories = _write_order_sources(tmp_path, order)

    assert load_class_order(
        mapping,
        timm,
        categories,
        expected_num_classes=5,
    ) == order
    assert len(class_order_sha256(order)) == 64


def test_load_class_order_rejects_torchvision_drift(tmp_path: Path) -> None:
    order = [f"n{index:08d}" for index in range(5)]
    mapping, timm, categories = _write_order_sources(tmp_path, order)
    categories.write_text(
        "\n".join(
            f"class {index},{wnid}"
            for index, wnid in enumerate([order[1], order[0], *order[2:]])
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="torchvision"):
        load_class_order(
            mapping,
            timm,
            categories,
            expected_num_classes=5,
        )


def test_balanced_selection_is_deterministic_and_source_bound(tmp_path: Path) -> None:
    order = [f"n{index:08d}" for index in range(5)]
    for class_index, wnid in enumerate(order):
        class_dir = tmp_path / "val" / wnid
        class_dir.mkdir(parents=True)
        for image_index in (1, 0):
            Image.new(
                "RGB",
                (8, 8),
                color=(class_index, image_index, 0),
            ).save(class_dir / f"{image_index}.jpg")

    selected = select_balanced_real_images(
        tmp_path / "val",
        order,
        samples_per_class=1,
    )
    assert [path.name for path, _ in selected] == ["0.jpg"] * 5
    assert [target for _, target in selected] == list(range(5))
    first_sha = selection_sha256(selected, real_dir=tmp_path / "val")
    second_sha = selection_sha256(
        select_balanced_real_images(
            tmp_path / "val",
            order,
            samples_per_class=1,
        ),
        real_dir=tmp_path / "val",
    )
    assert first_sha == second_sha


def test_build_checks_separates_execution_from_calibration_status() -> None:
    passing = build_checks(
        {"top1_accuracy": 0.6, "top5_accuracy": 0.8},
        min_top1=0.5,
        min_top5=0.75,
    )
    holding = build_checks(
        {"top1_accuracy": 0.4, "top5_accuracy": 0.8},
        min_top1=0.5,
        min_top5=0.75,
    )
    assert [row["status"] for row in passing] == ["pass", "pass"]
    assert [row["status"] for row in holding] == ["hold", "pass"]
    assert CLAIM_BOUNDARY["classifier_calibration_only"] is True
    assert CLAIM_BOUNDARY["training_launch_allowed"] is False
    assert CLAIM_BOUNDARY["promotion_or_release_allowed"] is False

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.utils import save_image

from scripts.build_generation_visual_audit import build_visual_audit


def _images(directory: Path, *, count: int, offset: float) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        image = torch.full((3, 8, 8), min(1.0, offset + index * 0.1))
        save_image(image, directory / f"{index:06d}.png")


def _sampling(
    directories: dict[int, Path],
    *,
    checkpoint_sha: str,
    checkpoint_step: int = 300_000,
) -> dict:
    return {
        "status": "completed",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "sampling": {"prefix_budgets": list(directories)},
        "output_dirs": {
            str(budget): directory.resolve().as_posix()
            for budget, directory in directories.items()
        },
        "sample_sets": {
            str(budget): {"count": 3, "sha256": str(budget) * 64}
            for budget in directories
        },
    }


def test_visual_audit_builds_hash_bound_matched_and_prefix_panels(tmp_path) -> None:
    cofitok = tmp_path / "cofitok"
    dense = tmp_path / "dense"
    prefix_1 = tmp_path / "prefix_1"
    prefix_2 = tmp_path / "prefix_2"
    _images(cofitok, count=3, offset=0.1)
    _images(dense, count=3, offset=0.2)
    _images(prefix_1, count=3, offset=0.3)
    _images(prefix_2, count=3, offset=0.4)

    report = build_visual_audit(
        cofitok_sampling=_sampling({2: cofitok}, checkpoint_sha="a" * 64),
        dense_sampling=_sampling({1: dense}, checkpoint_sha="b" * 64),
        prefix_sampling=_sampling(
            {1: prefix_1, 2: prefix_2}, checkpoint_sha="a" * 64
        ),
        cofitok_dir=cofitok,
        dense_dir=dense,
        indices=[0, 2],
        prefix_indices=[0, 2],
        prefix_budgets=[1, 2],
        output_dir=tmp_path / "audit",
    )

    assert report["status"] == "completed"
    assert len(report["git"]["revision"]) == 40
    assert isinstance(report["git"]["tracked_dirty"], bool)
    assert report["sources"]["cofitok"]["checkpoint_sha256"] == "a" * 64
    assert report["sources"]["dense_identity"]["sample_set_sha256"] == "1" * 64
    assert report["statistics"]["cofitok"]["exact_duplicate_count"] == 0
    assert report["statistics"]["cofitok"]["total_variation"] == pytest.approx(0.0)
    assert report["statistics"]["paired_similarity"]["image_count"] == 2
    assert report["statistics"]["paired_similarity"]["per_image"][0]["index"] == 0
    assert -1.0 <= report["statistics"]["paired_similarity"][
        "pixel_correlation_mean"
    ] <= 1.0
    assert report["panels"]["cofitok_prefix_paths"]["image_count"] == 4
    for panel in report["panels"].values():
        assert len(panel["sha256"]) == 64
        assert Path(panel["path"]).is_file()


def test_visual_audit_rejects_exact_duplicate_fixed_samples(tmp_path) -> None:
    cofitok = tmp_path / "cofitok"
    dense = tmp_path / "dense"
    prefix = tmp_path / "prefix"
    _images(cofitok, count=2, offset=0.1)
    _images(dense, count=2, offset=0.2)
    _images(prefix, count=2, offset=0.3)
    (cofitok / "000001.png").write_bytes((cofitok / "000000.png").read_bytes())

    with pytest.raises(ValueError, match="exact duplicate"):
        build_visual_audit(
            cofitok_sampling=_sampling({1: cofitok}, checkpoint_sha="a" * 64),
            dense_sampling=_sampling({1: dense}, checkpoint_sha="b" * 64),
            prefix_sampling=_sampling({1: prefix}, checkpoint_sha="a" * 64),
            cofitok_dir=cofitok,
            dense_dir=dense,
            indices=[0, 1],
            prefix_indices=[0, 1],
            prefix_budgets=[1],
            output_dir=tmp_path / "audit",
        )

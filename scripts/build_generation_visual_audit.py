from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _indices(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("indices must be comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("visual audit indices must be unique and non-negative")
    return values


def _budgets(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("prefix budgets must be comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(value < 1 for value in values):
        raise ValueError("visual audit prefix budgets are invalid")
    return values


def _sampling_source(
    report: dict[str, Any],
    *,
    budget: int,
    expected_dir: Path | None = None,
) -> dict[str, Any]:
    if report.get("status") != "completed":
        raise ValueError("visual audit requires completed sampling reports")
    output_dir = Path(report["output_dirs"][str(budget)]).resolve()
    if expected_dir is not None and output_dir != expected_dir.resolve():
        raise ValueError("visual audit directory differs from sampling report")
    sample_set = report.get("sample_sets", {}).get(str(budget), {})
    if int(sample_set.get("count", -1)) < 1 or len(str(sample_set.get("sha256", ""))) != 64:
        raise ValueError("visual audit sampling report lacks sample-set provenance")
    if len(str(report.get("checkpoint_sha256", ""))) != 64:
        raise ValueError("visual audit sampling report lacks checkpoint provenance")
    return {
        "directory": output_dir,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "checkpoint_step": int(report["checkpoint_step"]),
        "sample_set_sha256": sample_set["sha256"],
        "sample_count": int(sample_set["count"]),
        "budget": budget,
    }


def _load_indexed(directory: Path, indices: list[int]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images = []
    evidence = []
    expected_shape = None
    for index in indices:
        path = directory / f"{index:06d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"visual audit sample is missing: {path}")
        image = read_image(path).float() / 255.0
        if expected_shape is None:
            expected_shape = tuple(image.shape)
        elif tuple(image.shape) != expected_shape:
            raise ValueError("visual audit samples have inconsistent shapes")
        images.append(image)
        evidence.append(
            {
                "index": index,
                "path": path.resolve().as_posix(),
                "sha256": file_sha256(path),
            }
        )
    return images, evidence


def _atomic_grid(images: list[torch.Tensor], path: Path, *, nrow: int) -> dict[str, Any]:
    if not images or nrow < 1:
        raise ValueError("visual audit grid is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(torch.stack(images), nrow=nrow, padding=2, pad_value=1.0)
    temporary = path.with_name(f".{path.name}.part")
    try:
        save_image(grid, temporary, format="png")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.resolve().as_posix(),
        "sha256": file_sha256(path),
        "image_count": len(images),
        "nrow": nrow,
        "grid_shape": list(grid.shape),
    }


def _selected_statistics(images: list[torch.Tensor], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    stacked = torch.stack(images)
    hashes = [row["sha256"] for row in evidence]
    return {
        "image_count": len(images),
        "exact_duplicate_count": len(hashes) - len(set(hashes)),
        "pixel_mean": float(stacked.mean().item()),
        "pixel_std": float(stacked.std().item()),
        "pixel_min": float(stacked.min().item()),
        "pixel_max": float(stacked.max().item()),
    }


def build_visual_audit(
    *,
    cofitok_sampling: dict[str, Any],
    dense_sampling: dict[str, Any],
    prefix_sampling: dict[str, Any],
    cofitok_dir: Path,
    dense_dir: Path,
    indices: list[int],
    prefix_indices: list[int],
    prefix_budgets: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    git = git_provenance(PROJECT_ROOT)
    cofitok_budget = max(int(value) for value in cofitok_sampling["sampling"]["prefix_budgets"])
    dense_budget = max(int(value) for value in dense_sampling["sampling"]["prefix_budgets"])
    cofitok_source = _sampling_source(
        cofitok_sampling, budget=cofitok_budget, expected_dir=cofitok_dir
    )
    dense_source = _sampling_source(
        dense_sampling, budget=dense_budget, expected_dir=dense_dir
    )
    if cofitok_source["checkpoint_step"] != dense_source["checkpoint_step"]:
        raise ValueError("visual audit formal checkpoints use different steps")
    if max(indices) >= min(cofitok_source["sample_count"], dense_source["sample_count"]):
        raise ValueError("visual audit index exceeds the formal sample set")

    cofitok_images, cofitok_evidence = _load_indexed(cofitok_dir, indices)
    dense_images, dense_evidence = _load_indexed(dense_dir, indices)
    if tuple(cofitok_images[0].shape) != tuple(dense_images[0].shape):
        raise ValueError("visual audit matched samples have different image shapes")

    prefix_images = []
    prefix_evidence = []
    prefix_sources = {}
    resolved_prefix_sources = {
        budget: _sampling_source(prefix_sampling, budget=budget)
        for budget in prefix_budgets
    }
    for budget, source in resolved_prefix_sources.items():
        if source["checkpoint_sha256"] != cofitok_source["checkpoint_sha256"]:
            raise ValueError("prefix visual audit uses another CoFiTok checkpoint")
        prefix_sources[str(budget)] = {
            key: value for key, value in source.items() if key != "directory"
        }
    for index in prefix_indices:
        for budget in prefix_budgets:
            source = resolved_prefix_sources[budget]
            if index >= source["sample_count"]:
                raise ValueError("visual audit index exceeds prefix sample set")
            images, evidence = _load_indexed(source["directory"], [index])
            prefix_images.extend(images)
            prefix_evidence.extend([{**evidence[0], "prefix_budget": budget}])

    output_dir.mkdir(parents=True, exist_ok=True)
    panels = {
        "cofitok": _atomic_grid(
            cofitok_images,
            output_dir / "cofitok_fixed_samples.png",
            nrow=min(4, len(indices)),
        ),
        "dense_identity": _atomic_grid(
            dense_images,
            output_dir / "dense_fixed_samples.png",
            nrow=min(4, len(indices)),
        ),
        "cofitok_prefix_paths": _atomic_grid(
            prefix_images,
            output_dir / "cofitok_prefix_paths.png",
            nrow=len(prefix_budgets),
        ),
    }
    statistics = {
        "cofitok": _selected_statistics(cofitok_images, cofitok_evidence),
        "dense_identity": _selected_statistics(dense_images, dense_evidence),
    }
    if any(row["exact_duplicate_count"] > 0 for row in statistics.values()):
        raise ValueError("fixed formal samples contain exact duplicate PNGs")
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "deterministic_visual_quality_audit",
        "git": git,
        "claim_policy": {
            "quantitative_metric": False,
            "reason": "Fixed panels support human inspection and do not replace formal FID/IS/precision/recall.",
        },
        "indices": indices,
        "prefix_indices": prefix_indices,
        "prefix_budgets": prefix_budgets,
        "sources": {
            "cofitok": {
                key: value for key, value in cofitok_source.items() if key != "directory"
            },
            "dense_identity": {
                key: value for key, value in dense_source.items() if key != "directory"
            },
            "cofitok_prefix": prefix_sources,
        },
        "statistics": statistics,
        "panels": panels,
        "selected_files": {
            "cofitok": cofitok_evidence,
            "dense_identity": dense_evidence,
            "cofitok_prefix": prefix_evidence,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic matched and prefix generation visual-audit panels."
    )
    parser.add_argument("--cofitok-sampling-report", required=True)
    parser.add_argument("--dense-sampling-report", required=True)
    parser.add_argument("--prefix-sampling-report", required=True)
    parser.add_argument("--cofitok-dir", required=True)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--prefix-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--prefix-budgets", default="1,2,4,8")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = build_visual_audit(
        cofitok_sampling=_read(args.cofitok_sampling_report),
        dense_sampling=_read(args.dense_sampling_report),
        prefix_sampling=_read(args.prefix_sampling_report),
        cofitok_dir=Path(args.cofitok_dir),
        dense_dir=Path(args.dense_dir),
        indices=_indices(args.indices),
        prefix_indices=_indices(args.prefix_indices),
        prefix_budgets=_budgets(args.prefix_budgets),
        output_dir=output_dir,
    )
    write_json_report(output_dir / "visual_audit_report.json", report)
    print(output_dir / "visual_audit_report.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torchvision.utils import save_image

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RESULT_ROLE,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import file_identity
from scripts.build_generation_requested_class_visual_audit import (
    CLAIM_BOUNDARY,
    REPORT_FILENAME,
    build_requested_class_visual_audit,
    parse_indices,
)
from scripts.calibrate_generation_class_fidelity_classifier import (
    CLAIM_BOUNDARY as CALIBRATION_CLAIM_BOUNDARY,
    REPORT_ROLE as CALIBRATION_REPORT_ROLE,
    class_order_sha256,
    select_balanced_real_images,
    selection_sha256,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _image(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.full((3, 256, 256), value), path)


def _calibration(root: Path, order: list[str]) -> tuple[Path, Path]:
    real_dir = root / "real"
    for index, wnid in enumerate(order):
        _image(real_dir / wnid / "000000.png", 0.05 + index * 0.05)
    mapping = root / "label_to_wnid.json"
    timm = root / "imagenet_synsets.txt"
    categories = root / "imagenet.categories"
    _write_json(
        mapping,
        {"label_to_wnid": {str(index): wnid for index, wnid in enumerate(order)}},
    )
    timm.write_text("\n".join(order) + "\n", encoding="utf-8")
    categories.write_text(
        "\n".join(f"class {index},{wnid}" for index, wnid in enumerate(order)) + "\n",
        encoding="utf-8",
    )
    selected = select_balanced_real_images(real_dir, order, samples_per_class=1)
    report = root / "calibration.json"
    _write_json(
        report,
        {
            "status": "completed",
            "calibration_status": "pass",
            "role": CALIBRATION_REPORT_ROLE,
            "claim_boundary": CALIBRATION_CLAIM_BOUNDARY,
            "paths": {"real_dir": real_dir.resolve().as_posix()},
            "sources": {
                "label_to_wnid": file_identity(mapping),
                "timm_synsets": file_identity(timm),
                "torchvision_categories": file_identity(categories),
            },
            "selection": {
                "samples_per_class": 1,
                "sample_count": len(order),
                "class_order_sha256": class_order_sha256(order),
                "selected_sample_set_sha256": selection_sha256(
                    selected,
                    real_dir=real_dir,
                ),
            },
            "parameters": {"num_classes": len(order)},
            "git": {
                "revision": "c" * 40,
                "branch": "calibration",
                "tracked_dirty": False,
            },
            "runtime_environment_sha256": "d" * 64,
            "classifier": {"name": "test"},
            "metrics": {"top1_accuracy": 0.8, "top5_accuracy": 0.9},
        },
    )
    return report, real_dir


def _sampling_report(
    path: Path,
    directory: Path,
    *,
    budget: int,
    guidance_scale: float = 1.5,
) -> None:
    sampling = {
        "prefix_budgets": [budget],
        "start_index": 5,
        "num_samples": 5,
        "image_shape": [3, 256, 256],
        "class_schedule": "balanced_modulo",
        "sampler": "ddim",
        "sample_steps": 100,
        "seed": 0,
        "guidance_scale": guidance_scale,
    }
    _write_json(
        path,
        {
            "schema_version": 6,
            "status": "completed",
            "weights": "ema",
            "checkpoint_step": 50_000,
            "checkpoint_sha256": str(budget) * 64,
            "sampling_manifest_sha256": "a" * 64,
            "output_dirs": {str(budget): directory.resolve().as_posix()},
            "sample_sets": {str(budget): {"count": 5, "sha256": "b" * 64}},
            "sampling": sampling,
            "git": {
                "revision": "e" * 40,
                "branch": "sampling",
                "tracked_dirty": False,
            },
            "runtime_environment_sha256": "f" * 64,
        },
    )


def _quality_result(
    path: Path,
    cofitok_report: Path,
    dense_report: Path,
    *,
    num_classes: int = 5,
) -> None:
    cofitok = json.loads(cofitok_report.read_text(encoding="utf-8"))
    dense = json.loads(dense_report.read_text(encoding="utf-8"))
    cofitok_budget = int(cofitok["sampling"]["prefix_budgets"][0])
    dense_budget = int(dense["sampling"]["prefix_budgets"][0])
    matched_sampling = {
        key: value
        for key, value in cofitok["sampling"].items()
        if key not in {"prefix_budgets", "num_classes"}
    }
    cofitok_class_source = path.parent / "cofitok_class_fidelity.json"
    dense_class_source = path.parent / "dense_class_fidelity.json"
    _write_json(cofitok_class_source, {"status": "completed"})
    _write_json(dense_class_source, {"status": "completed"})
    _write_json(
        path,
        {
            "schema_version": 1,
            "status": "completed",
            "role": QUALITY_BRIDGE_RESULT_ROLE,
            "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
            "training": {
                "pair_validation": {
                    "status": "pass",
                    "training_recipe": {
                        "schema": "test_training_recipe",
                        "valid": True,
                        "expected_shared": {"model.num_classes": num_classes},
                        "observed": {
                            "cofitok": {"model.num_classes": num_classes},
                            "dense_identity": {"model.num_classes": num_classes},
                        },
                    },
                }
            },
            "terminal": {
                "methods": {
                    "cofitok": {
                        "sampling_report": file_identity(cofitok_report),
                        "sampling": cofitok["sampling"],
                        "selected_prefix_budget": cofitok_budget,
                        "sample_count": cofitok["sample_sets"][str(cofitok_budget)][
                            "count"
                        ],
                        "sample_set_sha256": cofitok["sample_sets"][str(cofitok_budget)][
                            "sha256"
                        ],
                    },
                    "dense_identity": {
                        "sampling_report": file_identity(dense_report),
                        "sampling": dense["sampling"],
                        "selected_prefix_budget": dense_budget,
                        "sample_count": dense["sample_sets"][str(dense_budget)][
                            "count"
                        ],
                        "sample_set_sha256": dense["sample_sets"][str(dense_budget)][
                            "sha256"
                        ],
                    },
                },
                "class_fidelity": {
                    "status": "hold",
                    "classifier": {"num_classes": num_classes},
                    "metrics": {
                        "cofitok": {"num_classes": num_classes},
                        "dense_identity": {"num_classes": num_classes},
                    },
                    "sampling_contract": {
                        "sampling": matched_sampling,
                        "sample_count_per_method": cofitok["sample_sets"][
                            str(cofitok_budget)
                        ]["count"],
                        "cofitok_prefix_budget": cofitok_budget,
                        "dense_prefix_budget": dense_budget,
                        "cofitok_sample_set_sha256": cofitok["sample_sets"][
                            str(cofitok_budget)
                        ]["sha256"],
                        "dense_sample_set_sha256": dense["sample_sets"][
                            str(dense_budget)
                        ]["sha256"],
                    },
                    "sources": {
                        "cofitok": file_identity(cofitok_class_source),
                        "dense_identity": file_identity(dense_class_source),
                    },
                },
            },
        },
    )


def _matched_sources(root: Path) -> tuple[Path, Path, Path, Path]:
    cofitok_dir = root / "cofitok"
    dense_dir = root / "dense"
    for offset, global_index in enumerate(range(5, 10)):
        _image(cofitok_dir / f"{global_index:06d}.png", 0.35 + offset * 0.02)
        _image(dense_dir / f"{global_index:06d}.png", 0.55 + offset * 0.02)
    cofitok_report = root / "cofitok_sampling.json"
    dense_report = root / "dense_sampling.json"
    _sampling_report(cofitok_report, cofitok_dir, budget=8)
    _sampling_report(dense_report, dense_dir, budget=1)
    return cofitok_report, dense_report, cofitok_dir, dense_dir


def test_parse_indices_requires_increasing_unique_global_indices() -> None:
    assert parse_indices("5, 7,9") == [5, 7, 9]
    with pytest.raises(ValueError, match="unique"):
        parse_indices("5,5")
    with pytest.raises(ValueError, match="increasing"):
        parse_indices("7,5")


def test_builds_and_replays_source_bound_real_cofitok_dense_panels(
    tmp_path: Path,
) -> None:
    order = [f"n{index:08d}" for index in range(5)]
    calibration, real_dir = _calibration(tmp_path, order)
    cofitok_report, dense_report, cofitok_dir, dense_dir = _matched_sources(tmp_path)
    quality_result = tmp_path / "quality_result.json"
    _quality_result(quality_result, cofitok_report, dense_report)
    output = tmp_path / "visual_audit"
    report = build_requested_class_visual_audit(
        cofitok_sampling_report=cofitok_report,
        dense_sampling_report=dense_report,
        cofitok_dir=cofitok_dir,
        dense_dir=dense_dir,
        quality_result=quality_result,
        classifier_calibration_report=calibration,
        real_dir=real_dir,
        indices=[5, 6, 7, 8, 9],
        panel_columns=3,
        output_dir=output,
    )

    assert report["status"] == "completed"
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert [row["requested_class_index"] for row in report["selected_files"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert len(report["panels"]) == 2
    assert report["panels"][0]["row_order"] == [
        "real_validation",
        "cofitok",
        "dense_identity",
    ]
    assert (output / REPORT_FILENAME).is_file()

    replay = build_requested_class_visual_audit(
        cofitok_sampling_report=cofitok_report,
        dense_sampling_report=dense_report,
        cofitok_dir=cofitok_dir,
        dense_dir=dense_dir,
        quality_result=quality_result,
        classifier_calibration_report=calibration,
        real_dir=real_dir,
        indices=[5, 6, 7, 8, 9],
        panel_columns=3,
        output_dir=output,
        resume=True,
    )
    assert replay == report


def test_rejects_protocol_drift_between_matched_sources(tmp_path: Path) -> None:
    order = [f"n{index:08d}" for index in range(5)]
    calibration, real_dir = _calibration(tmp_path, order)
    cofitok_report, dense_report, cofitok_dir, dense_dir = _matched_sources(tmp_path)
    _sampling_report(dense_report, dense_dir, budget=1, guidance_scale=2.0)
    quality_result = tmp_path / "quality_result.json"
    _quality_result(quality_result, cofitok_report, dense_report)

    with pytest.raises(ValueError, match="different sampling protocols"):
        build_requested_class_visual_audit(
            cofitok_sampling_report=cofitok_report,
            dense_sampling_report=dense_report,
            cofitok_dir=cofitok_dir,
            dense_dir=dense_dir,
            quality_result=quality_result,
            classifier_calibration_report=calibration,
            real_dir=real_dir,
            indices=[5],
            panel_columns=1,
            output_dir=tmp_path / "visual_audit",
        )


def test_rejects_quality_result_class_count_drift(tmp_path: Path) -> None:
    order = [f"n{index:08d}" for index in range(5)]
    calibration, real_dir = _calibration(tmp_path, order)
    cofitok_report, dense_report, cofitok_dir, dense_dir = _matched_sources(tmp_path)
    quality_result = tmp_path / "quality_result.json"
    _quality_result(quality_result, cofitok_report, dense_report, num_classes=6)

    with pytest.raises(ValueError, match="calibration and sampling class counts differ"):
        build_requested_class_visual_audit(
            cofitok_sampling_report=cofitok_report,
            dense_sampling_report=dense_report,
            cofitok_dir=cofitok_dir,
            dense_dir=dense_dir,
            quality_result=quality_result,
            classifier_calibration_report=calibration,
            real_dir=real_dir,
            indices=[5],
            panel_columns=1,
            output_dir=tmp_path / "visual_audit",
        )

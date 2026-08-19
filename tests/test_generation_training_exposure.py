from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.training_exposure import (
    compare_training_exposures,
    training_exposure_summary,
)
from scripts.build_generation_training_exposure_audit import (
    build_report,
    parse_training_spec,
)
from scripts import build_generation_training_exposure_audit as exposure_audit


def _provenance(dataset: str) -> dict:
    spec = FORMAL_GENERATION_DATASETS[dataset]
    report = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": dataset,
        "dataset_root": f"/datasets/{dataset}",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": spec.manifest_bytes,
            "sha256": spec.manifest_sha256,
        },
        "splits": {"train": spec.train_images, "val": spec.val_images},
        "issues": [],
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def _training(
    *,
    dataset: str = "imagenet_256",
    completed_steps: int = 50_000,
    target_steps: int = 100_000,
) -> dict:
    effective_batch = 64
    report = {
        "target_steps": target_steps,
        "completed_steps": completed_steps,
        "training_complete": completed_steps == target_steps,
        "parameter_count": 62_834_083,
        "git": {
            "revision": "b" * 40,
            "branch": "scale/test",
            "dirty": False,
        },
        "dataset_provenance": _provenance(dataset),
        "config": {
            "data": {"dataset": dataset, "batch_size": 16},
            "optimization": {"gradient_accumulation_steps": 4},
            "runtime": {"device": "cuda"},
        },
        "final_metrics": {
            "step": completed_steps,
            "samples_seen": completed_steps * effective_batch,
        },
    }
    report["latest_checkpoint"] = {
        "step": completed_steps,
        "checkpoint_sha256": "e" * 64,
    }
    return report


def _identity(name: str) -> dict:
    return {"path": f"/evidence/{name}.json", "bytes": 123, "sha256": "c" * 64}


def _milestone_sources(tmp_path: Path) -> dict[str, dict]:
    sources = {}
    for name in (
        "cofitok_generation",
        "dense_generation",
        "cofitok_checkpoint_eval",
        "dense_checkpoint_eval",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "git": {
                        "revision": "b" * 40,
                        "branch": "scale/test",
                        "tracked_dirty": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        sources[name] = {
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": "c" * 64,
        }
    return sources


def test_partial_full_data_exposure_is_dataset_normalized() -> None:
    row = training_exposure_summary(_training())

    assert row["status"] == "partial"
    assert row["samples_seen"] == 3_200_000
    assert row["completed_fraction"] == 0.5
    assert row["completed_equivalent_epochs"] == pytest.approx(
        3_200_000 / 1_281_167
    )
    assert row["target_equivalent_epochs"] == pytest.approx(
        6_400_000 / 1_281_167
    )


def test_complete_10pct_exposure_is_about_ten_times_larger_per_image() -> None:
    full = training_exposure_summary(_training())
    subset = training_exposure_summary(
        _training(
            dataset="imagenet_256_10pct",
            completed_steps=50_000,
            target_steps=50_000,
        ),
        require_complete=True,
    )

    assert subset["status"] == "complete"
    assert subset["samples_seen"] == full["samples_seen"]
    assert subset["completed_equivalent_epochs"] == pytest.approx(
        3_200_000 / 128_161
    )
    assert subset["completed_equivalent_epochs"] > 9.9 * full[
        "completed_equivalent_epochs"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["final_metrics"].update(samples_seen=1), "samples_seen"),
        (lambda report: report.update(training_complete=True), "training_complete"),
        (
            lambda report: report["dataset_provenance"]["splits"].update(train=1),
            "dataset split",
        ),
    ],
)
def test_exposure_rejects_inconsistent_training_evidence(mutation, message: str) -> None:
    report = _training()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        training_exposure_summary(report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["config"]["data"].update(batch_size=16.5),
            "micro batch size",
        ),
        (lambda report: report.update(completed_steps=50_000.5), "completed steps"),
        (
            lambda report: report["final_metrics"].update(samples_seen=3_200_000.5),
            "samples seen",
        ),
    ],
)
def test_exposure_rejects_non_integer_counts(mutation, message: str) -> None:
    report = _training()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        training_exposure_summary(report)


def test_exposure_comparison_separates_steps_images_and_epochs() -> None:
    full = training_exposure_summary(_training())
    subset = training_exposure_summary(
        _training(
            dataset="imagenet_256_10pct",
            completed_steps=50_000,
            target_steps=50_000,
        )
    )
    comparison = compare_training_exposures({"full": full, "subset": subset})

    assert comparison["same_dataset"] is False
    assert comparison["same_dataset_identity"] is False
    assert comparison["same_completed_steps"] is True
    assert comparison["same_images_seen"] is True
    assert comparison["same_dataset_normalized_exposure"] is False
    assert comparison["quality_metric_comparison_allowed"] is False


def test_exposure_comparison_requires_exact_dataset_identity_and_batch() -> None:
    first = training_exposure_summary(_training())
    different_identity = copy.deepcopy(first)
    different_identity["dataset_identity_sha256"] = "d" * 64
    identity_comparison = compare_training_exposures(
        {"first": first, "different_identity": different_identity}
    )

    assert identity_comparison["same_dataset"] is True
    assert identity_comparison["same_dataset_identity"] is False
    assert identity_comparison["step_budget_directly_comparable"] is False
    assert identity_comparison["image_budget_directly_comparable"] is False

    different_batch = copy.deepcopy(first)
    different_batch["effective_batch_size"] = 32
    batch_comparison = compare_training_exposures(
        {"first": first, "different_batch": different_batch}
    )

    assert batch_comparison["same_dataset_identity"] is True
    assert batch_comparison["same_effective_batch_size"] is False
    assert batch_comparison["step_budget_directly_comparable"] is False


def test_source_bound_report_preserves_claim_boundary() -> None:
    training = _training()
    report = build_report(
        {
            "full_cofitok_50k": (training, _identity("full")),
            "full_dense_50k": (copy.deepcopy(training), _identity("dense")),
        }
    )

    assert report["status"] == "pass"
    assert report["comparison"]["step_budget_directly_comparable"] is True
    assert report["comparison"]["dataset_normalized_budget_directly_comparable"] is True
    assert report["claim_boundary"]["sample_quality_claim_allowed"] is False
    assert report["claim_boundary"]["method_quality_ranking_allowed"] is False


def test_source_bound_report_surfaces_validated_quality_bridge_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "source": {"equivalent_epochs": 3_200_000 / 128_161},
        "bridge": {"equivalent_epochs": 6_400_000 / 1_281_167},
    }
    monkeypatch.setattr(
        exposure_audit,
        "validate_quality_bridge_preparation",
        lambda preparation: {"training_exposure": expected},
    )
    report = build_report(
        {"full_cofitok_50k": (_training(), _identity("full"))},
        quality_bridge_preparation=(
            {"status": "prepared"},
            _identity("quality_bridge_preparation"),
        ),
    )

    exposure = report["quality_bridge_plan"]["training_exposure"]
    assert exposure["source"]["equivalent_epochs"] == pytest.approx(
        3_200_000 / 128_161
    )
    assert exposure["bridge"]["equivalent_epochs"] == pytest.approx(
        6_400_000 / 1_281_167
    )


def test_source_bound_report_binds_exact_matched_milestone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cofitok = _training()
    dense = copy.deepcopy(cofitok)
    milestone = {
        "methods": {
            "cofitok": {
                "checkpoint_step": 50_000,
                "checkpoint_sha256": "e" * 64,
            },
            "dense_identity": {
                "checkpoint_step": 50_000,
                "checkpoint_sha256": "e" * 64,
            },
        }
    }
    monkeypatch.setattr(
        exposure_audit,
        "verify_milestone_source_reports",
        lambda report, source_profile: {
            "status": "verified",
            "source_profile": source_profile,
            "source_reports": _milestone_sources(tmp_path),
        },
    )
    monkeypatch.setattr(
        exposure_audit,
        "validate_milestone_report",
        lambda report, **kwargs: (
            {"status": "verified", "cofitok_fid": 100.0, "dense_fid": 110.0},
            [],
        ),
    )

    report = build_report(
        {
            "cofitok": (cofitok, _identity("cofitok")),
            "dense_identity": (dense, _identity("dense")),
        },
        milestone_report=(milestone, _identity("milestone")),
        expected_milestone_step=50_000,
        milestone_source_profile="quality_bridge",
    )

    assert report["milestone_binding"]["matched_training_exposure_verified"] is True
    assert report["milestone_binding"]["training_checkpoint_binding"]["cofitok"][
        "checkpoint_sha256"
    ] == "e" * 64
    assert report["claim_boundary"]["milestone_quality_diagnostic_allowed"] is True
    assert report["claim_boundary"]["formal_generation_claim_allowed"] is False


def test_milestone_binding_rejects_training_checkpoint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cofitok = _training()
    dense = copy.deepcopy(cofitok)
    milestone = {
        "methods": {
            "cofitok": {
                "checkpoint_step": 50_000,
                "checkpoint_sha256": "f" * 64,
            },
            "dense_identity": {
                "checkpoint_step": 50_000,
                "checkpoint_sha256": "e" * 64,
            },
        }
    }
    monkeypatch.setattr(
        exposure_audit,
        "verify_milestone_source_reports",
        lambda report, source_profile: {
            "status": "verified",
            "source_reports": _milestone_sources(tmp_path),
        },
    )
    monkeypatch.setattr(
        exposure_audit,
        "validate_milestone_report",
        lambda report, **kwargs: ({"status": "verified"}, []),
    )

    with pytest.raises(ValueError, match="training and milestone checkpoints differ"):
        build_report(
            {
                "cofitok": (cofitok, _identity("cofitok")),
                "dense_identity": (dense, _identity("dense")),
            },
            milestone_report=(milestone, _identity("milestone")),
            expected_milestone_step=50_000,
            milestone_source_profile="quality_bridge",
        )


def test_training_spec_requires_label_and_path() -> None:
    assert parse_training_spec("cofitok=report.json") == (
        "cofitok",
        Path("report.json"),
    )
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_training_spec("report.json")

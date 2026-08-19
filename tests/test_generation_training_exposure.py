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
        "output_dir": f"/runs/{dataset}",
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


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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


def _terminal_sources(
    *,
    cofitok: dict,
    dense: dict,
    cofitok_identity: dict,
    dense_identity: dict,
    preparation_identity: dict,
    milestone_identity: dict,
) -> dict[str, dict]:
    generic = _identity("terminal_source")
    sources = {
        name: copy.deepcopy(generic)
        for name in (
            "preparation",
            "launch_receipt",
            "cofitok_training",
            "dense_training",
            "training_pair_validation",
            "cofitok_training_audit",
            "dense_training_audit",
            "milestone_50000",
            "milestone_100000",
            "cofitok_sampling_preflight",
            "dense_sampling_preflight",
            "cofitok_generation",
            "dense_generation",
            "cofitok_checkpoint_eval",
            "dense_checkpoint_eval",
            "class_fidelity_qualification",
            "cofitok_class_fidelity",
            "dense_class_fidelity",
        )
    }
    sources["cofitok_training"] = {
        **cofitok_identity,
        "path": f"{cofitok['output_dir']}/training_report.json",
    }
    sources["dense_training"] = {
        **dense_identity,
        "path": f"{dense['output_dir']}/training_report.json",
    }
    sources["preparation"] = {
        **preparation_identity,
        "path": "/quality/reports/preparation.json",
    }
    sources["milestone_100000"] = {
        **milestone_identity,
        "path": "/quality/reports/milestones/step_00100000.json",
    }
    return sources


def _terminal_result(
    *,
    cofitok: dict,
    dense: dict,
    cofitok_identity: dict,
    dense_identity: dict,
    preparation_identity: dict,
    milestone_identity: dict,
) -> dict:
    git = {"revision": "b" * 40, "branch": "scale/test", "tracked_dirty": False}
    return {
        "schema_version": exposure_audit.QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
        "role": exposure_audit.QUALITY_BRIDGE_RESULT_ROLE,
        "status": "completed",
        "stage": exposure_audit.QUALITY_BRIDGE_RECIPE_STAGE,
        "git": git,
        "source_reports": _terminal_sources(
            cofitok=cofitok,
            dense=dense,
            cofitok_identity=cofitok_identity,
            dense_identity=dense_identity,
            preparation_identity=preparation_identity,
            milestone_identity=milestone_identity,
        ),
        "training": {
            "pair_validation": {
                "status": "pass",
                "expected_steps": 100_000,
                "expected_dataset": "imagenet_256",
                "expected_revision": "b" * 40,
                "expected_branch": "scale/test",
                "training_recipe": {
                    "stage": "stability_quality_bridge",
                    "valid": True,
                },
            }
        },
        "terminal": {
            "methods": {
                "cofitok": {
                    "checkpoint_step": 100_000,
                    "checkpoint_sha256": "e" * 64,
                },
                "dense_identity": {
                    "checkpoint_step": 100_000,
                    "checkpoint_sha256": "e" * 64,
                },
            }
        },
        "quality_screen": {"status": "hold"},
        "authorization_boundary": copy.deepcopy(
            exposure_audit.RESULT_AUTHORIZATION_BOUNDARY
        ),
    }


def _completed_execution_status() -> dict:
    return {
        "schema_version": 1,
        "role": "stability_full_data_quality_bridge_execution",
        "status": "completed",
        "detail": (
            "quality bridge terminal evidence verified; no larger-training "
            "authorization was created"
        ),
        "exit_code": None,
        "git": {
            "revision": "b" * 40,
            "branch": "scale/test",
            "tracked_dirty": False,
        },
        "quality_bridge_only": True,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "report_is_promotion_gate": False,
    }


def _terminal_build_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    cofitok = _training(completed_steps=100_000, target_steps=100_000)
    dense = copy.deepcopy(cofitok)
    cofitok["output_dir"] = "/runs/cofitok"
    dense["output_dir"] = "/runs/dense"
    cofitok_identity = _identity("snapshotted_cofitok")
    dense_identity = _identity("snapshotted_dense")
    preparation_identity = _identity("snapshotted_preparation")
    milestone_identity = _identity("snapshotted_milestone_100000")
    milestone = {
        "methods": {
            "cofitok": {
                "checkpoint_step": 100_000,
                "checkpoint_sha256": "e" * 64,
            },
            "dense_identity": {
                "checkpoint_step": 100_000,
                "checkpoint_sha256": "e" * 64,
            },
        }
    }
    monkeypatch.setattr(
        exposure_audit,
        "validate_quality_bridge_preparation",
        lambda preparation: {"training_exposure": {"status": "planned"}},
    )
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
        lambda report, **kwargs: ({"status": "verified"}, []),
    )
    monkeypatch.setattr(
        exposure_audit,
        "_replay_quality_bridge_terminal_result",
        lambda terminal: terminal,
    )
    terminal = _terminal_result(
        cofitok=cofitok,
        dense=dense,
        cofitok_identity=cofitok_identity,
        dense_identity=dense_identity,
        preparation_identity=preparation_identity,
        milestone_identity=milestone_identity,
    )
    return {
        "training_reports": {
            "cofitok": (cofitok, cofitok_identity),
            "dense_identity": (dense, dense_identity),
        },
        "quality_bridge_preparation": (
            {"status": "prepared"},
            preparation_identity,
        ),
        "milestone_report": (milestone, milestone_identity),
        "expected_milestone_step": 100_000,
        "milestone_source_profile": "quality_bridge",
        "quality_bridge_terminal_result": (
            terminal,
            _identity("snapshotted_terminal_result"),
        ),
        "quality_bridge_execution_status": (
            _completed_execution_status(),
            _identity("snapshotted_execution_status"),
        ),
    }


def test_partial_full_data_exposure_is_dataset_normalized() -> None:
    row = training_exposure_summary(_training())

    assert row["status"] == "partial"
    assert row["samples_seen"] == 3_200_000
    assert row["completed_fraction"] == 0.5
    assert row["completed_equivalent_epochs"] == pytest.approx(3_200_000 / 1_281_167)
    assert row["target_equivalent_epochs"] == pytest.approx(6_400_000 / 1_281_167)


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
    assert subset["completed_equivalent_epochs"] == pytest.approx(3_200_000 / 128_161)
    assert (
        subset["completed_equivalent_epochs"]
        > 9.9 * full["completed_equivalent_epochs"]
    )


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
def test_exposure_rejects_inconsistent_training_evidence(
    mutation, message: str
) -> None:
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
    assert exposure["source"]["equivalent_epochs"] == pytest.approx(3_200_000 / 128_161)
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
    assert (
        report["milestone_binding"]["training_checkpoint_binding"]["cofitok"][
            "checkpoint_sha256"
        ]
        == "e" * 64
    )
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


def test_terminal_binding_proves_exact_completed_matched_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _terminal_build_inputs(tmp_path, monkeypatch)

    report = build_report(**inputs)

    assert report["rows"]["cofitok"]["training_complete"] is True
    assert report["rows"]["dense_identity"]["completed_steps"] == 100_000
    assert report["comparison"]["same_images_seen"] is True
    assert report["comparison"]["same_dataset_normalized_exposure"] is True
    assert report["terminal_binding"]["terminal_result_binding_verified"] is True
    assert report["terminal_binding"]["active_runbook_verification_completed"] is True
    assert report["terminal_binding"]["checkpoint_binding"]["cofitok"] == {
        "step": 100_000,
        "checkpoint_sha256": "e" * 64,
    }
    assert report["terminal_binding"]["quality_screen"]["status"] == "hold"
    assert (
        report["claim_boundary"]["terminal_training_exposure_binding_allowed"] is True
    )
    assert report["claim_boundary"]["formal_generation_claim_allowed"] is False


def test_terminal_binding_rejects_another_training_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _terminal_build_inputs(tmp_path, monkeypatch)
    terminal = inputs["quality_bridge_terminal_result"][0]
    terminal["source_reports"]["dense_training"]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="another training report"):
        build_report(**inputs)


def test_terminal_replay_reopens_every_bound_json_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_names = (
        "preparation",
        "launch_receipt",
        "cofitok_training",
        "dense_training",
        "training_pair_validation",
        "cofitok_training_audit",
        "dense_training_audit",
        "milestone_50000",
        "milestone_100000",
        "cofitok_sampling_preflight",
        "dense_sampling_preflight",
        "cofitok_generation",
        "dense_generation",
        "cofitok_checkpoint_eval",
        "dense_checkpoint_eval",
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
    )
    sources = {}
    for name in source_names:
        path = tmp_path / f"{name}.json"
        _write(path, {"source": name})
        sources[name] = exposure_audit._source_identity(path)
    observed = {}

    def _fake_build(**kwargs):
        observed.update(kwargs)
        return {"status": "replayed"}

    monkeypatch.setattr(exposure_audit, "build_quality_bridge_result", _fake_build)
    terminal = {
        "git": {
            "revision": "b" * 40,
            "branch": "scale/test",
            "tracked_dirty": False,
        },
        "source_reports": sources,
        "milestones": {"50000": {"step": 50_000}, "100000": {"step": 100_000}},
        "terminal": {"physical_evidence": {"cofitok": {}, "dense_identity": {}}},
    }

    replayed = exposure_audit._replay_quality_bridge_terminal_result(terminal)

    assert replayed == {"status": "replayed"}
    assert observed["preparation"] == {"source": "preparation"}
    assert observed["dense_generation"] == {"source": "dense_generation"}
    assert observed["milestone_evidence"] == {
        50_000: {"step": 50_000},
        100_000: {"step": 100_000},
    }
    assert observed["expected_revision"] == "b" * 40


def test_terminal_replay_rejects_changed_bound_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preparation.json"
    _write(path, {"source": "preparation"})
    identity = exposure_audit._source_identity(path)
    _write(path, {"source": "changed"})
    terminal = {
        "git": {
            "revision": "b" * 40,
            "branch": "scale/test",
            "tracked_dirty": False,
        },
        "source_reports": {"preparation": identity},
        "milestones": {"50000": {}, "100000": {}},
        "terminal": {"physical_evidence": {}},
    }

    with pytest.raises(ValueError, match="terminal source changed: preparation"):
        exposure_audit._replay_quality_bridge_terminal_result(terminal)


def test_training_spec_requires_label_and_path() -> None:
    assert parse_training_spec("cofitok=report.json") == (
        "cofitok",
        Path("report.json"),
    )
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_training_spec("report.json")

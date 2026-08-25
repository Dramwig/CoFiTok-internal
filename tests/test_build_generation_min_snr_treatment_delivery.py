from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import build_generation_min_snr_treatment_delivery as delivery


TRAINING_REVISION = "a" * 40
TRAINING_TREE = "f" * 40
TRAINING_BRANCH = "scale/generation-min-snr-matched-pilot-v1-20260825"
IDENTITY = "b" * 64
RUNTIME = "c" * 64


def _manifest(*, method: str = "cofitok", gamma: float = 5.0) -> dict:
    dense = method == "dense_identity"
    return {
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "dirty": False,
        },
        "parameter_count": 999 if dense else 1_000,
        "runtime_environment_sha256": RUNTIME,
        "dataset_provenance": {
            "status": "pass",
            "formal": True,
            "dataset": "imagenet_256",
            "identity_sha256": IDENTITY,
        },
        "config": {
            "data": {"batch_size": 4, "dataset": "imagenet_256"},
            "diffusion": {"prediction_target": "epsilon"},
            "loss": {"min_snr_gamma": gamma},
            "model": {
                "token_count": 1 if dense else 8,
                "synthesis_mode": "dense_identity" if dense else "fixed_basis",
            },
            "optimization": {
                "gradient_accumulation_steps": 2,
                "log_interval": 50,
            },
            "runtime": {"steps": 100_000, "seed": 2027},
        },
    }


def _rows() -> list[dict]:
    rows = []
    for index, step in enumerate((1, 50, 100, 150, 200), start=1):
        unweighted = 1.0 / index
        rows.append(
            {
                "step": step,
                "samples_seen": step * 8,
                "total": unweighted,
                "epsilon": unweighted * 0.8,
                "epsilon_unweighted": unweighted,
                "min_snr_weight_mean": 0.85,
                "learning_rate": 1e-4,
                "grad_norm": 1.0,
                "cumulative_elapsed_seconds": float(index * 10),
            }
        )
    return rows


def _report(*, rows: list[dict] | None = None, manifest: dict | None = None) -> dict:
    return delivery.build_report(
        rows=_rows() if rows is None else rows,
        manifest=_manifest() if manifest is None else manifest,
        method="cofitok",
        cutoff_step=200,
        expected_gamma=5.0,
        expected_effective_batch=8,
        expected_scheduler_horizon=100_000,
        expected_seed=2027,
        expected_dataset="imagenet_256",
        expected_training_revision=TRAINING_REVISION,
        expected_training_tree=TRAINING_TREE,
        expected_training_branch=TRAINING_BRANCH,
        builder_identity={
            "revision": "d" * 40,
            "tree": "e" * 40,
            "branch": "analysis/test",
            "tracked_dirty": False,
            "training_revision_is_ancestor": True,
            "training_tree_verified": True,
            "training_semantic_sources_unchanged": True,
        },
    )


def test_report_verifies_delivery_without_claiming_quality_or_advantage() -> None:
    report = _report()

    assert report["status"] == "pass"
    assert report["trajectory"]["samples_seen"] == 1_600
    assert report["trajectory"]["downweighted_weight_row_count"] == 5
    assert report["trajectory"]["downweighted_epsilon_row_count"] == 5
    assert report["scientific_interpretation"]["treatment_delivery_verified"] is True
    assert report["scientific_interpretation"]["matched_method_comparison_present"] is False
    assert report["scientific_interpretation"]["generation_advantage_proven"] is False
    assert report["claim_boundary"]["quality_claim_allowed"] is False
    assert report["claim_boundary"]["formal_50k_result_substitute"] is False
    assert all(value is False for value in report["authorization_boundary"].values())


def test_report_rejects_gamma_or_method_identity_drift() -> None:
    with pytest.raises(ValueError, match="gamma differs"):
        _report(manifest=_manifest(gamma=0.0))

    dense = _manifest(method="dense_identity")
    with pytest.raises(ValueError, match="CoFiTok arm"):
        _report(manifest=dense)


def test_report_rejects_exposure_schedule_or_nonfinite_drift() -> None:
    rows = _rows()
    rows[-1]["samples_seen"] -= 1
    with pytest.raises(ValueError, match="samples_seen differs"):
        _report(rows=rows)

    rows = _rows()
    rows.pop(2)
    with pytest.raises(ValueError, match="exact logging schedule"):
        _report(rows=rows)

    rows = _rows()
    rows[-1]["epsilon"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        _report(rows=rows)


def test_report_rejects_absent_or_invalid_downweighting() -> None:
    rows = _rows()
    for row in rows:
        row["epsilon"] = row["epsilon_unweighted"]
        row["min_snr_weight_mean"] = 1.0
    with pytest.raises(ValueError, match="do not demonstrate active"):
        _report(rows=rows)

    rows = _rows()
    rows[-1]["epsilon"] = rows[-1]["epsilon_unweighted"] + 0.1
    with pytest.raises(ValueError, match="exceeds unweighted"):
        _report(rows=rows)


def test_report_rejects_dirty_or_wrong_training_identity() -> None:
    manifest = deepcopy(_manifest())
    manifest["git"]["dirty"] = True
    with pytest.raises(ValueError, match="Git identity differs"):
        _report(manifest=manifest)

    manifest = deepcopy(_manifest())
    manifest["runtime_environment_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="runtime identity"):
        _report(manifest=manifest)


def test_metrics_prefix_identity_is_stable_after_later_rows(tmp_path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    rows = _rows()
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    before = delivery._read_metrics_prefix(path, cutoff_step=200)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": 250,
                    "samples_seen": 2_000,
                    "total": 0.1,
                    "epsilon": 0.08,
                    "epsilon_unweighted": 0.1,
                    "min_snr_weight_mean": 0.85,
                    "learning_rate": 1e-4,
                    "grad_norm": 1.0,
                    "cumulative_elapsed_seconds": 60.0,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    after = delivery._read_metrics_prefix(path, cutoff_step=200)

    assert before["identity"] == after["identity"]
    assert before["observed_file"]["sha256"] != after["observed_file"]["sha256"]

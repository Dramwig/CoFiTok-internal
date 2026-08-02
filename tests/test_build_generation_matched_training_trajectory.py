from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import build_generation_matched_training_trajectory as trajectory


REVISION = "a" * 40
BRANCH = "scale/generation-stability-50k-preflight"
IDENTITY = "b" * 64
RUNTIME = "c" * 64


def _config(*, dense: bool) -> dict:
    return {
        "data": {"batch_size": 4, "dataset": "imagenet_256_10pct"},
        "diffusion": {"prediction_target": "epsilon"},
        "optimization": {
            "gradient_accumulation_steps": 2,
            "log_interval": 50,
        },
        "runtime": {"evaluation_interval": 100, "steps": 500},
        "model": {
            "base_channels": 32,
            "token_count": 1 if dense else 8,
            "token_channels": 3 if dense else 8,
            "predictor_use_feedback": not dense,
            "synthesis_mode": "dense_identity" if dense else "fixed_basis",
        },
        "loss": {
            "epsilon_weight": 1.0,
            "rollout_consistency_weight": 0.1,
            "rollout_consistency_start_step": 0,
            "rollout_consistency_warmup_steps": 200,
            "rollout_consistency_timestep_delta": 10,
            "rollout_consistency_unroll_steps": 2,
            "rollout_consistency_batch_fraction": 0.125,
            "rollout_consistency_clip_x0": True,
            "rollout_consistency_mode": "clipped_x0",
            "ema_teacher_consistency_weight": 0.25,
            "ema_teacher_consistency_start_step": 100,
            "ema_teacher_consistency_warmup_steps": 100,
            "ema_teacher_consistency_batch_fraction": 0.0625,
            "prefix_weight": 0.0,
        },
    }


def _manifest(*, dense: bool) -> dict:
    return {
        "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
        "parameter_count": 999 if dense else 1_000,
        "runtime_environment_sha256": RUNTIME,
        "dataset_provenance": {
            "status": "pass",
            "formal": True,
            "dataset": "imagenet_256_10pct",
            "identity_sha256": IDENTITY,
        },
        "config": _config(dense=dense),
    }


def _rows(*, dense: bool) -> list[dict]:
    result = []
    for step in (1, 50, 100, 150, 200):
        row = {
            "step": step,
            "samples_seen": step * 8,
            "epsilon": 0.5 / step + (0.001 if dense else 0.0),
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": min(step / 200, 1.0),
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": (
                0.0 if step < 100 else min((step - 100) / 100, 1.0)
            ),
        }
        if step in (100, 200):
            event = step // 100 - 1
            row.update(
                {
                    "validation_epsilon_mse": (
                        0.03 + event * 0.001 + (0.0002 if dense else 0.0)
                    ),
                    "validation_event_index": event,
                    "validation_batch_index": event,
                    "validation_num_images": 64,
                    "validation_noise_seed": 102030,
                }
            )
        result.append(row)
    return result


def _metrics(*, dense: bool) -> dict:
    return {"rows": _rows(dense=dense)}


def test_report_binds_matched_validation_without_claiming_quality() -> None:
    report = trajectory.build_report(
        cofitok_metrics=_metrics(dense=False),
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=_manifest(dense=False),
        dense_manifest=_manifest(dense=True),
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )

    assert report["status"] == "pass"
    assert report["images_seen_per_method"] == 1_600
    assert report["contract"]["generation_pair_contract"]["valid"] is True
    paired = report["paired_fixed_validation"]
    assert paired["summary"]["event_count"] == 2
    assert paired["summary"]["cofitok_lower_event_count"] == 2
    assert paired["events"][0]["validation_noise_seed"] == 102030
    regimes = paired["schedule_regimes"]
    assert regimes["contracts"]["rollout_consistency"]["full_scale_step"] == 200
    rollout = regimes["by_schedule"]["rollout_consistency"]
    assert [entry["phase"] for entry in rollout] == ["warmup", "full_scale"]
    assert rollout[0]["event_steps"] == [100]
    assert rollout[1]["event_steps"] == [200]
    ema_teacher = regimes["by_schedule"]["ema_teacher_consistency"]
    assert [entry["phase"] for entry in ema_teacher] == ["warmup", "full_scale"]
    assert regimes["individual_regime_significance_claim_allowed"] is False
    assert report["comparison_policy"]["total_loss_comparison_allowed"] is False
    assert report["comparison_policy"]["training_wall_clock_comparison_allowed"] is False
    assert report["claim_boundary"]["quality_claim_allowed"] is False
    assert report["claim_boundary"]["formal_50k_gate_substitute"] is False
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["schedule_regime_quality_claim_allowed"] is False


def test_report_rejects_sample_accounting_drift() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["samples_seen"] -= 1

    with pytest.raises(ValueError, match="samples_seen is invalid"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_validation_provenance_drift() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["validation_noise_seed"] += 1

    with pytest.raises(ValueError, match="validation provenance differs"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_observed_schedule_scale_drift() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["rollout_consistency_scale"] = 0.9

    with pytest.raises(ValueError, match="full-scale phase"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


@pytest.mark.parametrize(
    ("step", "contract", "expected"),
    [
        (50, {"enabled": False, "start_step": 0, "warmup_steps": 100}, "disabled"),
        (50, {"enabled": True, "start_step": 100, "warmup_steps": 100}, "inactive"),
        (100, {"enabled": True, "start_step": 100, "warmup_steps": 100}, "warmup"),
        (200, {"enabled": True, "start_step": 100, "warmup_steps": 100}, "full_scale"),
    ],
)
def test_schedule_phase_boundaries(
    step: int,
    contract: dict,
    expected: str,
) -> None:
    assert trajectory._schedule_phase(step, contract) == expected


def test_report_rejects_runtime_or_pair_contract_drift() -> None:
    dense_manifest = _manifest(dense=True)
    dense_manifest["runtime_environment_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="runtime_environment_sha256"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )

    dense_manifest = _manifest(dense=True)
    dense_manifest["config"]["data"]["batch_size"] = 8
    with pytest.raises(ValueError, match="training pair contract failed"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_metrics_prefix_identity_is_stable_after_later_rows(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = _rows(dense=False)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    before = trajectory._read_metrics_prefix(path, cutoff_step=200)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": 250,
                    "samples_seen": 2_000,
                    "epsilon": 0.01,
                    "rollout_consistency": 0.1,
                    "rollout_consistency_scale": 1.0,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    after = trajectory._read_metrics_prefix(path, cutoff_step=200)

    assert before["identity"] == after["identity"]
    assert before["observed_file"]["sha256"] != after["observed_file"]["sha256"]


def test_report_rejects_nonfinite_shared_metrics() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["epsilon"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_dirty_or_wrong_git_identity() -> None:
    dense_manifest = deepcopy(_manifest(dense=True))
    dense_manifest["git"]["dirty"] = True

    with pytest.raises(ValueError, match="dirty tracked worktree"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_nonformal_dataset_provenance() -> None:
    dense_manifest = deepcopy(_manifest(dense=True))
    dense_manifest["dataset_provenance"]["formal"] = False

    with pytest.raises(ValueError, match="lacks passing dataset provenance"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )

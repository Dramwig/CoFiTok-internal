from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.build_generation_runtime_compute_claim_guard import build_guard
from scripts.run_generation_quality_bridge_runtime_claim_guard_waiter import (
    SOURCE_SCOPE,
    run_waiter,
    validate_source_deployment,
    validate_source_status,
)


REVISION = "a" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
MONITOR = "generation_stability_full_data_quality_bridge_100k"
STEPS = 100_000
EFFECTIVE_BATCH = 64
ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _cost(*, elapsed: float, peak_vram: int, adjustment: float = 0.0) -> dict:
    samples = STEPS * EFFECTIVE_BATCH
    applied = adjustment > 0.0
    return {
        "valid": True,
        "target_steps": STEPS,
        "micro_batch_size": EFFECTIVE_BATCH,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": EFFECTIVE_BATCH,
        "expected_samples_seen": samples,
        "samples_seen": samples,
        "reported_elapsed_seconds": elapsed - adjustment,
        "resume_compute_adjustment": {
            "valid": True,
            "required": applied,
            "provided": applied,
            "applied": applied,
            "seconds": adjustment,
            "hours": adjustment / 3600.0,
            "event_count": 1 if applied else 0,
            "orphaned_optimizer_steps_lower_bound": 200 if applied else 0,
            "orphaned_images_lower_bound": 12_800 if applied else 0,
            "continuity_end_step": STEPS,
            "discovered_orphan_archive_count": 1 if applied else 0,
            "covered_orphan_archive_count": 1 if applied else 0,
            "required_reasons": ["orphaned_metrics_archives"] if applied else [],
            "issues": [],
        },
        "elapsed_seconds": elapsed,
        "elapsed_seconds_role": (
            "physical_lower_bound_including_orphaned_recovery_compute"
            if applied
            else "reported_training_elapsed_seconds"
        ),
        "images_per_second": samples / elapsed,
        "peak_vram_bytes": peak_vram,
    }


def _fairness_report(
    tmp_path: Path,
    *,
    cofitok_adjustment: float = 22.0,
) -> dict:
    output_root = tmp_path / "quality"
    cofitok_run = output_root / "cofitok"
    dense_run = output_root / "dense"
    cofitok = _cost(
        elapsed=10_000.0 + cofitok_adjustment,
        peak_vram=80_000,
        adjustment=cofitok_adjustment,
    )
    dense = _cost(elapsed=9_000.0, peak_vram=70_000)
    return {
        "schema_version": 1,
        "role": "quality_bridge_runtime_compute_fairness_audit",
        "status": "pass",
        "scope": {
            "cpu_only": True,
            "checkpoint_payload_loading_allowed": False,
            "gpu_execution_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "auditor": {},
        "contract": {
            "status": "verified",
            "sources": {},
            "training_git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "dataset": "imagenet_256",
            "output_root": output_root.resolve().as_posix(),
            "training_run_dirs": [
                cofitok_run.resolve().as_posix(),
                dense_run.resolve().as_posix(),
            ],
            "target_steps_per_method": STEPS,
            "runtime": {
                "micro_batch_size": EFFECTIVE_BATCH,
                "gradient_accumulation_steps": 1,
                "effective_batch_size": EFFECTIVE_BATCH,
                "runtime_environment_sha256": "b" * 64,
            },
            "parameters": {
                "cofitok": 62_834_083,
                "dense_identity": 62_824_707,
                "relative_gap": (62_834_083 - 62_824_707) / 62_824_707,
            },
            "runbook_contract": {},
        },
        "terminal_sources": {},
        "pair_validation": {},
        "observed_runtime_parity": {
            "status": "proven",
            "micro_batch_size": EFFECTIVE_BATCH,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": EFFECTIVE_BATCH,
            "canonical_images_per_method": STEPS * EFFECTIVE_BATCH,
        },
        "methods": {
            "cofitok": {
                "status": "verified",
                "training_cost": cofitok,
                "resume_compute_adjustment": {},
                "resume_compute_adjustment_verification": {},
            },
            "dense_identity": {
                "status": "verified",
                "training_cost": dense,
                "resume_compute_adjustment": None,
                "resume_compute_adjustment_verification": None,
            },
        },
        "descriptive_comparison": {
            "role": "measured_outcomes_not_predeclared_advantage",
            "cofitok_minus_dense_adjusted_elapsed_seconds": (
                cofitok["elapsed_seconds"] - dense["elapsed_seconds"]
            ),
            "cofitok_adjusted_elapsed_relative_change": (
                cofitok["elapsed_seconds"] / dense["elapsed_seconds"] - 1.0
            ),
            "cofitok_throughput_relative_change": (
                cofitok["images_per_second"] / dense["images_per_second"] - 1.0
            ),
            "cofitok_peak_vram_relative_change": 80_000 / 70_000 - 1.0,
        },
        "claim_boundary": {
            "matched_runtime_contract_established": True,
            "observed_pair_runtime_parity_established": True,
            "physical_recovery_compute_included": True,
            "training_speed_advantage_predeclared": False,
            "memory_advantage_predeclared": False,
            "sample_quality_established": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def _pair_report(fairness: dict, *, direct: bool) -> dict:
    contract = fairness["contract"]
    maximum_gap = 300.0 if direct else 16_377.985275
    coverage = {
        "started_before_training": True,
        "saw_training_active": True,
        "completed_after_training": True,
        "all_gpu_queries_complete": True,
        "maximum_gap_seconds": maximum_gap,
        "continuous": direct,
        "complete": direct,
    }
    reason = (
        "exclusive_gpu_observation_coverage"
        if direct
        else "incomplete_gpu_observation_coverage"
    )
    return {
        "schema_version": 2,
        "monitor": MONITOR,
        "status": "pass",
        "stage": "complete",
        "git": copy.deepcopy(contract["training_git"]),
        "runs": {
            "cofitok": {
                "run_dir": contract["training_run_dirs"][0],
                "last_step": STEPS,
                "complete": True,
            },
            "dense_identity": {
                "run_dir": contract["training_run_dirs"][1],
                "last_step": STEPS,
                "complete": True,
            },
        },
        "gpu_contention": {
            "schema_version": 1,
            "role": "generation_gpu_contention_evidence",
            "status": "pass_exclusive" if direct else "pass_observational_only",
            "binding": {
                "monitor_name": MONITOR,
                "training_revision": REVISION,
                "training_branch": BRANCH,
            },
            "first_observed_at": "2026-08-16T00:00:00+00:00",
            "last_observed_at": "2026-08-18T00:00:00+00:00",
            "observation_count": 10,
            "poll_seconds": 300.0,
            "coverage": coverage,
            "current": {},
            "unrelated_gpu_compute": {
                "observed": False,
                "observation_count": 0,
                "identity_overflow": False,
                "identities": [],
            },
            "training_wall_clock": {
                "measurement": "raw_process_wall_clock",
                "direct_comparison_allowed": direct,
                "reason": reason,
            },
        },
    }


def _build(
    tmp_path: Path,
    *,
    direct: bool,
    cofitok_adjustment: float = 22.0,
) -> dict:
    fairness = _fairness_report(
        tmp_path,
        cofitok_adjustment=cofitok_adjustment,
    )
    pair = _pair_report(fairness, direct=direct)
    fairness_path = tmp_path / "fairness.json"
    pair_path = tmp_path / "pair.json"
    fairness_identity = _write(fairness_path, fairness)
    pair_identity = _write(pair_path, pair)
    return build_guard(
        runtime_fairness_report_path=fairness_path,
        expected_runtime_fairness_sha256=fairness_identity["sha256"],
        pair_monitor_path=pair_path,
        expected_pair_monitor_sha256=pair_identity["sha256"],
        expected_monitor_name=MONITOR,
    )


def test_guard_marks_incomplete_gpu_coverage_observational_only(
    tmp_path: Path,
) -> None:
    report = _build(tmp_path, direct=False)

    assert report["status"] == "pass"
    assert report["decision"] == "runtime_cost_claims_observational_only"
    policy = report["claim_policy"]
    assert policy["runtime_configuration_parity_claim_allowed"] is True
    assert policy["training_wall_clock_direct_comparison_allowed"] is False
    assert policy["training_throughput_direct_comparison_allowed"] is False
    assert policy["cost_efficiency_ranking_allowed"] is False
    assert policy["observed_pair_runtime_parity_claim_allowed"] is False
    assert policy["physical_lower_bound_label_required"] is True
    assert report["metric_roles"]["adjusted_training_elapsed_and_throughput"][
        "role"
    ] == "observational_physical_lower_bounds_only"
    assert "physical lower bound" in report["claim_text"]


def test_guard_allows_descriptive_runtime_comparison_only_with_complete_coverage(
    tmp_path: Path,
) -> None:
    report = _build(tmp_path, direct=True, cofitok_adjustment=0.0)

    assert report["decision"] == "direct_runtime_outcome_comparison_allowed"
    policy = report["claim_policy"]
    assert policy["training_wall_clock_direct_comparison_allowed"] is True
    assert policy["training_throughput_direct_comparison_allowed"] is True
    assert policy["cost_efficiency_ranking_allowed"] is True
    assert policy["exclusive_gpu_observation_coverage_verified"] is True
    assert policy["recovery_adjusted_elapsed_exact_for_both_methods"] is True
    assert policy["physical_lower_bound_label_required"] is False
    assert policy["equal_wall_clock_budget_claim_allowed"] is False
    assert policy["training_speed_advantage_predeclared"] is False


def test_guard_keeps_recovery_lower_bound_observational_with_complete_coverage(
    tmp_path: Path,
) -> None:
    report = _build(tmp_path, direct=True, cofitok_adjustment=22.0)

    assert report["decision"] == "runtime_cost_claims_observational_only"
    policy = report["claim_policy"]
    assert policy["exclusive_gpu_observation_coverage_verified"] is True
    assert policy["recovery_adjusted_elapsed_exact_for_both_methods"] is False
    assert policy["training_wall_clock_direct_comparison_allowed"] is False
    assert policy["training_throughput_direct_comparison_allowed"] is False
    assert policy["cost_efficiency_ranking_allowed"] is False
    assert policy["physical_lower_bound_label_required"] is True
    assert "GPU observation coverage is complete" in report["claim_text"]
    assert "physical lower bound" in report["claim_text"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("required", False),
        ("provided", False),
        ("orphaned_images_lower_bound", 1),
        ("covered_orphan_archive_count", 0),
    ),
)
def test_guard_rejects_inconsistent_recovery_adjustment_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fairness = _fairness_report(tmp_path)
    pair = _pair_report(fairness, direct=False)
    fairness["methods"]["cofitok"]["training_cost"][
        "resume_compute_adjustment"
    ][field] = value
    fairness_path = tmp_path / "fairness.json"
    pair_path = tmp_path / "pair.json"
    fairness_identity = _write(fairness_path, fairness)
    pair_identity = _write(pair_path, pair)

    with pytest.raises(ValueError, match="recovery adjustment"):
        build_guard(
            runtime_fairness_report_path=fairness_path,
            expected_runtime_fairness_sha256=fairness_identity["sha256"],
            pair_monitor_path=pair_path,
            expected_pair_monitor_sha256=pair_identity["sha256"],
            expected_monitor_name=MONITOR,
        )


def test_guard_rejects_tampered_descriptive_runtime(tmp_path: Path) -> None:
    fairness = _fairness_report(tmp_path)
    pair = _pair_report(fairness, direct=False)
    fairness["descriptive_comparison"][
        "cofitok_adjusted_elapsed_relative_change"
    ] = -0.5
    fairness_path = tmp_path / "fairness.json"
    pair_path = tmp_path / "pair.json"
    fairness_identity = _write(fairness_path, fairness)
    pair_identity = _write(pair_path, pair)

    with pytest.raises(ValueError, match="descriptive field differs"):
        build_guard(
            runtime_fairness_report_path=fairness_path,
            expected_runtime_fairness_sha256=fairness_identity["sha256"],
            pair_monitor_path=pair_path,
            expected_pair_monitor_sha256=pair_identity["sha256"],
            expected_monitor_name=MONITOR,
        )


def test_guard_rejects_promoted_gpu_contention_decision(tmp_path: Path) -> None:
    fairness = _fairness_report(tmp_path)
    pair = _pair_report(fairness, direct=False)
    pair["gpu_contention"]["training_wall_clock"][
        "direct_comparison_allowed"
    ] = True
    fairness_path = tmp_path / "fairness.json"
    pair_path = tmp_path / "pair.json"
    fairness_identity = _write(fairness_path, fairness)
    pair_identity = _write(pair_path, pair)

    with pytest.raises(ValueError, match="decision is inconsistent"):
        build_guard(
            runtime_fairness_report_path=fairness_path,
            expected_runtime_fairness_sha256=fairness_identity["sha256"],
            pair_monitor_path=pair_path,
            expected_pair_monitor_sha256=pair_identity["sha256"],
            expected_monitor_name=MONITOR,
        )


def test_guard_rejects_pair_training_identity_drift(tmp_path: Path) -> None:
    fairness = _fairness_report(tmp_path)
    pair = _pair_report(fairness, direct=False)
    pair["git"]["revision"] = "c" * 40
    fairness_path = tmp_path / "fairness.json"
    pair_path = tmp_path / "pair.json"
    fairness_identity = _write(fairness_path, fairness)
    pair_identity = _write(pair_path, pair)

    with pytest.raises(ValueError, match="pair monitor identity differs"):
        build_guard(
            runtime_fairness_report_path=fairness_path,
            expected_runtime_fairness_sha256=fairness_identity["sha256"],
            pair_monitor_path=pair_path,
            expected_pair_monitor_sha256=pair_identity["sha256"],
            expected_monitor_name=MONITOR,
        )


def test_source_waiter_status_binds_terminal_report_identity(tmp_path: Path) -> None:
    deployment = {
        "path": (tmp_path / "deployment.json").resolve().as_posix(),
        "bytes": 100,
        "sha256": "d" * 64,
    }
    final = tmp_path / "final_report.json"
    audit = {
        "path": final.resolve().as_posix(),
        "bytes": 200,
        "sha256": "e" * 64,
    }
    status = {
        "schema_version": 1,
        "role": "quality_bridge_runtime_compute_fairness_waiter",
        "status": "pass",
        "detail": "terminal_runtime_compute_fairness_verified",
        "pid": 832_803,
        "deployment_receipt": deployment,
        "audit_output": audit,
        "scope": copy.deepcopy(SOURCE_SCOPE),
    }

    verified = validate_source_status(
        status,
        expected_pid=832_803,
        expected_deployment_identity=deployment,
        expected_source_final_report=final,
    )

    assert verified["terminal"] is True
    assert verified["audit_output"] == audit


def test_source_deployment_binds_exact_checkout_and_outputs(tmp_path: Path) -> None:
    source_status = tmp_path / "waiter_status.json"
    final = tmp_path / "final_report.json"
    source_git = {
        "path": "/root/checkouts/runtime-fairness/CoFiTok-internal",
        "revision": "f" * 40,
        "tree": "1" * 40,
        "branch": "analysis/runtime-fairness-v1",
        "tracked_dirty": False,
    }
    deployment = {
        "schema_version": 1,
        "role": "quality_bridge_runtime_compute_fairness_deployment_receipt",
        "status": "pass",
        "auditor": {"checkout": source_git, "source": {}},
        "targets": {
            "status_output": source_status.resolve().as_posix(),
            "audit_output": final.resolve().as_posix(),
        },
        "scope": {
            "cpu_only_waiter": True,
            "gpu_execution_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_or_release_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }

    validate_source_deployment(
        deployment,
        expected_source_git=source_git,
        expected_source_status=source_status,
        expected_source_final_report=final,
    )

    tampered = copy.deepcopy(deployment)
    tampered["targets"]["audit_output"] = (tmp_path / "other.json").as_posix()
    with pytest.raises(ValueError, match="deployment contract differs"):
        validate_source_deployment(
            tampered,
            expected_source_git=source_git,
            expected_source_status=source_status,
            expected_source_final_report=final,
        )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_waiter_builds_observational_guard_after_terminal_sources(
    tmp_path: Path,
) -> None:
    quality_root = tmp_path / "quality"
    source_root = quality_root / "reports/runtime_compute_fairness"
    source_status = source_root / "waiter_status.json"
    source_final = source_root / "final_report.json"
    source_deployment = source_root / "deployment_receipt.json"
    pair_monitor = quality_root / "pair_monitor.json"
    output_root = quality_root / "reports/runtime_compute_claim_guard_v1"
    status_output = output_root / "waiter_status.json"
    pid_file = output_root / "waiter.pid"
    deployment_output = output_root / "deployment_receipt.json"
    guard_output = output_root / "runtime_compute_claim_guard.json"

    fairness = _fairness_report(tmp_path)
    fairness["contract"]["output_root"] = quality_root.resolve().as_posix()
    fairness["contract"]["training_run_dirs"] = [
        (quality_root / "cofitok").resolve().as_posix(),
        (quality_root / "dense").resolve().as_posix(),
    ]
    pair = _pair_report(fairness, direct=False)
    fairness_identity = _write(source_final, fairness)
    _write(pair_monitor, pair)

    source_git = {
        "path": "/root/autodl-tmp/CoFiTok/checkouts/runtime-fairness/CoFiTok-internal",
        "revision": "f" * 40,
        "tree": "1" * 40,
        "branch": "analysis/runtime-fairness-v1",
        "tracked_dirty": False,
    }
    source_deployment_report = {
        "schema_version": 1,
        "role": "quality_bridge_runtime_compute_fairness_deployment_receipt",
        "status": "pass",
        "auditor": {"checkout": source_git, "source": {}},
        "targets": {
            "status_output": source_status.resolve().as_posix(),
            "audit_output": source_final.resolve().as_posix(),
        },
        "scope": {
            "cpu_only_waiter": True,
            "gpu_execution_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_or_release_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }
    source_deployment_identity = _write(
        source_deployment,
        source_deployment_report,
    )
    _write(
        source_status,
        {
            "schema_version": 1,
            "role": "quality_bridge_runtime_compute_fairness_waiter",
            "status": "pass",
            "detail": "terminal_runtime_compute_fairness_verified",
            "pid": 9_999_999,
            "deployment_receipt": source_deployment_identity,
            "audit_output": fairness_identity,
            "scope": copy.deepcopy(SOURCE_SCOPE),
        },
    )
    control_git = {
        "revision": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "branch": _git("branch", "--show-current"),
    }
    args = Namespace(
        project=ROOT,
        quality_output_root=quality_root,
        source_status=source_status,
        source_final_report=source_final,
        source_deployment_receipt=source_deployment,
        expected_source_deployment_receipt_sha256=source_deployment_identity[
            "sha256"
        ],
        pair_monitor=pair_monitor,
        output_root=output_root,
        status_output=status_output,
        pid_file=pid_file,
        deployment_receipt_output=deployment_output,
        guard_output=guard_output,
        expected_source_pid=9_999_999,
        expected_control_revision=control_git["revision"],
        expected_control_tree=control_git["tree"],
        expected_control_branch=control_git["branch"],
        expected_source_control_revision=source_git["revision"],
        expected_source_control_tree=source_git["tree"],
        expected_source_control_branch=source_git["branch"],
        expected_training_revision=REVISION,
        expected_training_branch=BRANCH,
        expected_monitor_name=MONITOR,
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert run_waiter(args) == 0
    status = json.loads(status_output.read_text(encoding="utf-8"))
    guard = json.loads(guard_output.read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["phase"] == "completed"
    assert guard["decision"] == "runtime_cost_claims_observational_only"
    assert guard["claim_policy"]["cost_efficiency_ranking_allowed"] is False
    assert not pid_file.exists()

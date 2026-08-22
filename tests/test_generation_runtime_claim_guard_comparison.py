from __future__ import annotations

import copy
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from cofitok.inference_replay import file_identity
from scripts.build_generation_runtime_compute_claim_guard import build_guard
from scripts.build_generation_runtime_claim_guard_comparison import build_comparison
from scripts.wait_generation_runtime_claim_guard_comparison import run_waiter
from tests.test_generation_runtime_compute_claim_guard import (
    BRANCH,
    MONITOR,
    REVISION,
    _cost,
    _fairness_report,
    _pair_report,
    _write,
)


CANONICAL_CONTROL = "6" * 40
STRICT_CONTROL = "0" * 40
ROOT = Path(__file__).resolve().parents[1]


def _status(
    path: Path,
    *,
    guard_identity: dict,
    control_revision: str,
    pid: int,
    source_pid: int,
) -> dict:
    return _write(
        path,
        {
            "schema_version": 1,
            "role": "generation_quality_bridge_runtime_claim_guard_waiter",
            "status": "pass",
            "detail": "runtime_cost_claims_observational_only",
            "phase": "completed",
            "pid": pid,
            "expected": {
                "control_git": {
                    "revision": control_revision,
                    "tree": "1" * 40,
                    "branch": "analysis/runtime-claim-guard",
                    "tracked_dirty": False,
                },
                "source_waiter": {"pid": source_pid},
                "pair_monitor": {
                    "training_revision": REVISION,
                    "training_branch": BRANCH,
                },
            },
            "deployment_receipt": {
                "path": "/tmp/deployment.json",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "runtime_claim_guard": {
                "identity": guard_identity,
                "status": "pass",
                "decision": "runtime_cost_claims_observational_only",
            },
        },
    )


def _wrapper(
    path: Path,
    *,
    canonical_pid: int,
    strict_pid: int,
    source_pid: int,
) -> dict:
    return _write(
        path,
        {
            "schema_version": 1,
            "role": "generation_runtime_claim_guard_strict_replay_wrapper_deployment",
            "status": "pass",
            "scope": {
                "cpu_only": True,
                "non_authorizing": True,
                "gpu_execution_allowed": False,
                "training_process_signals_allowed": False,
                "unrelated_process_signals_allowed": False,
                "old_waiter_signals_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "full_300k_launch_allowed": False,
            },
            "strict_control": {
                "revision": STRICT_CONTROL,
                "tree": "1" * 40,
                "branch": "analysis/runtime-claim-guard",
                "tracked_dirty": False,
            },
            "wrapper": {
                "pid": strict_pid,
                "ppid": 1,
                "cuda_visible_devices": "",
                "omp_num_threads": "1",
                "mkl_num_threads": "1",
                "ionice": "idle",
            },
            "old_canonical_waiter": {"pid": canonical_pid},
            "runtime_fairness_source_waiter": {"pid": source_pid},
            "behavior": {
                "waits_for_exact_old_pid_start_ticks_and_cmdline": True,
                "executes_only_after_old_waiter_exits_or_identity_changes": True,
                "writes_independent_output": True,
                "replaces_canonical_guard": False,
                "comparison_required_before_runtime_claim_trust": True,
            },
        },
    )


def _sources(tmp_path: Path) -> dict:
    fairness = _fairness_report(tmp_path)
    dense = _cost(
        elapsed=9_003.5,
        peak_vram=70_000,
        adjustment=3.5,
    )
    fairness["methods"]["dense_identity"]["training_cost"] = dense
    cofitok = fairness["methods"]["cofitok"]["training_cost"]
    fairness["descriptive_comparison"].update(
        {
            "cofitok_minus_dense_adjusted_elapsed_seconds": (
                cofitok["elapsed_seconds"] - dense["elapsed_seconds"]
            ),
            "cofitok_adjusted_elapsed_relative_change": (
                cofitok["elapsed_seconds"] / dense["elapsed_seconds"] - 1.0
            ),
            "cofitok_throughput_relative_change": (
                cofitok["images_per_second"] / dense["images_per_second"] - 1.0
            ),
        }
    )
    pair = _pair_report(fairness, direct=False)
    fairness_path = tmp_path / "fairness.json"
    pair_path = tmp_path / "pair.json"
    fairness_identity = _write(fairness_path, fairness)
    pair_identity = _write(pair_path, pair)
    strict = build_guard(
        runtime_fairness_report_path=fairness_path,
        expected_runtime_fairness_sha256=fairness_identity["sha256"],
        pair_monitor_path=pair_path,
        expected_pair_monitor_sha256=pair_identity["sha256"],
        expected_monitor_name=MONITOR,
    )
    canonical = copy.deepcopy(strict)
    canonical["claim_policy"].pop("exclusive_gpu_observation_coverage_verified")
    canonical["claim_policy"].pop("recovery_adjusted_elapsed_exact_for_both_methods")
    canonical["claim_policy"].pop("physical_lower_bound_label_required")
    canonical["claim_policy"].pop("observational_only_label_required")
    canonical["metric_roles"]["adjusted_training_elapsed_and_throughput"].pop(
        "physical_lower_bound_methods"
    )
    canonical_path = tmp_path / "canonical_guard.json"
    strict_path = tmp_path / "strict_guard.json"
    canonical_identity = _write(canonical_path, canonical)
    strict_identity = _write(strict_path, strict)
    canonical_pid = 22_6725
    strict_pid = 890_935
    source_pid = 75_555
    canonical_status_path = tmp_path / "canonical_status.json"
    strict_status_path = tmp_path / "strict_status.json"
    wrapper_path = tmp_path / "wrapper.json"
    canonical_status_identity = _status(
        canonical_status_path,
        guard_identity=canonical_identity,
        control_revision=CANONICAL_CONTROL,
        pid=canonical_pid,
        source_pid=source_pid,
    )
    strict_status_identity = _status(
        strict_status_path,
        guard_identity=strict_identity,
        control_revision=STRICT_CONTROL,
        pid=strict_pid,
        source_pid=source_pid,
    )
    wrapper_identity = _wrapper(
        wrapper_path,
        canonical_pid=canonical_pid,
        strict_pid=strict_pid,
        source_pid=source_pid,
    )
    return {
        "canonical": canonical,
        "canonical_path": canonical_path,
        "canonical_identity": canonical_identity,
        "canonical_status_path": canonical_status_path,
        "canonical_status_identity": canonical_status_identity,
        "strict": strict,
        "strict_path": strict_path,
        "strict_identity": strict_identity,
        "strict_status_path": strict_status_path,
        "strict_status_identity": strict_status_identity,
        "wrapper_path": wrapper_path,
        "wrapper_identity": wrapper_identity,
    }


def _build(source: dict) -> dict:
    return build_comparison(
        canonical_status_path=source["canonical_status_path"],
        expected_canonical_status_sha256=source["canonical_status_identity"][
            "sha256"
        ],
        canonical_guard_path=source["canonical_path"],
        expected_canonical_guard_sha256=source["canonical_identity"]["sha256"],
        strict_status_path=source["strict_status_path"],
        expected_strict_status_sha256=source["strict_status_identity"]["sha256"],
        strict_guard_path=source["strict_path"],
        expected_strict_guard_sha256=source["strict_identity"]["sha256"],
        wrapper_receipt_path=source["wrapper_path"],
        expected_wrapper_receipt_sha256=source["wrapper_identity"]["sha256"],
        expected_canonical_control_revision=CANONICAL_CONTROL,
        expected_strict_control_revision=STRICT_CONTROL,
        expected_training_revision=REVISION,
        expected_training_branch=BRANCH,
    )


def test_observational_canonical_guard_is_semantically_verified(
    tmp_path: Path,
) -> None:
    report = _build(_sources(tmp_path))

    assert report["status"] == "pass"
    assert report["decision"] == (
        "canonical_observational_runtime_claim_semantically_verified"
    )
    assert report["canonical_equivalence"][
        "semantically_equivalent_for_observational_runtime_claims"
    ] is True
    assert report["claim_policy"]["canonical_runtime_claim_trusted"] is True
    assert report["claim_policy"]["cost_efficiency_ranking_allowed"] is False
    assert report["claim_boundary"]["process_signals_allowed"] is False


def test_canonical_direct_ranking_is_rejected_by_strict_policy(
    tmp_path: Path,
) -> None:
    source = _sources(tmp_path)
    canonical = json.loads(source["canonical_path"].read_text(encoding="utf-8"))
    canonical["claim_policy"]["cost_efficiency_ranking_allowed"] = True
    canonical["claim_policy"]["training_wall_clock_direct_comparison_allowed"] = True
    canonical["claim_policy"]["training_throughput_direct_comparison_allowed"] = True
    source["canonical_identity"] = _write(source["canonical_path"], canonical)
    source["canonical_status_identity"] = _status(
        source["canonical_status_path"],
        guard_identity=source["canonical_identity"],
        control_revision=CANONICAL_CONTROL,
        pid=22_6725,
        source_pid=75_555,
    )

    report = _build(source)

    assert report["decision"] == "canonical_runtime_claim_rejected_strict_guard_controls"
    assert report["claim_policy"]["canonical_runtime_claim_trusted"] is False
    assert report["claim_policy"]["training_wall_clock_direct_comparison_allowed"] is False


def test_strict_guard_must_bind_both_recovery_lower_bounds(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    strict = json.loads(source["strict_path"].read_text(encoding="utf-8"))
    strict["metric_roles"]["adjusted_training_elapsed_and_throughput"][
        "physical_lower_bound_methods"
    ] = ["cofitok"]
    source["strict_identity"] = _write(source["strict_path"], strict)
    source["strict_status_identity"] = _status(
        source["strict_status_path"],
        guard_identity=source["strict_identity"],
        control_revision=STRICT_CONTROL,
        pid=890_935,
        source_pid=75_555,
    )

    with pytest.raises(ValueError, match="strict runtime claim policy"):
        _build(source)


def test_wrapper_must_prohibit_old_waiter_signals(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    wrapper = json.loads(source["wrapper_path"].read_text(encoding="utf-8"))
    wrapper["scope"]["old_waiter_signals_allowed"] = True
    source["wrapper_identity"] = _write(source["wrapper_path"], wrapper)

    with pytest.raises(ValueError, match="wrapper deployment contract"):
        _build(source)


def test_bound_guard_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    source = _sources(tmp_path)

    with pytest.raises(ValueError, match="canonical runtime claim guard SHA256 differs"):
        build_comparison(
            canonical_status_path=source["canonical_status_path"],
            expected_canonical_status_sha256=source["canonical_status_identity"][
                "sha256"
            ],
            canonical_guard_path=source["canonical_path"],
            expected_canonical_guard_sha256="f" * 64,
            strict_status_path=source["strict_status_path"],
            expected_strict_status_sha256=source["strict_status_identity"]["sha256"],
            strict_guard_path=source["strict_path"],
            expected_strict_guard_sha256=source["strict_identity"]["sha256"],
            wrapper_receipt_path=source["wrapper_path"],
            expected_wrapper_receipt_sha256=source["wrapper_identity"]["sha256"],
            expected_canonical_control_revision=CANONICAL_CONTROL,
            expected_strict_control_revision=STRICT_CONTROL,
            expected_training_revision=REVISION,
            expected_training_branch=BRANCH,
        )


def test_waiter_publishes_completed_comparison(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    output_root = tmp_path / "reports/runtime_claim_comparison"
    status_output = output_root / "waiter_status.json"
    pid_file = output_root / "waiter.pid.json"
    deployment_output = output_root / "deployment_receipt.json"
    comparison_output = output_root / "comparison.json"

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    args = Namespace(
        project=ROOT,
        quality_output_root=tmp_path,
        canonical_status=source["canonical_status_path"],
        canonical_guard=source["canonical_path"],
        strict_status=source["strict_status_path"],
        strict_guard=source["strict_path"],
        wrapper_receipt=source["wrapper_path"],
        expected_wrapper_receipt_sha256=source["wrapper_identity"]["sha256"],
        output_root=output_root,
        status_output=status_output,
        pid_file=pid_file,
        deployment_receipt_output=deployment_output,
        comparison_output=comparison_output,
        expected_control_revision=git("rev-parse", "HEAD"),
        expected_control_tree=git("rev-parse", "HEAD^{tree}"),
        expected_control_branch=git("branch", "--show-current"),
        expected_canonical_control_revision=CANONICAL_CONTROL,
        expected_strict_control_revision=STRICT_CONTROL,
        expected_training_revision=REVISION,
        expected_training_branch=BRANCH,
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert run_waiter(args) == 0
    status = json.loads(status_output.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_output.read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["comparison"]["identity"] == file_identity(comparison_output)
    assert comparison["claim_policy"]["canonical_runtime_claim_trusted"] is True
    assert comparison["claim_policy"]["cost_efficiency_ranking_allowed"] is False
    assert not pid_file.exists()

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofitok.generation.terminal_snr_large_capacity import (
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_EVALUATION_CONTRACT,
    METHODS,
    MILESTONE_STEPS,
    TARGET_STEPS,
)
from cofitok.generation.terminal_snr_large_capacity_execution import (
    GOAL_BINDING,
    LIVE_SNAPSHOT_BOUNDARY,
    PAIR_VALIDATION_BOUNDARY,
    RUNTIME_BASELINE,
    RUNTIME_BENCHMARK_STEPS,
    RUNTIME_CANDIDATES,
    RUNTIME_MAX_MEMORY_FRACTION,
    RUNTIME_WARMUP_STEPS,
    STAGE_BOUNDARY,
    STORAGE_ADDITIONAL_BYTES,
    STORAGE_CHECKPOINT_COUNT,
    STORAGE_CHECKPOINT_SIZE_MULTIPLIER,
    STORAGE_ESTIMATED_SAMPLE_BYTES,
    STORAGE_SAFETY_MARGIN_BYTES,
    STORAGE_SAMPLE_COUNT,
    STORAGE_STAGE,
    build_terminal_snr_large_capacity_pair_validation,
    build_terminal_snr_large_capacity_stage_authorization,
    terminal_snr_large_capacity_execution_lock_path,
    validate_terminal_snr_large_capacity_live_snapshot,
    validate_terminal_snr_large_capacity_pair_validation,
    validate_terminal_snr_large_capacity_runtime_selection,
    validate_terminal_snr_large_capacity_stage_authorization,
    validate_terminal_snr_large_capacity_storage_capacity,
)
from scripts.check_generation_storage_capacity import build_storage_capacity_report
from scripts import build_generation_terminal_snr_large_capacity_live_snapshot as live_snapshot
from scripts.select_generation_training_runtime import (
    _selection_contract,
    select_runtime_candidate,
)
from test_generation_terminal_snr_large_capacity import (
    _build as _build_preparation,
    _config_validation,
    _read_config,
)
from test_generation_runtime_selection import _candidate, _method


OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "terminal_snr_endpoint0975_large_capacity_300k_v1"
)
PREPARATION_ID = {
    "path": "/evidence/terminal_snr_large_capacity_preparation.json",
    "bytes": 123,
    "sha256": "a" * 64,
}
EXECUTION_GIT = {
    "revision": "b" * 40,
    "tree": "c" * 40,
    "branch": "scale/generation-terminal-snr-capacity-full-v1",
    "tracked_dirty": False,
}


def _identity(path: str, character: str) -> dict[str, object]:
    return {"path": path, "bytes": 123, "sha256": character * 64}


def _preparation() -> dict:
    return {
        "selection": {
            "output_root": OUTPUT_ROOT,
            "fresh_initialization_required": True,
            "resume_checkpoint_allowed": False,
            "condition": "endpoint0975",
            "methods": list(METHODS),
            "configured_training_steps": TARGET_STEPS,
        },
        "source_evidence": {
            "configs": {
                "cofitok": _identity("/configs/cofitok.json", "d"),
                "dense_identity": _identity("/configs/dense.json", "e"),
            }
        },
    }


def _patch_preparation(monkeypatch: pytest.MonkeyPatch) -> dict:
    prepared = _preparation()
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity_execution."
        "validate_terminal_snr_large_capacity_preparation_contract",
        lambda value: value,
    )
    return prepared


def _build(monkeypatch: pytest.MonkeyPatch) -> dict:
    return build_terminal_snr_large_capacity_stage_authorization(
        preparation=_patch_preparation(monkeypatch),
        preparation_identity=PREPARATION_ID,
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
        approved_by="user",
        approved_at="2026-09-08T00:00:00+00:00",
        source_instruction=(
            "Continue toward the explicitly requested 250M/300K matched "
            "training, per-method 50K DDIM-250 formal evaluation, and terminal audit."
        ),
    )


def _build_pair(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    preparation = _build_preparation(monkeypatch)
    stage = build_terminal_snr_large_capacity_stage_authorization(
        preparation=preparation,
        preparation_identity=PREPARATION_ID,
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
        approved_by="user",
        approved_at="2026-09-08T00:00:00+00:00",
        source_instruction=(
            "Explicit 250M/300K matched training, per-method 50K DDIM-250 "
            "formal evaluation, and terminal completion audit."
        ),
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity_execution."
        "validate_terminal_snr_large_capacity_config_pair",
        lambda **_: _config_validation(),
    )
    configs = preparation["source_evidence"]["configs"]
    report = build_terminal_snr_large_capacity_pair_validation(
        preparation=preparation,
        preparation_identity=PREPARATION_ID,
        stage_authorization=stage,
        stage_authorization_identity=_identity("/evidence/stage.json", "f"),
        cofitok_config=_read_config("cofitok"),
        cofitok_config_identity=configs["cofitok"],
        dense_config=_read_config("dense_identity"),
        dense_config_identity=configs["dense_identity"],
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
    )
    return report, preparation, stage


def _runtime_selection(preparation: dict) -> tuple[dict, dict, dict, str]:
    candidates = [
        _candidate(1, 64, _method(4.0), _method(3.8)),
        _candidate(2, 32, _method(3.2), _method(3.1)),
        _candidate(4, 16, _method(2.8), _method(2.9)),
    ]
    report = select_runtime_candidate(
        candidates,
        expected_effective_batch=EFFECTIVE_BATCH_SIZE,
        max_memory_fraction=RUNTIME_MAX_MEMORY_FRACTION,
        baseline_candidate=RUNTIME_BASELINE,
    )
    configs = preparation["source_evidence"]["configs"]
    config_sha = {
        method: configs[method]["sha256"] for method in METHODS
    }
    run_dirs = {
        method: f"{OUTPUT_ROOT}/training/{method}" for method in METHODS
    }
    benchmark_root = "/tmp/terminal-snr-large-capacity-runtime"
    lock = _selection_contract(
        run_dirs=[Path(run_dirs[method]) for method in METHODS],
        candidates=list(RUNTIME_CANDIDATES),
        baseline_candidate=RUNTIME_BASELINE,
        expected_effective_batch=EFFECTIVE_BATCH_SIZE,
        benchmark_steps=RUNTIME_BENCHMARK_STEPS,
        warmup_steps=RUNTIME_WARMUP_STEPS,
        max_memory_fraction=RUNTIME_MAX_MEMORY_FRACTION,
        target_steps=TARGET_STEPS,
        revision=EXECUTION_GIT["revision"],
        branch=EXECUTION_GIT["branch"],
        config_sha256=config_sha,
        benchmark_root=Path(benchmark_root),
    )
    report.update(
        git_revision=EXECUTION_GIT["revision"],
        config_sha256=config_sha,
        benchmark_root=benchmark_root,
        selection_lock=lock,
    )
    return report, configs, run_dirs, benchmark_root


def _storage_capacity() -> dict:
    reference_bytes = 1_010_933_866
    checkpoint_bytes = int(
        reference_bytes * STORAGE_CHECKPOINT_SIZE_MULTIPLIER
    )
    required = (
        STORAGE_CHECKPOINT_COUNT * checkpoint_bytes
        + STORAGE_SAMPLE_COUNT * STORAGE_ESTIMATED_SAMPLE_BYTES
        + STORAGE_ADDITIONAL_BYTES
        + STORAGE_SAFETY_MARGIN_BYTES
    )
    report = build_storage_capacity_report(
        stage=STORAGE_STAGE,
        path=Path("."),
        total_bytes=required * 3,
        used_bytes=required,
        free_bytes=required * 2,
        checkpoint_count=STORAGE_CHECKPOINT_COUNT,
        checkpoint_bytes=reference_bytes,
        checkpoint_size_multiplier=STORAGE_CHECKPOINT_SIZE_MULTIPLIER,
        sample_count=STORAGE_SAMPLE_COUNT,
        estimated_sample_bytes=STORAGE_ESTIMATED_SAMPLE_BYTES,
        additional_bytes=STORAGE_ADDITIONAL_BYTES,
        safety_margin_bytes=STORAGE_SAFETY_MARGIN_BYTES,
        git={
            "revision": EXECUTION_GIT["revision"],
            "branch": EXECUTION_GIT["branch"],
            "tracked_dirty": False,
        },
        hostname="pro6000",
        checked_at="2026-09-08T00:00:00+00:00",
    )
    report["filesystem"]["path"] = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation"
    )
    return report


def _live_snapshot(preparation: dict) -> tuple[dict, dict, dict]:
    runtime_id = _identity("/evidence/runtime.json", "1")
    storage_id = _identity("/evidence/storage.json", "2")
    configs = preparation["source_evidence"]["configs"]
    measured = _method(1.0)
    lock = terminal_snr_large_capacity_execution_lock_path(OUTPUT_ROOT)
    report = {
        "schema_version": (
            "cofitok_generation_terminal_snr_large_capacity_live_snapshot_v1"
        ),
        "role": (
            "terminal_snr_large_capacity_idle_gpu_and_output_absence_snapshot"
        ),
        "status": "pass",
        "execution_checkout": EXECUTION_GIT,
        "source_evidence": {
            "runtime_selection": copy.deepcopy(runtime_id),
            "storage_capacity": copy.deepcopy(storage_id),
            "configs": copy.deepcopy(configs),
        },
        "output_root": OUTPUT_ROOT,
        "execution_lock": lock,
        "training_run_dirs": {
            method: f"{OUTPUT_ROOT}/training/{method}" for method in METHODS
        },
        "gpu_inventory": [{
            "index": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "memory_used_mib": 1,
            "memory_total_mib": 97_887,
            "utilization_percent": 0,
        }],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root_absent": True,
        "training_state_absent": True,
        "execution_lock_free": True,
        "filesystem": {
            "path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
            "free_bytes": 512 * 1024**3,
            "required_free_bytes": 256 * 1024**3,
            "headroom_bytes": 256 * 1024**3,
        },
        "runtime_environment": measured["runtime_environment"],
        "runtime_environment_sha256": measured["runtime_environment_sha256"],
        "dataset_provenance": measured["dataset_provenance"],
        "dataset_identity_sha256": measured["dataset_provenance"][
            "identity_sha256"
        ],
        "hostname": "pro6000",
        "captured_at": "2026-09-08T00:00:00+00:00",
        "authorization_boundary": LIVE_SNAPSHOT_BOUNDARY,
    }
    return report, runtime_id, storage_id


def test_stage_authorization_binds_exact_fresh_300k_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    selection = report["selection"]
    assert selection["preparation"] == PREPARATION_ID
    assert selection["execution_checkout"] == EXECUTION_GIT
    assert selection["output_root"] == OUTPUT_ROOT
    assert selection["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert selection["effective_batch_size"] == EFFECTIVE_BATCH_SIZE
    assert selection["configured_training_steps"] == TARGET_STEPS
    assert selection["milestone_steps"] == list(MILESTONE_STEPS)
    assert selection["fresh_initialization_required"] is True
    assert selection["resume_allowed"] is False
    assert selection["formal_evaluation"] == FORMAL_EVALUATION_CONTRACT
    assert report["approval_record"]["goal_binding"] == GOAL_BINDING
    assert report["authorization_boundary"] == STAGE_BOUNDARY
    assert report["next_stage"]["execution_ready"] is False
    assert report["next_stage"]["training_launch_allowed"] is False


def test_stage_authorization_accepts_the_canonical_preparation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report = build_terminal_snr_large_capacity_stage_authorization(
        preparation=preparation,
        preparation_identity=PREPARATION_ID,
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
        approved_by="user",
        approved_at="2026-09-08T00:00:00+00:00",
        source_instruction=(
            "Explicit 250M/300K matched training, per-method 50K DDIM-250 "
            "formal evaluation, and terminal completion audit."
        ),
    )
    assert report["selection"]["output_root"] == OUTPUT_ROOT
    assert report["authorization_boundary"]["training_launch_allowed"] is False


def test_stage_authorization_is_not_execution_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    assert report["authorization_boundary"]["decision_is_execution_authorization"] is False
    assert report["authorization_boundary"]["gpu_runtime_preflight_allowed"] is True
    assert report["authorization_boundary"]["training_launch_allowed"] is False
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert report["authorization_boundary"]["process_signals_allowed"] is False


def test_stage_authorization_rejects_a_vague_source_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="explicit 250M/300K"):
        build_terminal_snr_large_capacity_stage_authorization(
            preparation=_patch_preparation(monkeypatch),
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
            approved_by="user",
            approved_at="now",
            source_instruction="continue the experiment",
        )


def test_stage_authorization_rejects_resume_or_changed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    prepared["selection"]["resume_checkpoint_allowed"] = True
    with pytest.raises(ValueError, match="selection differs"):
        build_terminal_snr_large_capacity_stage_authorization(
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
            approved_by="user",
            approved_at="now",
            source_instruction="explicit goal",
        )


def test_stage_authorization_rejects_implicit_launch_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["next_stage"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="authorization differs"):
        validate_terminal_snr_large_capacity_stage_authorization(
            altered,
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            expected_output_root=OUTPUT_ROOT,
        )


@pytest.mark.parametrize(
    "section",
    ["selection", "approval_record", "next_stage", "authorization_boundary"],
)
def test_stage_authorization_rejects_unexpected_nested_fields(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered[section]["unexpected_authority"] = True
    with pytest.raises(ValueError, match="authorization differs"):
        validate_terminal_snr_large_capacity_stage_authorization(
            altered,
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            expected_output_root=OUTPUT_ROOT,
        )


def test_stage_authorization_requires_exact_clean_execution_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    dirty = {**EXECUTION_GIT, "tracked_dirty": True}
    with pytest.raises(ValueError, match="exact clean checkout"):
        build_terminal_snr_large_capacity_stage_authorization(
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=dirty,
            output_root=OUTPUT_ROOT,
            approved_by="user",
            approved_at="now",
            source_instruction="explicit goal",
        )


def test_pair_validation_replays_exact_fresh_matched_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _, _ = _build_pair(monkeypatch)
    assert report["status"] == "pass"
    assert report["execution_checkout"] == EXECUTION_GIT
    assert report["validation"] == _config_validation()
    assert report["selection"]["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert report["selection"]["configured_training_steps"] == TARGET_STEPS
    assert report["selection"]["fresh_initialization_required"] is True
    assert report["selection"]["resume_allowed"] is False
    assert report["authorization_boundary"] == PAIR_VALIDATION_BOUNDARY
    assert report["next_stage"]["execution_ready"] is False
    assert report["next_stage"]["training_launch_allowed"] is False


def test_pair_validation_requires_preparation_config_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, preparation, stage = _build_pair(monkeypatch)
    del report
    configs = preparation["source_evidence"]["configs"]
    with pytest.raises(ValueError, match="config identity differs"):
        build_terminal_snr_large_capacity_pair_validation(
            preparation=preparation,
            preparation_identity=PREPARATION_ID,
            stage_authorization=stage,
            stage_authorization_identity=_identity("/evidence/stage.json", "f"),
            cofitok_config=_read_config("cofitok"),
            cofitok_config_identity={
                **configs["cofitok"],
                "sha256": "0" * 64,
            },
            dense_config=_read_config("dense_identity"),
            dense_config_identity=configs["dense_identity"],
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
        )


def test_pair_validation_rejects_drift_from_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, preparation, stage = _build_pair(monkeypatch)
    del report
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity_execution."
        "validate_terminal_snr_large_capacity_config_pair",
        lambda **_: {**_config_validation(), "relative_parameter_gap": 0.01},
    )
    configs = preparation["source_evidence"]["configs"]
    with pytest.raises(ValueError, match="differs from preparation"):
        build_terminal_snr_large_capacity_pair_validation(
            preparation=preparation,
            preparation_identity=PREPARATION_ID,
            stage_authorization=stage,
            stage_authorization_identity=_identity("/evidence/stage.json", "f"),
            cofitok_config=_read_config("cofitok"),
            cofitok_config_identity=configs["cofitok"],
            dense_config=_read_config("dense_identity"),
            dense_config_identity=configs["dense_identity"],
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
        )


def test_pair_validation_contract_rejects_implicit_launch_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, preparation, stage = _build_pair(monkeypatch)
    altered = copy.deepcopy(report)
    altered["authorization_boundary"]["training_launch_allowed"] = True
    configs = preparation["source_evidence"]["configs"]
    with pytest.raises(ValueError, match="pair validation differs"):
        validate_terminal_snr_large_capacity_pair_validation(
            altered,
            preparation=preparation,
            preparation_identity=PREPARATION_ID,
            stage_authorization=stage,
            stage_authorization_identity=_identity("/evidence/stage.json", "f"),
            cofitok_config=_read_config("cofitok"),
            cofitok_config_identity=configs["cofitok"],
            dense_config=_read_config("dense_identity"),
            dense_config_identity=configs["dense_identity"],
            execution_checkout=EXECUTION_GIT,
            expected_output_root=OUTPUT_ROOT,
        )


def test_large_capacity_runtime_selection_binds_base256_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report, configs, run_dirs, benchmark_root = _runtime_selection(preparation)
    validated = validate_terminal_snr_large_capacity_runtime_selection(
        report,
        execution_checkout=EXECUTION_GIT,
        config_identities=configs,
        run_dirs=run_dirs,
        benchmark_root=benchmark_root,
    )
    assert validated["micro_batch_size"] == 4
    assert validated["gradient_accumulation_steps"] == 16
    assert validated["effective_batch_size"] == 64
    assert validated["max_memory_fraction"] <= RUNTIME_MAX_MEMORY_FRACTION


def test_large_capacity_runtime_selection_rejects_selected_candidate_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report, configs, run_dirs, benchmark_root = _runtime_selection(preparation)
    report["selected"]["micro_batch_size"] = 2
    report["selected"]["gradient_accumulation_steps"] = 32
    with pytest.raises(ValueError, match="runtime selection differs"):
        validate_terminal_snr_large_capacity_runtime_selection(
            report,
            execution_checkout=EXECUTION_GIT,
            config_identities=configs,
            run_dirs=run_dirs,
            benchmark_root=benchmark_root,
        )


def test_large_capacity_storage_capacity_binds_full_artifact_budget() -> None:
    report = _storage_capacity()
    validated = validate_terminal_snr_large_capacity_storage_capacity(
        report,
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
    )
    assert validated["plan"]["checkpoint_count"] == 14
    assert validated["plan"]["sample_count"] == 116_384
    assert validated["headroom_bytes"] >= 0


def test_large_capacity_storage_capacity_rejects_weaker_reserve() -> None:
    report = _storage_capacity()
    report["plan"]["checkpoint_count"] -= 1
    with pytest.raises(ValueError, match="storage evidence differs"):
        validate_terminal_snr_large_capacity_storage_capacity(
            report,
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
        )


def test_large_capacity_live_snapshot_binds_idle_gpu_and_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report, runtime_id, storage_id = _live_snapshot(preparation)
    validated = validate_terminal_snr_large_capacity_live_snapshot(
        report,
        expected_output_root=OUTPUT_ROOT,
        expected_execution_checkout=EXECUTION_GIT,
        expected_execution_lock=report["execution_lock"],
        expected_runtime_selection_identity=runtime_id,
        expected_storage_capacity_identity=storage_id,
        expected_config_identities=preparation["source_evidence"]["configs"],
    )
    assert validated["gpu_compute_processes"] == []
    assert validated["output_root_absent"] is True
    assert validated["training_state_absent"] is True
    assert validated["authorization_boundary"]["training_launch_allowed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_compute_processes", [{"pid": 123, "process_name": "python"}]),
        ("conflicting_processes", [{"pid": 456, "command": "trainer"}]),
        ("output_root_absent", False),
        ("training_state_absent", False),
        ("execution_lock_free", False),
    ],
)
def test_large_capacity_live_snapshot_rejects_non_idle_or_existing_state(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report, runtime_id, storage_id = _live_snapshot(preparation)
    report[field] = value
    with pytest.raises(ValueError, match="live snapshot differs"):
        validate_terminal_snr_large_capacity_live_snapshot(
            report,
            expected_output_root=OUTPUT_ROOT,
            expected_execution_checkout=EXECUTION_GIT,
            expected_execution_lock=report["execution_lock"],
            expected_runtime_selection_identity=runtime_id,
            expected_storage_capacity_identity=storage_id,
            expected_config_identities=(
                preparation["source_evidence"]["configs"]
            ),
        )


def test_large_capacity_live_snapshot_rejects_bound_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report, runtime_id, storage_id = _live_snapshot(preparation)
    report["source_evidence"]["runtime_selection"]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="live snapshot differs"):
        validate_terminal_snr_large_capacity_live_snapshot(
            report,
            expected_output_root=OUTPUT_ROOT,
            expected_execution_checkout=EXECUTION_GIT,
            expected_execution_lock=report["execution_lock"],
            expected_runtime_selection_identity=runtime_id,
            expected_storage_capacity_identity=storage_id,
            expected_config_identities=(
                preparation["source_evidence"]["configs"]
            ),
        )


def test_large_capacity_live_snapshot_cli_binds_every_measured_source() -> None:
    args = live_snapshot.parse_args([
        "--project-root", "/tmp/execution",
        "--runtime-selection", "/evidence/runtime.json",
        "--expected-runtime-selection-sha256", "1" * 64,
        "--storage-capacity", "/evidence/storage.json",
        "--expected-storage-capacity-sha256", "2" * 64,
        "--cofitok-config", "/configs/cofitok.json",
        "--expected-cofitok-config-sha256", "3" * 64,
        "--dense-config", "/configs/dense.json",
        "--expected-dense-config-sha256", "4" * 64,
        "--cofitok-run-dir", f"{OUTPUT_ROOT}/training/cofitok",
        "--dense-run-dir", f"{OUTPUT_ROOT}/training/dense_identity",
        "--benchmark-root", "/tmp/runtime-benchmark",
        "--output-root", OUTPUT_ROOT,
        "--execution-lock",
        terminal_snr_large_capacity_execution_lock_path(OUTPUT_ROOT),
        "--storage-path", "/root/autodl-tmp/CoFiTok/checkpoints/generation",
        "--output", "/evidence/live.json",
    ])
    assert args.expected_runtime_selection_sha256 == "1" * 64
    assert args.expected_storage_capacity_sha256 == "2" * 64
    assert args.output_root == OUTPUT_ROOT


def test_large_capacity_live_snapshot_gpu_inventory_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter([
        SimpleNamespace(stdout=(
            "0, GPU-0, NVIDIA RTX PRO 6000, 1, 97887, 0\n"
        )),
        SimpleNamespace(stdout=""),
    ])
    monkeypatch.setattr(live_snapshot.subprocess, "run", lambda *_, **__: next(outputs))
    inventory, compute = live_snapshot._gpu_inventory()
    assert inventory == [{
        "index": 0,
        "uuid": "GPU-0",
        "name": "NVIDIA RTX PRO 6000",
        "memory_used_mib": 1,
        "memory_total_mib": 97_887,
        "utilization_percent": 0,
    }]
    assert compute == []


def test_large_capacity_live_snapshot_builder_cannot_launch_training() -> None:
    source = inspect.getsource(live_snapshot)
    assert "train_generation.py" in source
    assert "subprocess.Popen" not in source
    assert "nohup" not in source
    assert "full_300k_launch_allowed" not in source

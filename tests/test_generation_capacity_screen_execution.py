from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    CAPACITY_NAMES,
    PREPARATION_BOUNDARY,
    PREPARATION_ROLE,
    PREPARATION_SCHEMA,
)
from cofitok.generation.capacity_screen_execution import (
    EXECUTION_BOUNDARY,
    LAUNCH_BOUNDARY,
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    STAGE_AUTHORIZATION_ROLE,
    STAGE_AUTHORIZATION_SCHEMA,
    STAGE_AUTHORIZATION_SCOPE,
    STAGE_BOUNDARY,
    build_capacity_screen_execution_authorization,
    build_capacity_screen_launch_receipt,
    validate_capacity_screen_launch_receipt_contract,
)
from scripts.run_generation_capacity_screen import parse_args
import scripts.run_generation_capacity_screen as capacity_controller


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}
ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1"
LOCK = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ".capacity_qualification_v1.capacity_screen_execution.lock"
)
CONTROL = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ".capacity_qualification_v1.capacity_screen_control"
)
BENCHMARK = f"{CONTROL}/runtime_benchmarks"


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 10,
        "sha256": (name.encode().hex() + "0" * 64)[:64],
    }


def _preparation() -> dict[str, object]:
    return {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "capacity_screen_prepared",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation_git": copy.deepcopy(GIT),
        "source_evidence": {},
        "selection": {
            "dataset": "imagenet_256",
            "output_root": ROOT,
            "configured_training_horizon": 100_000,
            "stop_after_step": 10_000,
            "effective_batch_size": 64,
            "images_seen_per_arm": 640_000,
            "fresh_initialization_required_for_all_arms": True,
            "fresh_training_arms": list(ARM_NAMES),
            "capacity_intervention": ["model.base_channels"],
            "base_channels": {"base128": 128, "base256": 256},
            "parameter_counts": {
                arm: ARM_SPECS[arm]["parameter_count"] for arm in ARM_NAMES
            },
        },
        "configs": {arm: _identity(f"config_{arm}") for arm in ARM_NAMES},
        "pair_validations": {
            capacity: _identity(f"pair_{capacity}") for capacity in CAPACITY_NAMES
        },
        "validated_pairs": {},
        "within_method_capacity_differences": {},
        "evaluation_contract": {
            "arms": list(ARM_NAMES),
            "checkpoint_step": 10_000,
            "weights": "ema",
            "sampler": "ddim",
            "sample_steps": 100,
            "samples_per_arm": 1_000,
            "sampling_batch_size": 4,
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "eta": 0.0,
            "precision": "bf16",
            "seed": 0,
            "class_schedule": "balanced_modulo",
            "fixed_random_stream_across_arms": True,
            "fid_is_precision_recall_required": True,
            "class_fidelity_required": True,
            "checkpoint_evaluation_required_for_all_arms": True,
            "checkpoint_evaluation_images_per_arm": 256,
            "checkpoint_evaluation_timestep": 500,
            "cofitok_random_orders": 4,
            "dense_random_orders": 0,
            "rollout_required_for_all_arms": True,
            "rollout_images_per_arm": 64,
            "rollout_batch_size": 2,
            "rollout_seed": 2029,
            "primary_estimand": (
                "difference_in_differences_base256_minus_base128_"
                "by_cofitok_minus_dense"
            ),
        },
        "decision_contract": {
            "screen_may_only_prepare_confirmation": True,
            "confirmation_requires_separate_exact_authorization": True,
            "confirmation_samples_per_arm": 10_000,
            "screen_cannot_authorize_full_300k": True,
            "full_300k_requires_later_confirmatory_noncollapse_gate": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }


def _stage_authorization(preparation_id: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "selection": {
            "preparation": preparation_id,
            "execution_checkout": copy.deepcopy(GIT),
            "output_root": ROOT,
            "fresh_training_arms": list(ARM_NAMES),
            "configured_training_horizon": 100_000,
            "intentional_stop_step": 10_000,
            "effective_batch_size": 64,
            "screen_samples_per_arm": 1_000,
            "screen_sample_steps": 100,
        },
        "approval_record": {
            "approved_by": "user",
            "approved_at": "2026-09-02T00:00:00+08:00",
            "source_instruction": "complete the capacity qualification",
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }


def _authorization() -> tuple[dict[str, object], dict[str, object]]:
    preparation = _preparation()
    preparation_id = _identity("preparation")
    config_ids = copy.deepcopy(preparation["configs"])
    authorization = build_capacity_screen_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_authorization=_stage_authorization(preparation_id),
        stage_authorization_identity=_identity("stage_authorization"),
        execution_checkout=GIT,
        config_identities=config_ids,
        output_root=ROOT,
    )
    return authorization, preparation_id


def _runtime(run_dirs: dict[str, str]) -> dict[str, object]:
    config_ids = _preparation()["configs"]
    base256_runs = [run_dirs["base256_cofitok"], run_dirs["base256_dense_identity"]]
    config_sha = {
        "cofitok": config_ids["base256_cofitok"]["sha256"],
        "dense_identity": config_ids["base256_dense_identity"]["sha256"],
    }
    return {
        "status": "selected",
        "git_revision": GIT["revision"],
        "config_sha256": config_sha,
        "benchmark_root": BENCHMARK,
        "runtime_environment_sha256": "c" * 64,
        "dataset_identity_sha256": "e" * 64,
        "selection_sha256": "d" * 64,
        "selected": {
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 16,
        },
        "selection_lock": {
            "training_run_dirs": base256_runs,
            "training_target_steps": 100_000,
            "expected_effective_batch_size": 64,
            "git": {
                "revision": GIT["revision"],
                "branch": GIT["branch"],
                "tracked_dirty": False,
            },
            "config_sha256": config_sha,
            "benchmark_root": BENCHMARK,
        },
    }


def _storage() -> dict[str, object]:
    return {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "status": "pass",
        "filesystem": {
            "path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
            "free_bytes": 500 * 1024**3,
        },
        "plan": {
            "sample_count": 4_000,
            "checkpoint_count": 8,
            "required_free_bytes": 100 * 1024**3,
        },
        "headroom_bytes": 400 * 1024**3,
    }


def _live() -> dict[str, object]:
    return {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": copy.deepcopy(GIT),
        "gpu_inventory": [
            {
                "memory_used_mib": 0,
                "memory_total_mib": 97_887,
                "utilization_percent": 0,
            }
        ],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root": ROOT,
        "execution_lock": LOCK,
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 500 * 1024**3,
        "runtime_environment_sha256": "c" * 64,
        "dataset_identity_sha256": "e" * 64,
        "captured_at": "2026-09-03T00:00:00+00:00",
    }


def test_builds_exact_four_arm_launch_receipt() -> None:
    authorization, preparation_id = _authorization()
    run_dirs = {arm: f"{ROOT}/training/{arm}" for arm in ARM_NAMES}
    report = build_capacity_screen_launch_receipt(
        preparation=_preparation(),
        preparation_identity=preparation_id,
        execution_authorization=authorization,
        execution_authorization_identity=_identity("authorization"),
        runtime_selection=_runtime(run_dirs),
        runtime_selection_identity=_identity("runtime"),
        storage_capacity=_storage(),
        storage_capacity_identity=_identity("storage"),
        live_snapshot=_live(),
        live_snapshot_identity=_identity("live"),
        config_identities=_preparation()["configs"],
        execution_checkout=GIT,
        output_root=ROOT,
        execution_lock=LOCK,
        run_dirs=run_dirs,
        benchmark_root=BENCHMARK,
        training_state_absent_at_launch=True,
    )

    assert report["runtime_selection"]["effective_batch_size"] == 64
    assert report["run_dirs"] == run_dirs
    assert report["authorization_boundary"] == LAUNCH_BOUNDARY
    assert validate_capacity_screen_launch_receipt_contract(
        report, expected_execution_checkout=GIT
    ) == report


def test_stage_authorization_cannot_enable_300k() -> None:
    preparation = _preparation()
    preparation_id = _identity("preparation")
    stage = _stage_authorization(preparation_id)
    stage["authorization_boundary"]["full_300k_launch_allowed"] = True

    with pytest.raises(ValueError, match="stage authorization differs"):
        build_capacity_screen_execution_authorization(
            preparation=preparation,
            preparation_identity=preparation_id,
            stage_authorization=stage,
            stage_authorization_identity=_identity("stage"),
            execution_checkout=GIT,
            config_identities=preparation["configs"],
            output_root=ROOT,
        )


def test_launch_rejects_changed_runtime_environment() -> None:
    authorization, preparation_id = _authorization()
    run_dirs = {arm: f"{ROOT}/training/{arm}" for arm in ARM_NAMES}
    live = _live()
    live["runtime_environment_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="runtime environment changed"):
        build_capacity_screen_launch_receipt(
            preparation=_preparation(),
            preparation_identity=preparation_id,
            execution_authorization=authorization,
            execution_authorization_identity=_identity("authorization"),
            runtime_selection=_runtime(run_dirs),
            runtime_selection_identity=_identity("runtime"),
            storage_capacity=_storage(),
            storage_capacity_identity=_identity("storage"),
            live_snapshot=live,
            live_snapshot_identity=_identity("live"),
            config_identities=_preparation()["configs"],
            execution_checkout=GIT,
            output_root=ROOT,
            execution_lock=LOCK,
            run_dirs=run_dirs,
            benchmark_root=BENCHMARK,
            training_state_absent_at_launch=True,
        )


def test_launch_rejects_existing_training_state() -> None:
    authorization, preparation_id = _authorization()
    run_dirs = {arm: f"{ROOT}/training/{arm}" for arm in ARM_NAMES}

    with pytest.raises(ValueError, match="absent initial training state"):
        build_capacity_screen_launch_receipt(
            preparation=_preparation(),
            preparation_identity=preparation_id,
            execution_authorization=authorization,
            execution_authorization_identity=_identity("authorization"),
            runtime_selection=_runtime(run_dirs),
            runtime_selection_identity=_identity("runtime"),
            storage_capacity=_storage(),
            storage_capacity_identity=_identity("storage"),
            live_snapshot=_live(),
            live_snapshot_identity=_identity("live"),
            config_identities=_preparation()["configs"],
            execution_checkout=GIT,
            output_root=ROOT,
            execution_lock=LOCK,
            run_dirs=run_dirs,
            benchmark_root=BENCHMARK,
            training_state_absent_at_launch=False,
        )


def test_authorization_boundary_is_exact() -> None:
    authorization, _ = _authorization()
    assert authorization["authorization_boundary"] == EXECUTION_BOUNDARY
    assert authorization["authorization_boundary"]["capacity_confirmation_allowed"] is False
    assert authorization["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_launch_rejects_changed_dataset_identity() -> None:
    authorization, preparation_id = _authorization()
    run_dirs = {arm: f"{ROOT}/training/{arm}" for arm in ARM_NAMES}
    live = _live()
    live["dataset_identity_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="dataset identity changed"):
        build_capacity_screen_launch_receipt(
            preparation=_preparation(),
            preparation_identity=preparation_id,
            execution_authorization=authorization,
            execution_authorization_identity=_identity("authorization"),
            runtime_selection=_runtime(run_dirs),
            runtime_selection_identity=_identity("runtime"),
            storage_capacity=_storage(),
            storage_capacity_identity=_identity("storage"),
            live_snapshot=live,
            live_snapshot_identity=_identity("live"),
            config_identities=_preparation()["configs"],
            execution_checkout=GIT,
            output_root=ROOT,
            execution_lock=LOCK,
            run_dirs=run_dirs,
            benchmark_root=BENCHMARK,
            training_state_absent_at_launch=True,
        )


def test_launch_rejects_non_sibling_execution_lock() -> None:
    authorization, preparation_id = _authorization()
    run_dirs = {arm: f"{ROOT}/training/{arm}" for arm in ARM_NAMES}

    with pytest.raises(ValueError, match="exact sibling lock"):
        build_capacity_screen_launch_receipt(
            preparation=_preparation(),
            preparation_identity=preparation_id,
            execution_authorization=authorization,
            execution_authorization_identity=_identity("authorization"),
            runtime_selection=_runtime(run_dirs),
            runtime_selection_identity=_identity("runtime"),
            storage_capacity=_storage(),
            storage_capacity_identity=_identity("storage"),
            live_snapshot=_live(),
            live_snapshot_identity=_identity("live"),
            config_identities=_preparation()["configs"],
            execution_checkout=GIT,
            output_root=ROOT,
            execution_lock=f"{ROOT}/wrong.lock",
            run_dirs=run_dirs,
            benchmark_root=BENCHMARK,
            training_state_absent_at_launch=True,
        )


def test_capacity_screen_controller_exposes_all_four_arm_configs() -> None:
    args = parse_args(
        [
            "--project-root", "/tmp/project",
            "--preparation", "/tmp/preparation.json",
            "--expected-preparation-sha256", "a" * 64,
            "--authorization", "/tmp/authorization.json",
            "--expected-authorization-sha256", "b" * 64,
            "--launch-receipt", "/tmp/launch.json",
            "--expected-launch-receipt-sha256", "c" * 64,
            "--runtime-selection", "/tmp/runtime.json",
            "--expected-runtime-selection-sha256", "d" * 64,
            "--live-snapshot", "/tmp/live.json",
            "--expected-live-snapshot-sha256", "e" * 64,
            "--base128-cofitok-config", "/tmp/base128_cofitok.json",
            "--base128-dense-identity-config", "/tmp/base128_dense.json",
            "--base256-cofitok-config", "/tmp/base256_cofitok.json",
            "--base256-dense-identity-config", "/tmp/base256_dense.json",
            "--output-root", "/tmp/output",
            "--real-dir", "/tmp/real",
            "--classifier-checkpoint", "/tmp/classifier.pt",
            "--cache-root", "/tmp/cache",
        ]
    )

    assert args.base128_cofitok_config == Path("/tmp/base128_cofitok.json")
    assert args.base128_dense_identity_config == Path("/tmp/base128_dense.json")
    assert args.base256_cofitok_config == Path("/tmp/base256_cofitok.json")
    assert args.base256_dense_identity_config == Path("/tmp/base256_dense.json")


def test_capacity_screen_resume_root_is_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "screen"
    root.mkdir()
    authorization = {"path": "/tmp/auth", "bytes": 1, "sha256": "a" * 64}
    launch = {"path": "/tmp/launch", "bytes": 1, "sha256": "b" * 64}
    status_path = tmp_path / "control" / "controller_status.json"
    capacity_controller._write_status(
        status_path,
        status="running",
        stage="training_base128_cofitok",
        detail="test",
        authorization=authorization,
        launch_receipt=launch,
    )

    capacity_controller._validate_root_for_resume(
        root,
        status_path=status_path,
        authorization_identity=authorization,
        launch_identity=launch,
    )

    (root / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected entries"):
        capacity_controller._validate_root_for_resume(
            root,
            status_path=status_path,
            authorization_identity=authorization,
            launch_identity=launch,
        )


def test_capacity_screen_resume_rollout_requires_exact_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint_step_00010000.pt"
    checkpoint.write_bytes(b"checkpoint")
    report_path = tmp_path / "rollout_stability_report.json"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    expected_config = {"resolved": True}
    monkeypatch.setattr(
        capacity_controller,
        "verify_training_checkpoint",
        lambda _: {"checkpoint_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        capacity_controller,
        "_resolved_config",
        lambda *_args, **_kwargs: expected_config,
    )
    git = {"revision": "a" * 40, "branch": "screen", "tracked_dirty": False}
    protocol = {
        "num_images": 64,
        "batch_size": 2,
        "teacher_timesteps": capacity_controller.ROLLOUT_TEACHER_TIMESTEPS,
        "sample_steps": capacity_controller.SAMPLE_STEPS,
        "guidance_scale": 1.5,
        "teacher_guidance_scale": 1.0,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "clip_x0": True,
        "precision": "bf16",
        "seed": capacity_controller.ROLLOUT_SEED,
    }
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": checkpoint.resolve().as_posix(),
                "checkpoint_sha256": "c" * 64,
                "checkpoint_step": capacity_controller.STOP_STEP,
                "weights": "ema",
                "protocol": protocol,
                "config": expected_config,
                "git": git,
            }
        ),
        encoding="utf-8",
    )

    capacity_controller._validate_resume_rollout(
        report_path=report_path,
        checkpoint=checkpoint,
        config_path=config_path,
        arm="base128_cofitok",
        execution_git=git,
    )

    changed = json.loads(report_path.read_text(encoding="utf-8"))
    changed["protocol"]["sample_steps"] = 50
    report_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="completed rollout binding differs"):
        capacity_controller._validate_resume_rollout(
            report_path=report_path,
            checkpoint=checkpoint,
            config_path=config_path,
            arm="base128_cofitok",
            execution_git=git,
        )

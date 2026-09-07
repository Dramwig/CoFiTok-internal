from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofitok.generation import terminal_snr_screen_execution as execution
from cofitok.generation.terminal_snr_screen import ARM_NAMES
from scripts import run_generation_capacity_screen as shared_controller
from scripts import run_generation_terminal_snr_screen as controller
from scripts import (
    validate_generation_terminal_snr_screen_stage_authorization as stage_validator_cli,
)


ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1"
LOCK = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ".terminal_snr_endpoint_screen_v1.terminal_snr_screen_execution.lock"
)
CONTROL = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ".terminal_snr_endpoint_screen_v1.terminal_snr_screen_control"
)
BENCHMARK = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ".terminal_snr_endpoint_screen_v1.terminal_snr_runtime_benchmarks"
)
GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/terminal-snr",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 10,
        "sha256": (name.encode().hex() + "0" * 64)[:64],
    }


def _preparation() -> dict[str, object]:
    return {
        "selection": {
            "output_root": ROOT,
            "fresh_training_arms": list(ARM_NAMES),
            "fresh_initialization_required_for_all_arms": True,
            "single_scientific_config_field": (
                "diffusion.cosine_endpoint_fraction"
            ),
            "configured_training_horizon": 100_000,
            "stop_after_step": 10_000,
            "effective_batch_size": 64,
            "images_seen_per_arm": 640_000,
        },
        "configs": {arm: _identity(f"config_{arm}") for arm in ARM_NAMES},
        "evaluation_contract": {
            "arms": list(ARM_NAMES),
            "checkpoint_step": 10_000,
            "samples_per_arm": 1_000,
            "sample_steps": 100,
            "seed": 2027,
            "weights": "ema",
            "fixed_random_stream_across_arms": True,
            "fid_is_precision_recall_required": True,
            "class_fidelity_required": True,
            "checkpoint_evaluation_required_for_all_arms": True,
            "rollout_required_for_all_arms": True,
        },
        "thresholds": copy.deepcopy(execution.SCREEN_THRESHOLDS),
    }


def _stage(preparation_id: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": execution.STAGE_AUTHORIZATION_SCHEMA,
        "role": execution.STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": execution.STAGE_AUTHORIZATION_SCOPE,
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
            "single_scientific_config_field": (
                "diffusion.cosine_endpoint_fraction"
            ),
        },
        "approval_record": {
            "approved_by": "user",
            "approved_at": "2026-09-07T00:00:00+08:00",
            "source_instruction": "继续修复再推进",
        },
        "authorization_boundary": copy.deepcopy(execution.STAGE_BOUNDARY),
    }


def _authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    preparation = _preparation()
    monkeypatch.setattr(
        execution,
        "validate_terminal_snr_screen_preparation_contract",
        lambda report, **_: copy.deepcopy(report),
    )
    preparation_id = _identity("preparation")
    authorization = execution.build_terminal_snr_screen_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_authorization=_stage(preparation_id),
        stage_authorization_identity=_identity("stage"),
        execution_checkout=GIT,
        config_identities=preparation["configs"],
        output_root=ROOT,
    )
    return authorization, preparation_id


def _runtime(run_dirs: dict[str, str]) -> dict[str, object]:
    configs = _preparation()["configs"]
    config_sha = {
        "cofitok": configs["endpoint0975_cofitok"]["sha256"],
        "dense_identity": configs["endpoint0975_dense_identity"]["sha256"],
    }
    endpoint_runs = [
        run_dirs["endpoint0975_cofitok"],
        run_dirs["endpoint0975_dense_identity"],
    ]
    return {
        "status": "selected",
        "git_revision": GIT["revision"],
        "config_sha256": config_sha,
        "benchmark_root": BENCHMARK,
        "runtime_environment_sha256": "c" * 64,
        "dataset_identity_sha256": "d" * 64,
        "selection_sha256": "e" * 64,
        "selected": {
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 16,
        },
        "selection_lock": {
            "training_run_dirs": endpoint_runs,
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
        "schema_version": execution.LIVE_SNAPSHOT_SCHEMA,
        "role": execution.LIVE_SNAPSHOT_ROLE,
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
        "dataset_identity_sha256": "d" * 64,
        "captured_at": "2026-09-07T00:00:00+00:00",
    }


def _launch_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    authorization, preparation_id = _authorization(monkeypatch)
    run_dirs = {arm: f"{ROOT}/training/{arm}" for arm in ARM_NAMES}
    return {
        "preparation": _preparation(),
        "preparation_identity": preparation_id,
        "execution_authorization": authorization,
        "execution_authorization_identity": _identity("authorization"),
        "runtime_selection": _runtime(run_dirs),
        "runtime_selection_identity": _identity("runtime"),
        "storage_capacity": _storage(),
        "storage_capacity_identity": _identity("storage"),
        "live_snapshot": _live(),
        "live_snapshot_identity": _identity("live"),
        "config_identities": _preparation()["configs"],
        "execution_checkout": GIT,
        "output_root": ROOT,
        "execution_lock": LOCK,
        "run_dirs": run_dirs,
        "benchmark_root": BENCHMARK,
        "training_state_absent_at_launch": True,
    }


def test_builds_exact_terminal_snr_launch_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = execution.build_terminal_snr_screen_launch_receipt(
        **_launch_inputs(monkeypatch)
    )

    assert report["stage"] == "terminal_snr_screen"
    assert report["runtime_selection"]["effective_batch_size"] == 64
    assert report["authorization_boundary"] == execution.LAUNCH_BOUNDARY
    assert execution.validate_terminal_snr_screen_launch_receipt_contract(
        report, expected_execution_checkout=GIT
    ) == report


def test_launch_receipt_physically_replays_every_bound_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _launch_inputs(monkeypatch)
    report = execution.build_terminal_snr_screen_launch_receipt(**inputs)
    stage = inputs["execution_authorization"]["validated_stage_authorization"]
    payloads = {
        inputs["preparation_identity"]["path"]: inputs["preparation"],
        inputs["execution_authorization_identity"]["path"]: inputs[
            "execution_authorization"
        ],
        inputs["runtime_selection_identity"]["path"]: inputs[
            "runtime_selection"
        ],
        inputs["storage_capacity_identity"]["path"]: inputs["storage_capacity"],
        inputs["live_snapshot_identity"]["path"]: inputs["live_snapshot"],
        inputs["execution_authorization"]["stage_authorization"]["path"]: stage,
    }
    identities = {
        descriptor["path"]: descriptor
        for descriptor in (
            inputs["preparation_identity"],
            inputs["execution_authorization_identity"],
            inputs["runtime_selection_identity"],
            inputs["storage_capacity_identity"],
            inputs["live_snapshot_identity"],
            inputs["execution_authorization"]["stage_authorization"],
            *inputs["config_identities"].values(),
        )
    }
    monkeypatch.setattr(
        execution,
        "artifact_identity",
        lambda path: copy.deepcopy(identities[str(path)]),
    )
    monkeypatch.setattr(
        execution,
        "read_object",
        lambda path, **_: copy.deepcopy(payloads[str(path)]),
    )

    assert execution.validate_terminal_snr_screen_launch_receipt_physical(
        report, expected_execution_checkout=GIT
    ) == report


def test_launch_contract_rejects_tampered_embedded_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = execution.build_terminal_snr_screen_launch_receipt(
        **_launch_inputs(monkeypatch)
    )
    report["authorization"]["thresholds"][
        "both_methods_min_relative_fid_improvement"
    ] = 0.0

    with pytest.raises(ValueError, match="authorization contract differs"):
        execution.validate_terminal_snr_screen_launch_receipt_contract(
            report, expected_execution_checkout=GIT
        )


def test_stage_authorization_cannot_enable_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_id = _identity("preparation")
    stage = _stage(preparation_id)
    stage["authorization_boundary"]["frozen_confirmation_allowed"] = True
    monkeypatch.setattr(
        execution,
        "validate_terminal_snr_screen_preparation_contract",
        lambda report, **_: copy.deepcopy(report),
    )

    with pytest.raises(ValueError, match="stage authorization differs"):
        execution.build_terminal_snr_screen_execution_authorization(
            preparation=_preparation(),
            preparation_identity=preparation_id,
            stage_authorization=stage,
            stage_authorization_identity=_identity("stage"),
            execution_checkout=GIT,
            config_identities=_preparation()["configs"],
            output_root=ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_environment_sha256", "f" * 64, "runtime environment changed"),
        ("dataset_identity_sha256", "f" * 64, "dataset identity changed"),
    ],
)
def test_launch_rejects_changed_live_identity(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    inputs = _launch_inputs(monkeypatch)
    inputs["live_snapshot"][field] = value

    with pytest.raises(ValueError, match=message):
        execution.build_terminal_snr_screen_launch_receipt(**inputs)


def test_launch_rejects_existing_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _launch_inputs(monkeypatch)
    inputs["training_state_absent_at_launch"] = False

    with pytest.raises(ValueError, match="absent initial training state"):
        execution.build_terminal_snr_screen_launch_receipt(**inputs)


def test_launch_rejects_wrong_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _launch_inputs(monkeypatch)
    inputs["execution_lock"] = f"{ROOT}/wrong.lock"

    with pytest.raises(ValueError, match="exact sibling lock"):
        execution.build_terminal_snr_screen_launch_receipt(**inputs)


def test_exact_control_paths() -> None:
    assert execution.terminal_snr_screen_execution_lock_path(ROOT) == LOCK
    assert execution.terminal_snr_screen_control_root(ROOT) == CONTROL
    assert execution.terminal_snr_screen_benchmark_root(ROOT) == BENCHMARK


def test_controller_exposes_all_terminal_snr_configs() -> None:
    args = controller.parse_args([
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
        "--control-cofitok-config", "/tmp/control_cofitok.json",
        "--control-dense-identity-config", "/tmp/control_dense.json",
        "--endpoint0975-cofitok-config", "/tmp/endpoint_cofitok.json",
        "--endpoint0975-dense-identity-config", "/tmp/endpoint_dense.json",
        "--output-root", "/tmp/output",
        "--real-dir", "/tmp/real",
        "--classifier-checkpoint", "/tmp/classifier.pt",
        "--cache-root", "/tmp/cache",
    ])

    for arm in ARM_NAMES:
        assert isinstance(getattr(args, f"{arm}_config"), type(args.project_root))


def test_controller_bindings_are_restorable() -> None:
    original_names = shared_controller.ARM_NAMES
    original_seed = shared_controller.SAMPLE_SEED

    saved = controller._configure_core()
    try:
        assert shared_controller.ARM_NAMES == ARM_NAMES
        assert shared_controller.SAMPLE_SEED == 2027
        assert shared_controller.ROLE == controller.ROLE
    finally:
        controller._restore_core(saved)

    assert shared_controller.ARM_NAMES is original_names
    assert shared_controller.SAMPLE_SEED == original_seed


def test_stage_validator_maps_builder_root_to_expected_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        stage_authorization=Path("/tmp/stage.json"),
        expected_stage_authorization_sha256="a" * 64,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(stage_validator_cli, "parse_validate_stage_args", lambda: args)
    monkeypatch.setattr(
        stage_validator_cli,
        "reject_symlink_chain",
        lambda path, **_: path,
    )
    monkeypatch.setattr(stage_validator_cli, "file_sha256", lambda _: "a" * 64)
    monkeypatch.setattr(stage_validator_cli, "read_object", lambda *_args, **_: {})
    monkeypatch.setattr(
        stage_validator_cli,
        "stage_kwargs",
        lambda _: {
            "preparation_identity": _identity("preparation"),
            "execution_checkout": GIT,
            "output_root": ROOT,
        },
    )

    def validate(_actual: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        stage_validator_cli,
        "validate_terminal_snr_screen_stage_authorization",
        validate,
    )
    stage_validator_cli.main()

    assert captured["expected_output_root"] == ROOT
    assert "output_root" not in captured

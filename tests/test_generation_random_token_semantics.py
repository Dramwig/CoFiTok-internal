from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import ModelConfig
from cofitok.inference_replay import file_identity
from cofitok.models.synthesis import build_synthesis_bank
from cofitok.reporting import write_json_report
from scripts.evaluate_generation_random_token_semantics import (
    COMPONENT_PANEL_FILENAME,
    EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT,
    MATCHED_FACTORIZATION_SCOPE,
    PREFIX_PANEL_FILENAME,
    REPORT_ROLE,
    REPORT_SCHEMA_VERSION,
    evaluate_random_token_synthesis,
    require_gpu_idle,
    factorization_supervisor_status_identity,
    parse_prefix_budgets,
    prepare_output_directory,
    terminal_quality_result_identity,
    terminal_system_guard_identity,
    validate_loaded_target,
    validate_completed_report,
)


RUNBOOK = Path("artifacts/runbooks/generation_random_token_semantic_visual_100k_v1.sh")


def _model_config() -> ModelConfig:
    return ModelConfig(
        image_channels=3,
        image_size=8,
        token_count=2,
        token_channels=2,
        base_channels=8,
        predictor_type="scalable_unet",
        predictor_depth=1,
        predictor_use_feedback=False,
        synthesis_mode="fixed_basis",
        synthesis_kernel_size=1,
        gamma_mode="fixed_one",
        token_channel_schedule=[1, 2],
        token_spatial_strides=[2, 1],
    )


def _synthesis(config: ModelConfig) -> torch.nn.Module:
    return build_synthesis_bank(
        synthesis_mode=config.synthesis_mode,
        token_count=config.token_count,
        token_channels=config.token_channels,
        image_channels=config.image_channels,
        kernel_size=config.synthesis_kernel_size,
        gamma_mode=config.gamma_mode,
        token_strides=config.synthesis_token_strides,
        active_token_channels=config.synthesis_active_token_channels,
        token_channel_schedule=config.token_channel_schedule,
        output_size=config.image_size,
    )


def test_prefix_budget_parser_is_unique_sorted_and_bounded() -> None:
    assert parse_prefix_budgets("2,1,2", 2) == [1, 2]
    with pytest.raises(ValueError, match="must be in"):
        parse_prefix_budgets("0,2", 2)


def test_gpu_idle_preflight_is_fail_closed(monkeypatch) -> None:
    def idle_run(*args, **kwargs):
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(
        "scripts.evaluate_generation_random_token_semantics.subprocess.run",
        idle_run,
    )
    assert require_gpu_idle()["gpu_idle_before_checkpoint_load"] is True

    def busy_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="1234, /opt/python, 4096, GPU-example\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(
        "scripts.evaluate_generation_random_token_semantics.subprocess.run",
        busy_run,
    )
    with pytest.raises(RuntimeError, match="GPU is not idle.*1234"):
        require_gpu_idle()


def test_loaded_target_requires_exact_factorized_ema_checkpoint() -> None:
    loaded = SimpleNamespace(
        weights="ema",
        checkpoint_sha256="a" * 64,
        checkpoint_step=100_000,
        config=SimpleNamespace(model=SimpleNamespace(synthesis_mode="fixed_basis")),
    )
    validate_loaded_target(
        loaded,
        requested_weights="ema",
        expected_checkpoint_sha256="a" * 64,
        expected_checkpoint_step=100_000,
    )
    with pytest.raises(ValueError, match="exact requested"):
        validate_loaded_target(
            loaded,
            requested_weights="ema",
            expected_checkpoint_sha256="b" * 64,
            expected_checkpoint_step=100_000,
        )
    loaded.config.model.synthesis_mode = "dense_identity"
    with pytest.raises(ValueError, match="factorized"):
        validate_loaded_target(
            loaded,
            requested_weights="ema",
            expected_checkpoint_sha256="a" * 64,
            expected_checkpoint_step=100_000,
        )


def test_terminal_quality_result_binds_exact_checkpoint_and_non_authorizing_boundary(
    tmp_path,
) -> None:
    result_path = tmp_path / "quality_bridge_result.json"
    checkpoint = {
        "path": (tmp_path / "checkpoint.pt").resolve().as_posix(),
        "sha256": "a" * 64,
        "step": 100_000,
    }
    result = {
        "status": "completed",
        "role": "stability_full_data_quality_bridge_result",
        "stage": "stability_quality_bridge",
        "git": {
            "revision": "b" * 40,
            "branch": "scale/generation-stability-quality-bridge-100k",
            "tracked_dirty": False,
        },
        "terminal": {
            "methods": {
                "cofitok": {
                    "checkpoint": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_step": checkpoint["step"],
                    "weights": "ema",
                }
            }
        },
        "authorization_boundary": {
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_authorization_allowed": False,
        },
    }
    write_json_report(result_path, result)
    identity = terminal_quality_result_identity(
        result_path,
        checkpoint=checkpoint,
        expected_training_revision="b" * 40,
        expected_training_branch="scale/generation-stability-quality-bridge-100k",
    )
    assert identity["path"] == result_path.resolve().as_posix()
    assert len(identity["sha256"]) == 64
    result["authorization_boundary"]["full_300k_launch_allowed"] = True
    write_json_report(result_path, result)
    with pytest.raises(ValueError, match="authorization boundary"):
        terminal_quality_result_identity(
            result_path,
            checkpoint=checkpoint,
            expected_training_revision="b" * 40,
            expected_training_branch="scale/generation-stability-quality-bridge-100k",
        )


def test_terminal_system_sources_must_be_complete_and_non_authorizing(tmp_path) -> None:
    quality_identity = {
        "path": (tmp_path / "quality.json").resolve().as_posix(),
        "bytes": 1,
        "sha256": "a" * 64,
    }
    guard_path = tmp_path / "terminal_guard.json"
    guard = {
        "schema_version": 1,
        "role": "generation_terminal_system_claim_guard",
        "status": "hold",
        "sources": {"quality_bridge_result": quality_identity},
        "evidence": {
            "class_fidelity_classifier_integrity": {"status": "verified"},
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "class_fidelity_classifier_physical_integrity_verified": True,
            "larger_training_launch_allowed": False,
            "inference_export_authorization_allowed": False,
            "release_authorization_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
        },
    }
    write_json_report(guard_path, guard)
    assert terminal_system_guard_identity(
        guard_path,
        terminal_quality_result=quality_identity,
    )["path"] == guard_path.resolve().as_posix()

    missing_policy = copy.deepcopy(guard)
    missing_policy["claim_policy"].pop(
        "class_fidelity_classifier_physical_integrity_verified"
    )
    write_json_report(guard_path, missing_policy)
    with pytest.raises(ValueError, match="terminal system guard contract differs"):
        terminal_system_guard_identity(
            guard_path,
            terminal_quality_result=quality_identity,
        )

    unverified_evidence = copy.deepcopy(guard)
    unverified_evidence["evidence"]["class_fidelity_classifier_integrity"][
        "status"
    ] = "unverified"
    write_json_report(guard_path, unverified_evidence)
    with pytest.raises(ValueError, match="terminal system guard contract differs"):
        terminal_system_guard_identity(
            guard_path,
            terminal_quality_result=quality_identity,
        )

    write_json_report(guard_path, guard)

    followup_path = tmp_path / "followup.json"
    write_json_report(followup_path, {"status": "selected"})
    followup_identity = file_identity(followup_path)

    source_binding_path = tmp_path / "source_binding.json"
    source_binding = {
        "schema_version": 1,
        "role": "generation_factorization_quality_regression_source_binding",
        "status": "verified",
        "scope": MATCHED_FACTORIZATION_SCOPE,
        "sources": {"followup_decision": followup_identity},
        "selected_route": {
            "id": "run_matched_factorization_quality_regression_probe",
            "category": "matched_quality_regression",
            "failed_checks": ["matched_fid_tolerance"],
        },
    }
    write_json_report(source_binding_path, source_binding)
    source_binding_identity = file_identity(source_binding_path)

    authorization_path = tmp_path / "execution_authorization.json"
    authorization = {
        "schema_version": 1,
        "role": "generation_factorization_quality_regression_execution_authorization",
        "status": "authorized",
        "scope": MATCHED_FACTORIZATION_SCOPE,
        "source_binding": source_binding_identity,
        "git": {
            "revision": EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT["revision"],
            "branch": EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT["branch"],
            "tracked_dirty": False,
        },
        "authorization_boundary": {
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }
    write_json_report(authorization_path, authorization)
    authorization_identity = file_identity(authorization_path)

    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic = {
        "schema_version": 1,
        "role": "generation_factorization_quality_regression_diagnostic",
        "status": "completed",
        "scope": MATCHED_FACTORIZATION_SCOPE,
        "sources": {"execution_authorization": authorization_identity},
        "claim_boundary": {
            "training_launch_allowed": False,
            "quality_advantage_claim_allowed": False,
        },
    }
    write_json_report(diagnostic_path, diagnostic)

    supervisor_path = tmp_path / "supervisor.json"
    deployment_path = tmp_path / "deployment.json"
    deployment = {
        "schema_version": 1,
        "role": "generation_factorization_quality_regression_supervisor_deployment",
        "status": "pass",
        "control": {"checkout": EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT},
        "targets": {"status_output": supervisor_path.resolve().as_posix()},
    }
    write_json_report(deployment_path, deployment)
    supervisor = {
        "schema_version": 1,
        "role": "generation_factorization_quality_regression_supervisor",
        "status": "completed",
        "detail": "matched_factorization_quality_regression_diagnostic_completed",
        "child_pid": 12345,
        "project": EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT["path"],
        "sources": {
            "deployment_receipt": file_identity(deployment_path),
            "quality_bridge_followup_decision": followup_identity,
            "source_binding": source_binding_identity,
            "execution_authorization": authorization_identity,
            "diagnostic_report": file_identity(diagnostic_path),
        },
        "authorization_boundary": {
            "training_launch_allowed": False,
            "checkpoint_promotion_allowed": False,
            "followup_experiment_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
    }
    write_json_report(supervisor_path, supervisor)
    assert factorization_supervisor_status_identity(supervisor_path)["path"] == (
        supervisor_path.resolve().as_posix()
    )
    supervisor["detail"] = "factorization_quality_regression_diagnostic_not_selected"
    supervisor["child_pid"] = None
    write_json_report(supervisor_path, supervisor)
    with pytest.raises(ValueError, match="terminal contract"):
        factorization_supervisor_status_identity(supervisor_path)

    supervisor["detail"] = "matched_factorization_quality_regression_diagnostic_completed"
    supervisor["child_pid"] = 12345
    write_json_report(supervisor_path, supervisor)
    diagnostic["claim_boundary"]["quality_advantage_claim_allowed"] = True
    write_json_report(diagnostic_path, diagnostic)
    with pytest.raises(ValueError, match="identity differs"):
        factorization_supervisor_status_identity(supervisor_path)


def test_random_token_diagnostic_audits_fixed_basis_and_writes_panels(tmp_path) -> None:
    config = _model_config()
    synthesis = _synthesis(config).eval()
    diagnostic = evaluate_random_token_synthesis(
        synthesis=synthesis,
        model_config=config,
        device=torch.device("cpu"),
        num_images=3,
        seed=19,
        precision="fp32",
        prefix_budgets=[1, 2],
        clip_sigma=3.0,
        component_panel_path=tmp_path / COMPONENT_PANEL_FILENAME,
        prefix_panel_path=tmp_path / PREFIX_PANEL_FILENAME,
    )

    architecture = diagnostic["synthesis_architecture"]
    numerical = diagnostic["numerical_contract"]
    assert architecture["module_type"] == "FixedBasisSynthesisBank"
    assert architecture["parameter_count"] == 0
    assert architecture["trainable_parameter_count"] == 0
    assert architecture["bias_parameter_names"] == []
    assert architecture["nonlinear_or_attention_modules"] == []
    assert numerical["zero_token_max_abs"] == 0.0
    assert numerical["cross_token_leakage_max_abs"] == 0.0
    assert numerical["linearity_max_abs"] <= 1e-6
    assert diagnostic["token_layout"]["spatial_sizes"] == [4, 8]
    assert len(diagnostic["random_components"]) == 2
    assert set(diagnostic["random_prefixes"]) == {"1", "2"}
    for artifact in diagnostic["artifacts"].values():
        path = Path(artifact["path"])
        assert path.is_file()
        assert artifact["bytes"] == path.stat().st_size
        assert len(artifact["sha256"]) == 64


def test_completed_report_rehashes_bound_panels(tmp_path) -> None:
    component = tmp_path / COMPONENT_PANEL_FILENAME
    prefix = tmp_path / PREFIX_PANEL_FILENAME
    component.write_bytes(b"component")
    prefix.write_bytes(b"prefix")
    source = tmp_path / "source.py"
    quality = tmp_path / "quality.json"
    guard = tmp_path / "guard.json"
    supervisor = tmp_path / "supervisor.json"
    source.write_bytes(b"source")
    quality.write_bytes(b"quality")
    guard.write_bytes(b"guard")
    supervisor.write_bytes(b"supervisor")
    from cofitok.inference_replay import file_identity
    from cofitok.reporting import file_sha256

    artifacts = {
        "components_vs_gaussian": {
            "path": component.resolve().as_posix(),
            "bytes": component.stat().st_size,
            "sha256": file_sha256(component),
        },
        "prefixes_vs_gaussian": {
            "path": prefix.resolve().as_posix(),
            "bytes": prefix.stat().st_size,
            "sha256": file_sha256(prefix),
        },
    }
    manifest = {
        "git": {"revision": "a" * 40},
        "evaluator_source": file_identity(source),
        "checkpoint": {"sha256": "b" * 64},
        "terminal_quality_result": file_identity(quality),
        "terminal_system_guard": file_identity(guard),
        "factorization_supervisor_status": file_identity(supervisor),
        "runtime_preflight": {"gpu_idle_before_checkpoint_load": True},
        "request": {"num_images": 2},
        "output_dir": tmp_path.resolve().as_posix(),
        "authorization_policy": {"non_authorizing": True},
    }
    manifest_identity = {"path": "manifest", "bytes": 1, "sha256": "c" * 64}
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "manifest": manifest_identity,
        "git": manifest["git"],
        "evaluator_source": manifest["evaluator_source"],
        "checkpoint": manifest["checkpoint"],
        "terminal_quality_result": manifest["terminal_quality_result"],
        "terminal_system_guard": manifest["terminal_system_guard"],
        "factorization_supervisor_status": manifest[
            "factorization_supervisor_status"
        ],
        "runtime_preflight": manifest["runtime_preflight"],
        "request": manifest["request"],
        "authorization_policy": manifest["authorization_policy"],
        "diagnostic": {"artifacts": artifacts},
    }
    validate_completed_report(
        report,
        manifest_identity=manifest_identity,
        manifest=manifest,
    )
    component.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact identity differs"):
        validate_completed_report(
            report,
            manifest_identity=manifest_identity,
            manifest=manifest,
        )


def test_output_directory_rejects_unexpected_files(tmp_path) -> None:
    output = tmp_path / "diagnostic"
    output.mkdir()
    write_json_report(output / "unexpected.json", {"status": "stale"})
    with pytest.raises(ValueError, match="unexpected files"):
        prepare_output_directory(output, resume=True)


def test_runbook_is_terminal_idle_and_non_authorizing() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "quality_bridge_result.json" in source
    assert 'result.get("status") != "completed"' in source
    assert 'boundary.get(key) is not False' in source
    assert 'guard.get("status") not in {"pass", "hold"}' in source
    assert "class_fidelity_classifier_physical_integrity_verified" in source
    assert "class_fidelity_classifier_integrity" in source
    assert 'supervisor.get("status") != "completed"' in source
    assert 'supervisor.get("detail") != (' in source
    assert "matched_factorization_quality_regression_diagnostic_completed" in source
    assert "factorization_quality_regression_diagnostic_not_selected" not in source
    assert 'required_sources.issubset(supervisor.get("sources", {}))' in source
    assert "raise SystemExit(75)" in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "GPU is not idle" in source
    assert 'REQUIRED_IDLE_GPU_POLLS="5"' in source
    assert "idle_poll <= REQUIRED_IDLE_GPU_POLLS" in source
    assert "command -v flock" in source
    assert 'exec 8>"${RUNBOOK_LOCK}"' in source
    assert "flock -n 8" in source
    assert "RESUME_ARGS=(--resume)" in source
    assert "--weights ema" in source
    assert "--expected-checkpoint-sha256" in source
    assert "--terminal-quality-result" in source
    assert "--terminal-system-guard" in source
    assert "--factorization-supervisor-status" in source
    assert "b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e" in source
    assert "300k" not in source.lower().replace("full_300k_launch_allowed", "")
    assert "train_generation.py" not in source


def test_entrypoint_validates_terminal_sources_and_gpu_before_model_load() -> None:
    source = Path("scripts/evaluate_generation_random_token_semantics.py").read_text(
        encoding="utf-8"
    )
    run_source = source[source.index("def _run(") : source.index("def main(")]
    load_index = run_source.index("loaded = load_generation_model")
    assert run_source.index("preflight_quality_result = terminal_quality_result_identity") < load_index
    assert run_source.index("preflight_terminal_guard = terminal_system_guard_identity") < load_index
    assert run_source.index("preflight_factorization_status = factorization_supervisor_status_identity") < load_index
    assert run_source.index("runtime_preflight = require_gpu_idle()") < load_index

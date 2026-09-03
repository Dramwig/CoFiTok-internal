from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

import cofitok.generation.exposure_capacity_result as result_builder
import scripts.evaluate_generation_metrics as metrics_evaluator
import scripts.run_generation_exposure_capacity_continuation as controller


def test_result_validation_command_binds_stage_authorization_and_receipt(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        python=Path("python"),
        project_root=tmp_path / "execution",
        source_project_root=tmp_path / "source",
        authorization=tmp_path / "authorization.json",
        expected_authorization_sha256="a" * 64,
        gate=tmp_path / "gate.json",
        expected_gate_sha256="b" * 64,
        preparation=tmp_path / "preparation.json",
        expected_preparation_sha256="c" * 64,
        standing_authorization=tmp_path / "standing.json",
        expected_standing_authorization_sha256="d" * 64,
        stage_authorization=tmp_path / "stage.json",
        expected_stage_authorization_sha256="e" * 64,
        cofitok_config=tmp_path / "cofitok.json",
        dense_config=tmp_path / "dense.json",
        real_dir=tmp_path / "real",
        classifier_checkpoint=tmp_path / "classifier.pth",
        cache_root=tmp_path / "cache",
    )
    root = tmp_path / "output"

    command = [str(value) for value in controller._validation_command(args, root)]

    assert command[command.index("--stage-authorization") + 1] == str(
        args.stage_authorization
    )
    assert command[command.index("--expected-stage-authorization-sha256") + 1] == (
        args.expected_stage_authorization_sha256
    )
    assert command[command.index("--validation-receipt") + 1] == str(
        root / "exposure_capacity_result.validation.json"
    )
    assert command[command.index("--execution-project-root") + 1] == str(
        args.project_root
    )
    assert command[command.index("--validator-project-root") + 1] == str(
        args.project_root
    )


def _controller_status(root: Path, authorization: dict[str, object], *, status: str = "failed") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "controller_status.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": controller.ROLE,
                "status": status,
                "stage": "training_cofitok",
                "authorization": authorization,
                "terminal_status": "hold",
                "generation_advantage_proven": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_allowed": False,
                "release_allowed": False,
                "process_signals_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_claimed_output_root_is_created_inside_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock(_path: Path):
        events.append("lock_enter")
        yield
        events.append("lock_exit")

    monkeypatch.setattr(controller, "_execution_lock", fake_lock)
    output_root = tmp_path / "candidate"
    with controller._claimed_output_root(output_root, tmp_path / "candidate.lock"):
        assert output_root.is_dir()
        assert events == ["lock_enter"]
    assert events == ["lock_enter", "lock_exit"]


def test_authorized_execution_lock_is_read_from_target(tmp_path: Path) -> None:
    output_root = tmp_path / "exposure"
    expected_lock = tmp_path / ".exposure.exposure_execution.lock"
    authorization = {
        "qualification_contract": {"output_root": output_root.as_posix()},
        "target": {"execution_lock": expected_lock.as_posix()},
    }

    assert controller._authorized_execution_lock(
        authorization,
        output_root=output_root,
    ) == expected_lock.resolve()


@pytest.mark.skipif(controller.fcntl is None, reason="POSIX flock is required")
def test_execution_lock_rejects_a_competing_lock(tmp_path: Path) -> None:
    lock = tmp_path / "candidate.lock"
    with controller._execution_lock(lock):
        with pytest.raises(RuntimeError, match="another bounded continuation"):
            with controller._execution_lock(lock):
                pass


def test_resume_reopens_only_an_incomplete_root_bound_to_the_authorization(
    tmp_path: Path,
) -> None:
    authorization = {"path": "/tmp/auth.json", "bytes": 12, "sha256": "a" * 64}
    root = tmp_path / "candidate"
    status = _controller_status(root, authorization)

    assert controller._validate_resumable_output_root(root, status, authorization)["status"] == "failed"

    tampered = json.loads(status.read_text(encoding="utf-8"))
    tampered["authorization"]["sha256"] = "b" * 64
    status.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization identity"):
        controller._validate_resumable_output_root(root, status, authorization)


def test_resume_rejects_terminal_or_completed_roots(tmp_path: Path) -> None:
    authorization = {"path": "/tmp/auth.json", "bytes": 12, "sha256": "a" * 64}
    root = tmp_path / "candidate"
    status = _controller_status(root, authorization, status="completed")
    with pytest.raises(ValueError, match="non-terminal"):
        controller._validate_resumable_output_root(root, status, authorization)

    status = _controller_status(root, authorization, status="failed")
    (root / "exposure_capacity_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains a result"):
        controller._validate_resumable_output_root(root, status, authorization)


def test_training_resume_action_requires_explicit_resume_for_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.json").write_text(json.dumps({"checkpoint": "checkpoint.pt"}), encoding="utf-8")
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(controller, "resolve_latest_checkpoint", lambda _run: checkpoint)
    monkeypatch.setattr(controller, "verify_training_checkpoint", lambda _checkpoint: {"step": 100_001})
    execution = {"revision": "b" * 40, "branch": "execution"}

    with pytest.raises(RuntimeError, match="with --resume"):
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=tmp_path / "config.json",
            execution_checkout=execution,
            resume_requested=False,
        )
    assert (
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=tmp_path / "config.json",
            execution_checkout=execution,
            resume_requested=True,
        )
        == "auto"
    )


def test_training_resume_action_skips_only_a_verified_completed_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.json").write_text(json.dumps({"checkpoint": "checkpoint.pt"}), encoding="utf-8")
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(controller, "resolve_latest_checkpoint", lambda _run: checkpoint)
    monkeypatch.setattr(controller, "verify_training_checkpoint", lambda _checkpoint: {"step": controller.TARGET_STEP})
    config = tmp_path / "config.json"
    report = {
        "training_complete": True,
        "completed_steps": controller.TARGET_STEP,
        "target_steps": controller.TARGET_STEP,
        "config_path": config.resolve().as_posix(),
        "git": {"revision": "b" * 40, "branch": "execution", "dirty": False},
    }
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")

    assert (
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=config,
            execution_checkout={"revision": "b" * 40, "branch": "execution"},
            resume_requested=True,
        )
        == "skip"
    )

    report["git"]["revision"] = "c" * 40
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="execution identity"):
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=config,
            execution_checkout={"revision": "b" * 40, "branch": "execution"},
            resume_requested=True,
        )


def test_runbook_resume_is_explicitly_opt_in() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_exposure_capacity_continuation_100k_to_110k.sh"
    )
    source = runbook.read_text(encoding="utf-8")
    assert 'RESUME="${EXPOSURE_CONTINUATION_RESUME:-false}"' in source
    assert "RESUME_ARGS+=(--resume)" in source


def _rollout_report_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    checkpoint = tmp_path / "checkpoint_step_00110000.pt"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    sidecar.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "rollout" / "rollout_stability_report.json"
    report_path.parent.mkdir()
    report = {
        "schema_version": 1,
        "status": "completed",
        "git": {"revision": "b" * 40, "branch": "execution", "dirty": False},
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": "a" * 64,
        "checkpoint_integrity_manifest": sidecar.resolve().as_posix(),
        "checkpoint_step": controller.TARGET_STEP,
        "weights": "ema",
        "protocol": {
            "num_images": 64,
            "batch_size": 4,
            "teacher_timesteps": controller.ROLLOUT_TEACHER_TIMESTEPS,
            "sample_steps": controller.SAMPLE_STEPS,
            "guidance_scale": 1.5,
            "teacher_guidance_scale": 1.0,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "clip_x0": True,
            "precision": "bf16",
            "seed": controller.ROLLOUT_SEED,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, checkpoint, report


def test_resume_rollout_reuse_requires_checkpoint_git_and_protocol_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path, checkpoint, report = _rollout_report_fixture(tmp_path)
    monkeypatch.setattr(
        controller,
        "verify_training_checkpoint",
        lambda _checkpoint: {
            "step": controller.TARGET_STEP,
            "checkpoint_sha256": "a" * 64,
        },
    )
    execution = {"revision": "b" * 40, "branch": "execution"}

    assert controller._validate_resumable_rollout_report(
        report_path,
        method="cofitok",
        checkpoint=checkpoint,
        execution_checkout=execution,
    ) == report

    report["protocol"]["guidance_scale"] = 2.0  # type: ignore[index]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol differs at guidance_scale"):
        controller._validate_resumable_rollout_report(
            report_path,
            method="cofitok",
            checkpoint=checkpoint,
            execution_checkout=execution,
        )


def test_result_rollout_report_matches_evaluator_schema_and_top_level_weights(
    tmp_path: Path,
) -> None:
    report_path, _checkpoint, _report = _rollout_report_fixture(tmp_path)

    validated = result_builder._rollout_report(
        report_path,
        method="cofitok",
        expected_checkpoint={"sha256": "a" * 64},
    )

    assert validated["protocol"] == result_builder.EXPECTED_ROLLOUT_PROTOCOL


def test_result_rollout_report_rejects_protocol_drift(
    tmp_path: Path,
) -> None:
    report_path, _checkpoint, report = _rollout_report_fixture(tmp_path)
    report["protocol"]["teacher_timesteps"] = [999, 750]  # type: ignore[index]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol differs at teacher_timesteps"):
        result_builder._rollout_report(
            report_path,
            method="cofitok",
            expected_checkpoint={"sha256": "a" * 64},
        )


def test_result_uses_candidate_gate_runtime_for_training_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control_runtime = "a" * 64
    training_runtime = "b" * 64
    captured_runtimes: list[str] = []
    training_report = {"runtime_environment_sha256": training_runtime}
    validated_authorization = {
        "schema_version": "authorization",
        "decision": "bounded",
        "scope": "exposure",
        "source_checkpoints": {"cofitok": {}, "dense_identity": {}},
        "live_prelaunch": {
            "dataset_identity_sha256": "c" * 64,
            "runtime_environment_sha256": control_runtime,
        },
    }

    monkeypatch.setattr(
        result_builder,
        "validate_authorization_contract",
        lambda *args, **kwargs: validated_authorization,
    )
    monkeypatch.setattr(
        result_builder,
        "read_object",
        lambda *args, **kwargs: training_report,
    )

    def verified_checkpoint(*args, **kwargs):
        captured_runtimes.append(kwargs["expected_runtime_environment_sha256"])
        return {
            "checkpoint": {"sha256": "d" * 64},
            "git": {"revision": "e" * 40, "branch": "execution"},
            "config": {"data": {"dataset": "imagenet_256"}},
        }

    sampling_protocol = {
        "sample_steps": 100,
        "num_samples": 10_000,
        "weights": "ema",
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "seed": 2027,
        "start_index": 0,
        "class_schedule": "balanced_modulo",
    }
    monkeypatch.setattr(
        result_builder,
        "_verified_final_checkpoint",
        verified_checkpoint,
    )
    monkeypatch.setattr(
        result_builder,
        "_sampling_provenance",
        lambda *args, **kwargs: ({}, {"sampling": sampling_protocol}),
    )
    monkeypatch.setattr(
        result_builder,
        "_metrics_report",
        lambda *args, **kwargs: {
            "report": {},
            "metrics": {
                "frechet_inception_distance": 1.0,
                "recall": 0.1,
            },
        },
    )
    monkeypatch.setattr(
        result_builder,
        "_class_report",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        result_builder,
        "_checkpoint_eval",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        result_builder,
        "_rollout_report",
        lambda *args, **kwargs: {
            "report": {},
            "protocol": result_builder.EXPECTED_ROLLOUT_PROTOCOL,
        },
    )

    result_builder.build_result(
        authorization={},
        gate={"live_prelaunch": {"runtime_environment_sha256": training_runtime}},
        preparation={},
        preparation_identity={},
        gate_identity={},
        standing_identity={},
        execution_checkout={},
        config_identities={"cofitok": {}, "dense_identity": {}},
        authorization_identity={},
        candidate_gate_identity={},
        cofitok_run_dir=tmp_path / "cofitok",
        dense_run_dir=tmp_path / "dense",
        cofitok_training_report=tmp_path / "cofitok-training.json",
        dense_training_report=tmp_path / "dense-training.json",
        cofitok_sampling_report=tmp_path / "cofitok-sampling.json",
        dense_sampling_report=tmp_path / "dense-sampling.json",
        cofitok_metrics_report=tmp_path / "cofitok-metrics.json",
        dense_metrics_report=tmp_path / "dense-metrics.json",
        cofitok_class_report=tmp_path / "cofitok-class.json",
        dense_class_report=tmp_path / "dense-class.json",
        cofitok_checkpoint_report=tmp_path / "cofitok-checkpoint.json",
        dense_checkpoint_report=tmp_path / "dense-checkpoint.json",
        cofitok_rollout_report=tmp_path / "cofitok-rollout.json",
        dense_rollout_report=tmp_path / "dense-rollout.json",
    )

    assert captured_runtimes == [training_runtime, training_runtime]
    assert training_runtime != control_runtime


def test_result_uses_authorized_source_path_for_horizon_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step_00110000.pt"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    sidecar.write_text("{}", encoding="utf-8")
    metrics = {
        "step": result_builder.TARGET_STEP,
        "samples_seen": result_builder.EXPECTED_IMAGES_SEEN,
        "loss": 0.1,
    }
    (run_dir / "train_metrics.jsonl").write_text(
        json.dumps(metrics) + "\n",
        encoding="utf-8",
    )
    checkpoint_sha = "d" * 64
    checkpoint_bytes = checkpoint.stat().st_size
    (run_dir / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "step": result_builder.TARGET_STEP,
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_sha256": checkpoint_sha,
                "integrity_manifest": sidecar.name,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        result_builder,
        "verify_training_checkpoint",
        lambda _checkpoint: {
            "step": result_builder.TARGET_STEP,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_sha,
        },
    )
    monkeypatch.setattr(
        result_builder,
        "resolve_latest_checkpoint",
        lambda _run_dir: checkpoint,
    )

    source_checkpoint_path = tmp_path / "source" / "checkpoint_step_00100000.pt"
    source_sidecar_path = source_checkpoint_path.with_name(
        f"{source_checkpoint_path.name}.integrity.json"
    )
    source_checkpoint = {
        "path": source_checkpoint_path.as_posix(),
        "bytes": 123,
        "sha256": "a" * 64,
    }
    source_sidecar = {
        "path": source_sidecar_path.as_posix(),
        "bytes": 456,
        "sha256": "b" * 64,
    }
    execution = {"revision": "e" * 40, "branch": "execution"}
    runtime_sha = "f" * 64
    dataset_sha = "c" * 64
    config_path = tmp_path / "continuation.json"
    config = {
        "name": "continuation",
        "runtime": {"steps": result_builder.TARGET_STEP},
        "data": {"dataset": "imagenet_256", "batch_size": 64},
    }
    report = {
        "training_complete": True,
        "completed_steps": result_builder.TARGET_STEP,
        "target_steps": result_builder.TARGET_STEP,
        "final_metrics": metrics,
        "config": config,
        "config_path": config_path.as_posix(),
        "git": {**execution, "dirty": False},
        "dataset_provenance": {"identity_sha256": dataset_sha, "status": "pass"},
        "runtime_environment_sha256": runtime_sha,
        "horizon_extension": {
            "schema_version": result_builder.HORIZON_EXTENSION_SCHEMA_VERSION,
            "kind": "bounded_training_horizon_extension",
            "source_horizon_steps": result_builder.SOURCE_STEP,
            "target_horizon_steps": result_builder.TARGET_STEP,
            "additional_horizon_steps": result_builder.TARGET_STEP
            - result_builder.SOURCE_STEP,
            "source_checkpoint_step": result_builder.SOURCE_STEP,
            "allowed_config_mismatch_paths": ["config.name", "config.runtime.steps"],
            "target_config_sha256": result_builder._canonical_object_sha256(config),
            "source_checkpoint": {
                **source_checkpoint,
                "filename": source_checkpoint_path.name,
                "step": result_builder.SOURCE_STEP,
                "integrity_manifest": source_sidecar["path"],
                "integrity_manifest_bytes": source_sidecar["bytes"],
                "integrity_manifest_sha256": source_sidecar["sha256"],
            },
            "scheduler": {
                "policy": result_builder.HORIZON_EXTENSION_SCHEDULER_POLICY,
                "source_horizon_steps": result_builder.SOURCE_STEP,
                "effective_horizon_steps": result_builder.SOURCE_STEP,
                "target_horizon_steps": result_builder.TARGET_STEP,
                "explicit_resume_target_steps_required": True,
                "restored_last_epoch": result_builder.SOURCE_STEP,
            },
        },
        "resume": source_checkpoint["path"],
        "resume_revision_transition": {
            "schema_version": 1,
            "reason": "sampler_rng_state_device_compatibility",
            "source_revision": result_builder.SOURCE_CHECKOUT["revision"],
            "target_revision": execution["revision"],
            "branch": execution["branch"],
            "source_checkpoint": {
                **source_checkpoint,
                "filename": source_checkpoint_path.name,
                "step": result_builder.SOURCE_STEP,
                "integrity_manifest": source_sidecar["path"],
                "git_branch": result_builder.SOURCE_CHECKOUT["branch"],
            },
        },
        "metrics_resume_reconciliation": {},
        "output_dir": run_dir.as_posix(),
        "latest_checkpoint": {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_bytes,
            "step": result_builder.TARGET_STEP,
        },
        "parameter_count": 1,
    }
    report_path = run_dir / "training_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    validated = result_builder._verified_final_checkpoint(
        report,
        method="cofitok",
        expected_run_dir=run_dir,
        report_path=report_path,
        expected_source={
            "step": result_builder.SOURCE_STEP,
            "checkpoint": source_checkpoint,
            "integrity_manifest": source_sidecar,
        },
        expected_execution_checkout=execution,
        expected_config_identity={"path": config_path.as_posix()},
        expected_dataset_identity_sha256=dataset_sha,
        expected_runtime_environment_sha256=runtime_sha,
    )

    assert validated["checkpoint"]["path"] == checkpoint.resolve().as_posix()


def test_result_normalizes_top_level_sampling_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    report_path = tmp_path / "sampling_report.json"
    sampling = {
        "sample_steps": 100,
        "num_samples": 10_000,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "seed": 2027,
        "start_index": 0,
        "class_schedule": "balanced_modulo",
    }
    report = {
        "status": "completed",
        "sampling": sampling,
        "checkpoint_step": result_builder.TARGET_STEP,
        "checkpoint_sha256": "a" * 64,
        "weights": "ema",
        "output_dirs": {"8": generated.as_posix()},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    images = [generated / f"{index:05d}.png" for index in range(10_000)]
    provenance = {
        "sampling": sampling,
        "checkpoint_step": result_builder.TARGET_STEP,
        "weights": "ema",
        "sample_set_sha256": "b" * 64,
    }
    monkeypatch.setattr(metrics_evaluator, "find_images", lambda _path: images)
    monkeypatch.setattr(
        metrics_evaluator,
        "validate_sampling_provenance",
        lambda *_args, **_kwargs: provenance,
    )
    monkeypatch.setattr(
        result_builder,
        "sample_set_sha256",
        lambda _images: provenance["sample_set_sha256"],
    )

    _, validated = result_builder._sampling_provenance(
        report_path,
        expected_checkpoint={"sha256": "a" * 64},
        method="cofitok",
    )

    assert "weights" not in sampling
    assert validated["sampling"] == {**sampling, "weights": "ema"}
    assert validated["provenance"]["sampling"] == sampling

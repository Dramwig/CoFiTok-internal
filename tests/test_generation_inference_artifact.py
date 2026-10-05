from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cofitok.configs import (
    DataConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    config_to_dict,
)
from cofitok.generation import (
    GenerationRequest,
    GenerationSession,
    export_ema_inference_artifact,
    inference_export_manifest_path,
    verify_inference_artifact,
    verify_inference_export_manifest,
    verify_generation_release_receipt,
    write_generation_release_receipt,
)
from cofitok.generation_gate import (
    GENERATION_GATE_SCHEMA_VERSION,
    REQUIRED_GENERATION_GATES,
)
from cofitok.generation_gate_sources import (
    GATE_SOURCE_SUFFIXES,
    build_generation_gate_source_reports,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.models import CoFiTokTiny
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path


ROOT = Path(__file__).resolve().parents[1]
from scripts.preflight_generation_sampling import run_sampling_preflight


SOURCE_ENVIRONMENT_SHA = "e" * 64
SOURCE_GIT = {
    "revision": "a" * 40,
    "branch": "scale/generative-system",
    "dirty": False,
}


def _training_authorization(tmp_path) -> dict:
    return {
        "schema_version": 1,
        "status": "pass",
        "stage": "scaling",
        "decision": "promote_to_full_imagenet256",
        "gate_path": (tmp_path / "promotion_gate.json").resolve().as_posix(),
        "gate_bytes": 12_345,
        "gate_sha256": "b" * 64,
        "gate_identity_sha256": "c" * 64,
        "validated_thresholds": {
            "min_samples": 10_000.0,
            "max_fid_regression": 0.05,
            "max_absolute_fid": 100.0,
            "max_endpoint_regression": 0.05,
        },
    }


def _full_gate() -> dict:
    summary = {
        "cofitok_fid": 19.5,
        "dense_fid": 19.0,
        "cofitok_endpoint_mse": 0.104,
        "dense_endpoint_mse": 0.1,
        "cofitok_precision": 0.35,
        "dense_precision": 0.36,
        "cofitok_recall": 0.34,
        "dense_recall": 0.35,
        "ordered_rank": 1,
        "order_count": 24,
        "coarse_token_energy_ratio": 0.06,
    }
    thresholds = {
        "min_samples": 50_000,
        "max_fid_regression": 0.05,
        "max_absolute_fid": 20.0,
        "max_endpoint_regression": 0.05,
        "min_coarse_token_energy_ratio": 0.05,
        "min_precision": 0.30,
        "min_recall": 0.30,
        "max_precision_regression": 0.05,
        "max_recall_regression": 0.05,
    }

    def evidence(name: str) -> dict:
        if name == "fid_within_tolerance":
            return {
                "cofitok_fid": summary["cofitok_fid"],
                "dense_fid": summary["dense_fid"],
                "max_regression": thresholds["max_fid_regression"],
            }
        if name == "absolute_fid_quality":
            return {
                "cofitok_fid": summary["cofitok_fid"],
                "max_absolute_fid": thresholds["max_absolute_fid"],
            }
        if name == "endpoint_within_tolerance":
            return {
                "cofitok_endpoint_mse": summary["cofitok_endpoint_mse"],
                "dense_endpoint_mse": summary["dense_endpoint_mse"],
                "max_regression": thresholds["max_endpoint_regression"],
            }
        if name == "ordered_prefix_path":
            return {
                "rank": summary["ordered_rank"],
                "order_count": summary["order_count"],
            }
        if name == "coarse_token_utilization":
            return {
                "valid": True,
                "source_metric": "component_energy_ratio_per_sample_mean",
                "token_count": 8,
                "coarse_token_count": 6,
                "component_energy_ratios": [0.01] * 6 + [0.30, 0.64],
                "coarse_token_energy_ratio": summary["coarse_token_energy_ratio"],
                "min_coarse_token_energy_ratio": thresholds[
                    "min_coarse_token_energy_ratio"
                ],
            }
        if name == "restricted_synthesis_contract":
            return {"zero_token_max_abs": 0.0}
        if name == "shuffle_mismatch":
            return {"shuffled_to_ordered_endpoint_ratio": 1.2}
        if name == "full_precision_recall_quality":
            return {
                "enforced": True,
                **{
                    key: thresholds[key]
                    for key in (
                        "min_precision",
                        "min_recall",
                        "max_precision_regression",
                        "max_recall_regression",
                    )
                },
                **{
                    key: summary[key]
                    for key in (
                        "cofitok_precision",
                        "dense_precision",
                        "cofitok_recall",
                        "dense_recall",
                    )
                },
            }
        return {}

    return {
        "schema_version": GENERATION_GATE_SCHEMA_VERSION,
        "stage": "full",
        "status": "pass",
        "decision": "large_scale_generation_ready",
        "thresholds": thresholds,
        "gates": [
            {"name": name, "passed": True, "evidence": evidence(name)}
            for name in sorted(REQUIRED_GENERATION_GATES["full"])
        ],
        "summary": summary,
    }


def _release_gate_path(tmp_path):
    gate = _full_gate()
    paths = {}
    for index, (name, suffix) in enumerate(GATE_SOURCE_SUFFIXES["full"].items()):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"name": name, "index": index}),
            encoding="utf-8",
        )
        paths[name] = path
    gate["source_reports"] = build_generation_gate_source_reports(
        stage="full",
        paths=paths,
    )
    path = tmp_path / "final_generation_gate.json"
    path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    return path


def _training_checkpoint(
    tmp_path,
    *,
    include_provenance: bool = True,
    include_authorization: bool = False,
    step: int = 31,
):
    config = ExperimentConfig(
        name="inference_export_cpu",
        data=DataConfig(image_size=8, channels=3),
        diffusion=DiffusionConfig(num_train_timesteps=4, schedule_type="cosine"),
        model=ModelConfig(
            image_channels=3,
            image_size=8,
            token_count=2,
            token_channels=4,
            base_channels=8,
            predictor_type="scalable_unet",
            predictor_channel_multipliers=[1],
            predictor_num_res_blocks=1,
            predictor_attention_resolutions=[],
            predictor_num_heads=1,
            num_classes=5,
            synthesis_active_token_channels=[2, 4],
        ),
        runtime=RuntimeConfig(device="cpu", precision="fp32"),
        optimization=OptimizationConfig(ema_warmup_steps=0),
    )
    model = CoFiTokTiny(config.model)
    ema = ExponentialMovingAverage(model, warmup_steps=0)
    with torch.no_grad():
        for value in ema.shadow.values():
            if torch.is_floating_point(value):
                value.add_(0.01)
    path = tmp_path / "training.pt"
    payload = {
        "format_version": 1,
        "config": config_to_dict(config),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "step": step,
    }
    if include_provenance:
        payload["extra_state"] = {
            "runtime_environment_sha256": SOURCE_ENVIRONMENT_SHA,
            "git": SOURCE_GIT,
        }
        if include_authorization:
            payload["extra_state"]["training_authorization"] = (
                _training_authorization(tmp_path)
            )
    torch.save(payload, path)
    integrity = {
        "schema_version": 1,
        "checkpoint": path.name,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_format_version": 1,
        "step": step,
    }
    if include_provenance:
        integrity.update(
            runtime_environment_sha256=SOURCE_ENVIRONMENT_SHA,
            git_revision=SOURCE_GIT["revision"],
            git_branch=SOURCE_GIT["branch"],
            git_dirty=SOURCE_GIT["dirty"],
        )
        if include_authorization:
            authorization = _training_authorization(tmp_path)
            integrity.update(
                authorization_stage=authorization["stage"],
                authorization_decision=authorization["decision"],
                authorization_gate_bytes=authorization["gate_bytes"],
                authorization_gate_sha256=authorization["gate_sha256"],
                authorization_gate_identity_sha256=authorization[
                    "gate_identity_sha256"
                ],
            )
    write_json_report(
        checkpoint_integrity_path(path),
        integrity,
    )
    return path


def _completion_audit_path(
    tmp_path: Path,
    *,
    cofitok_export: dict,
    dense_export: dict,
) -> Path:
    def evidence(report: dict, *, smoke_count: int) -> dict:
        return {
            "artifact_path": report["artifact"],
            "artifact_sha256": report["artifact_sha256"],
            "artifact_bytes": report["artifact_bytes"],
            "export_manifest": report["export_manifest"],
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            "source_checkpoint_bytes": report["source_checkpoint_bytes"],
            "source_runtime_environment_sha256": report[
                "source_runtime_environment_sha256"
            ],
            "source_git": report["source_git"],
            "execution_git": None,
            "export_runtime_environment_sha256": None,
            "execution_runtime_environment_sha256": None,
            "training_authorization": report["source_training_authorization"],
            "release_authorization": report["release_authorization"],
            "smoke_output_count": smoke_count,
            "smoke_output_sha256": ["9" * 64] * smoke_count,
        }

    audit = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "expected_revisions": {
            "deployment_source": "1" * 40,
            "ten_percent_training": "2" * 40,
            "full_training": "3" * 40,
        },
        "checks": [
            {
                "name": "deployable_ema_inference_artifacts",
                "status": "pass",
                "evidence": {
                    "cofitok": evidence(cofitok_export, smoke_count=4),
                    "dense_identity": evidence(dense_export, smoke_count=2),
                },
            }
        ],
        "failed_checks": [],
        "missing_checks": [],
        "warnings": [],
    }
    path = tmp_path / "completion_audit.json"
    write_json_report(path, audit)
    return path


def _release_receipt_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = _training_checkpoint(
        tmp_path,
        include_authorization=True,
        step=300_000,
    )
    release_gate = _release_gate_path(tmp_path)
    cofitok_artifact = tmp_path / "cofitok_ema_inference.pt"
    dense_artifact = tmp_path / "dense_ema_inference.pt"
    cofitok_export = export_ema_inference_artifact(
        source,
        cofitok_artifact,
        release_gate=release_gate,
    )
    dense_export = export_ema_inference_artifact(
        source,
        dense_artifact,
        release_gate=release_gate,
    )
    audit = _completion_audit_path(
        tmp_path,
        cofitok_export=cofitok_export,
        dense_export=dense_export,
    )
    receipt = tmp_path / "release_receipt.json"
    write_generation_release_receipt(audit, receipt)
    return cofitok_artifact, audit, receipt


def test_ema_export_is_smaller_verified_and_sample_equivalent(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    release_gate = _release_gate_path(tmp_path)

    report = export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )
    reused = export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )

    assert report["status"] == "completed"
    assert report["schema_version"] == 4
    assert report["weights"] == "ema_export"
    assert report["verified"] is True
    assert report["artifact_bytes"] < report["source_checkpoint_bytes"]
    assert reused["reused"] is True
    verified = verify_inference_artifact(artifact)
    assert verified["schema_version"] == 4
    assert verified["artifact_format_version"] == 4
    assert verified["artifact_sha256"] == report["artifact_sha256"]
    assert report["source_runtime_environment_sha256"] == SOURCE_ENVIRONMENT_SHA
    assert report["source_git"] == SOURCE_GIT
    assert report["source_training_authorization"] == _training_authorization(
        tmp_path
    )
    assert verified["source_training_authorization"] == _training_authorization(
        tmp_path
    )
    assert report["release_authorization"] == verified["release_authorization"]
    assert report["release_authorization"]["stage"] == "full"
    assert report["release_authorization"]["decision"] == (
        "large_scale_generation_ready"
    )
    manifest_path = inference_export_manifest_path(artifact)
    manifest = verify_inference_export_manifest(
        manifest_path,
        expected_artifact=artifact,
    )
    assert report["export_manifest"] == {
        "path": manifest_path.resolve().as_posix(),
        "bytes": manifest_path.stat().st_size,
        "sha256": file_sha256(manifest_path),
    }
    assert manifest["source"]["sha256"] == report["source_checkpoint_sha256"]
    assert manifest["target"]["artifact"] == artifact.resolve().as_posix()
    assert manifest["release_authorization"] == report["release_authorization"]

    request = GenerationRequest(
        seeds=(9,),
        class_labels=(2,),
        sample_steps=1,
        guidance_scale=1.0,
        precision="fp32",
    )
    source_result = GenerationSession.from_checkpoint(source, weights="ema").generate(
        request
    )
    export_session = GenerationSession.from_checkpoint(
        artifact,
        weights="ema",
        require_release_authorization=True,
    )
    export_result = export_session.generate(request)
    torch.testing.assert_close(
        source_result.images,
        export_result.images,
        rtol=0.0,
        atol=0.0,
    )
    assert export_result.metadata["weights"] == "ema_export"
    assert export_result.metadata["source_checkpoint_sha256"] == report[
        "source_checkpoint_sha256"
    ]
    assert (
        export_result.metadata["source_runtime_environment_sha256"]
        == SOURCE_ENVIRONMENT_SHA
    )
    assert export_result.metadata["source_git"] == SOURCE_GIT
    assert export_result.metadata["training_authorization"] == (
        _training_authorization(tmp_path)
    )
    assert export_result.metadata["release_authorization"] == report[
        "release_authorization"
    ]
    assert export_result.metadata["release_authorization_required"] is True
    preflight = run_sampling_preflight(
        artifact,
        batch_size=1,
        prefix_budget=2,
        guidance_scale=1.0,
        precision="fp32",
        require_release_authorization=True,
    )
    assert preflight["status"] == "passed"
    assert preflight["release_authorization_required"] is True
    assert preflight["training_authorization"] == _training_authorization(
        tmp_path
    )
    assert preflight["release_authorization"] == report[
        "release_authorization"
    ]

    cli_report_path = tmp_path / "cli_export_report.json"
    cli_artifact = tmp_path / "cli_cofitok_ema_inference.pt"
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(ROOT / "src"), inherited_pythonpath) if path
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_generation_inference_artifact.py"),
            "--checkpoint",
            str(source),
            "--output",
            str(cli_artifact),
            "--release-gate",
            str(release_gate),
            "--resume",
            "--report",
            str(cli_report_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cli_report = json.loads(cli_report_path.read_text(encoding="utf-8"))
    execution = cli_report["execution"]
    assert len(execution["git"]["revision"]) == 40
    assert execution["runtime_environment"]["device"]["type"] == "cpu"
    assert execution["runtime_environment_sha256"] == (
        runtime_environment_sha256(execution["runtime_environment"])
    )
    cli_manifest = verify_inference_export_manifest(
        inference_export_manifest_path(cli_artifact),
        expected_artifact=cli_artifact,
    )
    assert cli_manifest["execution"] == execution
    assert cli_report["resume_requested"] is True
    assert cli_report["partial_outputs_recovered"] is False


def test_ema_export_rejects_source_without_deployment_provenance(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_provenance=False)

    with pytest.raises(ValueError, match="runtime environment provenance"):
        export_ema_inference_artifact(
            source,
            tmp_path / "unprovenanced_inference.pt",
        )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows symlink creation requires elevated privileges",
)
def test_inference_artifact_rejects_symlinked_artifact_and_integrity_manifest(
    tmp_path,
) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)

    artifact_alias = tmp_path / "artifact_alias.pt"
    artifact_alias.symlink_to(artifact)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        verify_inference_artifact(artifact_alias)

    integrity_path = checkpoint_integrity_path(artifact)
    integrity_target = tmp_path / "artifact_integrity_target.json"
    integrity_target.write_bytes(integrity_path.read_bytes())
    integrity_path.unlink()
    integrity_path.symlink_to(integrity_target)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        verify_inference_artifact(artifact)


def test_completed_export_replay_preserves_all_bound_bytes(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    first = export_ema_inference_artifact(source, artifact)
    paths = (
        artifact,
        checkpoint_integrity_path(artifact),
        inference_export_manifest_path(artifact),
    )
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size, file_sha256(path))
        for path in paths
    }

    replay = export_ema_inference_artifact(source, artifact, resume=True)

    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size, file_sha256(path))
        for path in paths
    }
    assert replay["reused"] is True
    assert replay["resume_requested"] is True
    assert replay["partial_outputs_recovered"] is False
    assert replay["artifact_sha256"] == first["artifact_sha256"]
    assert after == before


def test_partial_export_requires_manifest_bound_explicit_resume(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    checkpoint_integrity_path(artifact).unlink()
    stale = artifact.parent / f"{artifact.name}.tmp-stale"
    stale.write_bytes(b"partial temporary")

    with pytest.raises(ValueError, match="requires explicit resume"):
        export_ema_inference_artifact(source, artifact)
    assert artifact.is_file()
    assert stale.is_file()

    recovered = export_ema_inference_artifact(source, artifact, resume=True)

    assert recovered["reused"] is False
    assert recovered["resume_requested"] is True
    assert recovered["partial_outputs_recovered"] is True
    assert not stale.exists()
    verify_inference_artifact(artifact)
    verify_inference_export_manifest(
        inference_export_manifest_path(artifact),
        expected_artifact=artifact,
    )


def test_partial_export_without_manifest_is_never_deleted(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    checkpoint_integrity_path(artifact).unlink()
    inference_export_manifest_path(artifact).unlink()
    artifact_sha = file_sha256(artifact)

    with pytest.raises(ValueError, match="lacks a matching export manifest"):
        export_ema_inference_artifact(source, artifact, resume=True)

    assert artifact.is_file()
    assert file_sha256(artifact) == artifact_sha
    assert not checkpoint_integrity_path(artifact).exists()
    assert not inference_export_manifest_path(artifact).exists()


def test_partial_export_rejects_manifest_drift_before_deserialization(
    tmp_path,
    monkeypatch,
) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    checkpoint_integrity_path(artifact).unlink()
    manifest_path = inference_export_manifest_path(artifact)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution"] = {"unbound": "environment"}
    write_json_report(manifest_path, manifest)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before manifest rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="manifest request differs"):
        export_ema_inference_artifact(source, artifact, resume=True)
    assert artifact.is_file()


def test_complete_but_tampered_export_is_not_repaired_by_resume(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    tampered_sha = file_sha256(artifact)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        export_ema_inference_artifact(source, artifact, resume=True)

    assert file_sha256(artifact) == tampered_sha
    assert checkpoint_integrity_path(artifact).is_file()
    assert inference_export_manifest_path(artifact).is_file()


def test_concurrent_export_fails_closed_before_creating_outputs(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    report = tmp_path / "export_report.json"
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(ROOT / "src"), inherited_pythonpath) if path
    )

    with exclusive_output_lock(artifact, role="test_export_owner"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_generation_inference_artifact.py"),
                "--checkpoint",
                str(source),
                "--output",
                str(artifact),
                "--report",
                str(report),
                "--resume",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert "already locked" in result.stderr
    assert not artifact.exists()
    assert not checkpoint_integrity_path(artifact).exists()
    assert not inference_export_manifest_path(artifact).exists()
    assert not report.exists()


def test_inference_artifact_rejects_model_weights_and_tampering(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)

    with pytest.raises(ValueError, match="only exported EMA"):
        GenerationSession.from_checkpoint(artifact, weights="model")

    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_production_mode_rejects_unreleased_inference_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("inference artifact was deserialized before policy rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="release-authorized inference artifact"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            require_release_authorization=True,
        )


def test_inference_artifact_rejects_source_sidecar_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["source_git_revision"] = "b" * 40
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="source Git provenance mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_inference_artifact_rejects_training_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=_release_gate_path(tmp_path),
    )
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["source_training_authorization"][
        "gate_identity_sha256"
    ] = "d" * 64
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="training authorization mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_export_reuse_rejects_source_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    release_gate = _release_gate_path(tmp_path)
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )
    source_integrity_path = checkpoint_integrity_path(source)
    source_integrity = json.loads(
        source_integrity_path.read_text(encoding="utf-8")
    )
    source_integrity["authorization_gate_identity_sha256"] = "d" * 64
    write_json_report(source_integrity_path, source_integrity)

    with pytest.raises(ValueError, match="authorization differs"):
        export_ema_inference_artifact(
            source,
            artifact,
            release_gate=release_gate,
        )


def test_formal_inference_export_requires_release_gate(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)

    with pytest.raises(ValueError, match="requires a full release gate"):
        export_ema_inference_artifact(
            source,
            tmp_path / "unauthorized_inference.pt",
        )


def test_completion_receipt_authorizes_consumer_load_and_preflight(tmp_path) -> None:
    artifact, audit, receipt = _release_receipt_fixture(tmp_path)

    authorization = verify_generation_release_receipt(receipt, artifact)
    session = GenerationSession.from_checkpoint(
        artifact,
        weights="ema",
        completion_receipt=receipt,
        require_completion_authorization=True,
    )
    result = session.generate(
        GenerationRequest(
            seeds=(7,),
            class_labels=(2,),
            sample_steps=1,
            guidance_scale=1.0,
            precision="fp32",
        )
    )
    preflight = run_sampling_preflight(
        artifact,
        batch_size=1,
        prefix_budget=2,
        guidance_scale=1.0,
        precision="fp32",
        completion_receipt=receipt,
        require_completion_authorization=True,
    )

    assert authorization["method"] == "cofitok"
    assert authorization["completion_audit"]["path"] == audit.resolve().as_posix()
    assert result.metadata["completion_authorization"] == authorization
    assert result.metadata["completion_authorization_required"] is True
    assert result.metadata["release_authorization_required"] is True
    assert preflight["status"] == "passed"
    assert preflight["completion_authorization"] == authorization
    assert preflight["completion_authorization_required"] is True


def test_release_receipt_requires_bound_export_manifest_at_consumption(tmp_path) -> None:
    artifact, _, receipt = _release_receipt_fixture(tmp_path)
    manifest = inference_export_manifest_path(artifact)
    manifest.unlink()

    with pytest.raises(FileNotFoundError, match="release source is missing"):
        verify_generation_release_receipt(receipt, artifact)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix symlink creation requires elevated privileges on Windows",
)
def test_release_receipt_rejects_symlinked_receipt(tmp_path) -> None:
    artifact, _, receipt = _release_receipt_fixture(tmp_path)
    alias = tmp_path / "release_receipt_alias.json"
    alias.symlink_to(receipt)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        verify_generation_release_receipt(alias, artifact)


def test_release_artifact_remains_portable_after_source_checkpoint_archival(
    tmp_path,
) -> None:
    artifact, _, receipt = _release_receipt_fixture(tmp_path)
    manifest = json.loads(
        inference_export_manifest_path(artifact).read_text(encoding="utf-8")
    )
    source = Path(manifest["source"]["path"])
    checkpoint_integrity_path(source).unlink()
    source.unlink()

    authorization = verify_generation_release_receipt(receipt, artifact)

    assert authorization["method"] == "cofitok"


def test_stability_completion_profile_publishes_release_receipt(tmp_path) -> None:
    artifact, audit, receipt = _release_receipt_fixture(tmp_path)
    receipt.unlink()
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["profile"] = "stability_generation_system_v1"
    payload["status"] = "pass"
    payload["expectations"] = {
        "full_training_revision": "4" * 40,
        "full_evaluation_revision": "5" * 40,
        "export_revision": "5" * 40,
    }
    payload.pop("expected_revisions")
    payload["checks"][0]["name"] = "stability_release_authorized_inference"
    write_json_report(audit, payload)

    written = write_generation_release_receipt(audit, receipt)
    authorization = verify_generation_release_receipt(receipt, artifact)

    assert written["completion_profile"] == "stability_generation_system_v1"
    assert written["completion_expectations"] == payload["expectations"]
    assert authorization["completion_profile"] == (
        "stability_generation_system_v1"
    )


def test_completion_authorization_requires_receipt_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    artifact, _, _ = _release_receipt_fixture(tmp_path)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("artifact was deserialized before receipt policy rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="requires a release receipt"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            require_completion_authorization=True,
        )


def test_completion_receipt_rejects_changed_terminal_audit_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    artifact, audit, receipt = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["warnings"] = ["post-release mutation"]
    write_json_report(audit, changed)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("artifact was deserialized before audit drift rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="changed after release"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            completion_receipt=receipt,
            require_completion_authorization=True,
        )


def test_release_receipt_rejects_incomplete_completion_audit(tmp_path) -> None:
    artifact, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed.update(status="in_progress", complete=False)
    changed["missing_checks"] = ["formal_50k_generation"]
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="did not pass"):
        write_generation_release_receipt(
            audit,
            tmp_path / "invalid_release_receipt.json",
        )

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="not authorized"):
        verify_generation_release_receipt(
            _release_receipt_fixture(other)[2],
            artifact,
        )


def test_formal_export_rejects_release_source_drift_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    release_gate = _release_gate_path(tmp_path)
    gate = json.loads(release_gate.read_text(encoding="utf-8"))
    source_report = Path(gate["source_reports"]["dense_generation"]["path"])
    source_report.write_text("changed\n", encoding="ascii")

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before gate-source validation")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="source report changed after binding"):
        export_ema_inference_artifact(
            source,
            tmp_path / "source_drift_inference.pt",
            release_gate=release_gate,
        )


def test_inference_artifact_rejects_release_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=_release_gate_path(tmp_path),
    )
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["release_authorization"]["gate_identity_sha256"] = "d" * 64
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="release authorization mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_export_reuse_rejects_changed_release_gate(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    release_gate = _release_gate_path(tmp_path)
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )
    changed_gate = json.loads(release_gate.read_text(encoding="utf-8"))
    changed_gate["gates"].append(
        {"name": "additional_release_audit", "passed": True, "evidence": {}}
    )
    release_gate.write_text(
        json.dumps(changed_gate, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="release authorization differs"):
        export_ema_inference_artifact(
            source,
            artifact,
            release_gate=release_gate,
        )


def test_formal_export_runbook_binds_full_release_gate() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_export_inference_artifacts.sh"
    ).read_text(encoding="utf-8")

    assert runbook.count('--release-gate "$FINAL_GATE"') == 2
    assert runbook.count("--resume") == 2
    assert runbook.count(".export_manifest.json") == 2
    assert runbook.count('test -f "$COFITOK_EXPORT_MANIFEST"') == 1
    assert runbook.count('test -f "$DENSE_EXPORT_MANIFEST"') == 1

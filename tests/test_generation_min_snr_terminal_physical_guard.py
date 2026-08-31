from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cofitok.environment import runtime_environment_sha256
from cofitok.generation import (
    INFERENCE_API,
    SAMPLING_MANIFEST_SCHEMA_VERSION,
    SAMPLING_PROTOCOL_SCHEMA,
    SAMPLING_REPORT_SCHEMA_VERSION,
)
from cofitok.generation import min_snr_pilot
from cofitok.generation import min_snr_terminal_guard as guard
from cofitok.image_integrity import sample_set_sha256
from cofitok.reporting import file_sha256
from cofitok.diffusion import select_sampling_timesteps
from scripts.evaluate_generation_metrics import find_images, validate_sampling_provenance


REVISION = "a" * 40
TREE = "b" * 40
GUARD_REVISION = "c" * 40
GUARD_TREE = "d" * 40
GUARD_BRANCH = "analysis/generation-min-snr-terminal-physical-replay-v1-20260825"


def _identity(name: str) -> dict[str, Any]:
    digit = hex((sum(name.encode("utf-8")) % 15) + 1)[2:]
    return {"path": f"/source/{name}", "bytes": 1, "sha256": digit * 64}


def _guard_git() -> dict[str, Any]:
    return {
        "revision": GUARD_REVISION,
        "tree": GUARD_TREE,
        "branch": GUARD_BRANCH,
        "tracked_dirty": False,
    }


def _source_code_binding() -> dict[str, Any]:
    files = {name: _identity(name) for name in guard.SOURCE_CODE_PATHS}
    return {
        "training_source": {
            "revision": REVISION,
            "tree": TREE,
            "branch": min_snr_pilot.PILOT_BRANCH,
            "tracked_dirty": False,
        },
        "guard_source": _guard_git(),
        "training_source_is_ancestor": True,
        "training_replay_blobs": {
            name: "e" * 40 for name in guard.TRAINING_REPLAY_CODE_PATHS
        },
        "files": files,
        "aggregate_sha256": guard._canonical_digest(files),
    }


def _fake_physical_evidence() -> dict[str, Any]:
    return {
        "selection_status": "no_shared_min_snr_candidate_at_50k",
        "result_logical_replay_exact": True,
        "all_checkpoint_payloads_physically_hashed": True,
        "all_sample_sets_physically_hashed": True,
        "all_classifier_weights_physically_hashed": True,
        "all_source_reports_physically_hashed": True,
        "all_training_control_checkpoints_physically_bound": True,
        "all_checkpoint_payloads_rehashed_after_replay": True,
        "all_terminal_sources_rehashed_after_replay": True,
    }


def _fake_source_replay(binding: Any) -> dict[str, Any]:
    return {
        "files": copy.deepcopy(dict(binding["files"])),
        "aggregate_sha256": binding["aggregate_sha256"],
        "physical_identities_verified": True,
    }


def test_terminal_guard_is_permanently_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fake_physical_evidence()
    monkeypatch.setattr(guard, "physical_replay", lambda **kwargs: copy.deepcopy(evidence))
    monkeypatch.setattr(guard, "_verify_source_code_files", _fake_source_replay)
    report = guard.build_terminal_guard(
        output_root=tmp_path,
        result_path=tmp_path / "result.json",
        expected_result_sha256="f" * 64,
        expected_revision=REVISION,
        expected_tree=TREE,
        expected_branch=min_snr_pilot.PILOT_BRANCH,
        guard_git=_guard_git(),
        source_code_binding=_source_code_binding(),
    )

    assert guard.validate_terminal_guard(report) == report
    assert report["terminal_status"] == "hold"
    assert report["generation_advantage_proven"] is False
    assert not any(report["authorization_boundary"].values())
    assert report["execution_policy"]["persistent_process_allowed"] is False
    assert report["execution_policy"]["waiter_deployment_allowed"] is False

    tampered = copy.deepcopy(report)
    tampered["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        guard.validate_terminal_guard(tampered)


def test_terminal_guard_rejects_drift_between_physical_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def replay(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        del kwargs
        calls += 1
        value = _fake_physical_evidence()
        value["pass"] = calls
        return value

    monkeypatch.setattr(guard, "physical_replay", replay)
    monkeypatch.setattr(guard, "_verify_source_code_files", _fake_source_replay)
    with pytest.raises(ValueError, match="changed during double replay"):
        guard.build_terminal_guard(
            output_root=tmp_path,
            result_path=tmp_path / "result.json",
            expected_result_sha256="f" * 64,
            expected_revision=REVISION,
            expected_tree=TREE,
            expected_branch=min_snr_pilot.PILOT_BRANCH,
            guard_git=_guard_git(),
            source_code_binding=_source_code_binding(),
        )
    assert calls == 2


def test_terminal_guard_rejects_source_drift_between_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def source_replay(binding: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        replay = _fake_source_replay(binding)
        if calls == 2:
            replay["files"] = copy.deepcopy(replay["files"])
            first_name = next(iter(replay["files"]))
            replay["files"][first_name]["sha256"] = "0" * 64
        return replay

    monkeypatch.setattr(
        guard,
        "physical_replay",
        lambda **kwargs: copy.deepcopy(_fake_physical_evidence()),
    )
    monkeypatch.setattr(guard, "_verify_source_code_files", source_replay)
    with pytest.raises(ValueError, match="changed during double replay"):
        guard.build_terminal_guard(
            output_root=tmp_path,
            result_path=tmp_path / "result.json",
            expected_result_sha256="f" * 64,
            expected_revision=REVISION,
            expected_tree=TREE,
            expected_branch=min_snr_pilot.PILOT_BRANCH,
            guard_git=_guard_git(),
            source_code_binding=_source_code_binding(),
        )
    assert calls == 3


def _write_checkpoint(
    path: Path,
    payload: bytes = b"physical checkpoint\n",
    *,
    revision: str = REVISION,
    branch: str = min_snr_pilot.PILOT_BRANCH,
) -> None:
    path.write_bytes(payload)
    integrity = {
        "schema_version": 1,
        "checkpoint": path.name,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_format_version": 1,
        "step": min_snr_pilot.PILOT_STEP,
        "git_revision": revision,
        "git_branch": branch,
        "git_dirty": False,
        "dataset_identity_sha256": min_snr_pilot.DATASET_IDENTITY_SHA256,
        "runtime_environment_sha256": min_snr_pilot.LEGACY_RUNTIME_ENVIRONMENT_SHA256,
    }
    path.with_name(f"{path.name}.integrity.json").write_text(
        json.dumps(integrity), encoding="utf-8"
    )


def test_physical_checkpoint_replay_rejects_payload_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_step_00050000.pt"
    _write_checkpoint(checkpoint)
    evidence = guard._verify_checkpoint(
        checkpoint,
        label="pilot cofitok",
        expected_step=min_snr_pilot.PILOT_STEP,
    )
    assert evidence["physical_sha256_verified"] is True

    checkpoint.write_bytes(b"tampered checkpoint\n")
    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        guard._verify_checkpoint(
            checkpoint,
            label="pilot cofitok",
            expected_step=min_snr_pilot.PILOT_STEP,
        )


def _legacy_audit_case(tmp_path: Path, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = tmp_path / method
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step_00050000.pt"
    _write_checkpoint(
        checkpoint,
        payload=f"{method} checkpoint\n".encode("ascii"),
        revision=min_snr_pilot.LEGACY_REVISION,
        branch=min_snr_pilot.LEGACY_BRANCH,
    )
    integrity_path = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    latest = {
        "schema_version": 1,
        "step": min_snr_pilot.PILOT_STEP,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_format_version": 1,
        "integrity_manifest": integrity_path.name,
        "git_revision": min_snr_pilot.LEGACY_REVISION,
        "git_branch": min_snr_pilot.LEGACY_BRANCH,
        "git_dirty": False,
        "dataset_identity_sha256": min_snr_pilot.DATASET_IDENTITY_SHA256,
        "runtime_environment_sha256": (
            min_snr_pilot.LEGACY_RUNTIME_ENVIRONMENT_SHA256
        ),
    }
    latest_path = run_dir / "latest.json"
    latest_path.write_bytes(
        (json.dumps(latest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    rows = [
        {"step": 1, "samples_seen": min_snr_pilot.EFFECTIVE_BATCH_SIZE, "total": 1.0},
        {
            "step": min_snr_pilot.PILOT_STEP,
            "samples_seen": min_snr_pilot.IMAGES_PER_METHOD,
            "total": 0.1,
        },
    ]
    metrics_path = run_dir / "train_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    checkpoint_identity = guard.file_identity(checkpoint)
    audit = {
        "schema_version": 1,
        "role": "generation_checkpoint_physical_integrity_milestone_audit",
        "status": "pass",
        "checkpoint": {
            "step": min_snr_pilot.PILOT_STEP,
            "payload": checkpoint_identity,
            "integrity": integrity,
            "integrity_manifest": guard.file_identity(integrity_path),
            "physical_sha256_verified": True,
        },
        "latest_pointer": {
            "content": latest,
            "identity": guard.file_identity(latest_path),
            "exact_target_binding": True,
        },
        "metrics": {
            "identity": guard.file_identity(metrics_path),
            "first_step": 1,
            "last_step": min_snr_pilot.PILOT_STEP,
            "row_count": len(rows),
            "strictly_increasing": True,
            "samples_seen_binding_verified": True,
            "target_row": rows[-1],
        },
        "training_checkout": {
            "revision": min_snr_pilot.LEGACY_REVISION,
            "branch": min_snr_pilot.LEGACY_BRANCH,
            "tracked_dirty": False,
        },
    }
    return audit, checkpoint_identity


def test_preparation_replay_physically_binds_legacy_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audits: dict[str, dict[str, Any]] = {}
    checkpoint_identities: dict[str, dict[str, Any]] = {}
    for method in min_snr_pilot.METHODS:
        audits[method], checkpoint_identities[method] = _legacy_audit_case(
            tmp_path,
            method,
        )

    source_payloads = {name: {"name": name} for name in guard.PREPARATION_SOURCE_NAMES}
    source_payloads["legacy_cofitok_checkpoint_audit_50k"] = audits["cofitok"]
    source_payloads["legacy_dense_checkpoint_audit_50k"] = audits["dense_identity"]
    source_identities: dict[str, dict[str, Any]] = {}
    for name, payload in source_payloads.items():
        path = tmp_path / "preparation_sources" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        source_identities[name] = guard.file_identity(path)
    semantic_identities: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(min_snr_pilot.TRAINING_SEMANTIC_FILES):
        path = tmp_path / "semantic" / f"source_{index}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        semantic_identities[name] = guard.file_identity(path)
    prepared = {
        "builder_git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": min_snr_pilot.PILOT_BRANCH,
            "tracked_dirty": False,
        },
        "source_identities": source_identities,
        "training_source_delta": {
            "semantic_file_identities": semantic_identities,
        },
        "legacy_control_policy": {
            "controls": {
                method: {
                    "checkpoint": checkpoint_identities[method],
                    "step": min_snr_pilot.PILOT_STEP,
                    "samples_seen": min_snr_pilot.IMAGES_PER_METHOD,
                    "physical_sha256_verified": True,
                }
                for method in min_snr_pilot.METHODS
            }
        },
        "output_root": tmp_path.resolve().as_posix(),
    }
    monkeypatch.setattr(
        min_snr_pilot,
        "validate_preparation",
        lambda report: copy.deepcopy(dict(report)),
    )
    monkeypatch.setattr(
        min_snr_pilot,
        "build_preparation",
        lambda **kwargs: copy.deepcopy(prepared),
    )
    evidence = guard._verify_preparation_sources(prepared)
    assert evidence["logical_replay_exact"] is True
    assert set(evidence["legacy_controls"]) == set(min_snr_pilot.METHODS)
    assert all(
        row["preparation_control_match"] is True
        for row in evidence["legacy_controls"].values()
    )

    tampered = copy.deepcopy(prepared)
    tampered["legacy_control_policy"]["controls"]["cofitok"]["checkpoint"][
        "sha256"
    ] = "0" * 64
    monkeypatch.setattr(
        min_snr_pilot,
        "build_preparation",
        lambda **kwargs: copy.deepcopy(tampered),
    )
    with pytest.raises(ValueError, match="preparation control differs"):
        guard._verify_preparation_sources(tampered)


def test_classifier_replay_rejects_weight_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = b"classifier weights\n"
    path = tmp_path / "resnet50.pth"
    path.write_bytes(weights)
    digest = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr(guard, "CLASS_FIDELITY_CLASSIFIER_BYTES", len(weights))
    monkeypatch.setattr(guard, "CLASS_FIDELITY_CLASSIFIER_SHA256", digest)
    report = {
        "classifier": {
            "name": guard.CLASS_FIDELITY_CLASSIFIER_NAME,
            "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
            "weights_path": path.resolve().as_posix(),
            "weights_bytes": len(weights),
            "weights_sha256": digest,
            "num_classes": 1000,
            "categories_sha256": guard.CLASS_FIDELITY_CATEGORIES_SHA256,
            "preprocessing": guard.CLASS_FIDELITY_PREPROCESSING,
        }
    }
    assert guard._verify_classifier(report, label="arm", cache={})[
        "physical_sha256_verified"
    ] is True

    path.write_bytes(b"tampered weights\n")
    with pytest.raises(ValueError, match="physical identity differs"):
        guard._verify_classifier(report, label="arm", cache={})


def test_source_report_replay_rejects_report_tampering(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    source.write_text('{"status":"completed"}\n', encoding="utf-8")
    claimed = guard.file_identity(source)
    assert guard._actual_source_identities({"report": claimed})["report"] == claimed

    source.write_text('{"status":"failed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="terminal source changed"):
        guard._actual_source_identities({"report": claimed})


def test_min_snr_metrics_replay_rejects_weighted_loss_above_unweighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(min_snr_pilot, "PILOT_STEP", 2)
    rows = [
        {
            "step": step,
            "samples_seen": step * min_snr_pilot.EFFECTIVE_BATCH_SIZE,
            "total": 1.0,
            "epsilon": 0.5,
            "epsilon_unweighted": 1.0,
            "min_snr_weight_mean": 0.8,
            "learning_rate": 1e-4,
            "grad_norm": 0.1,
        }
        for step in (1, 2)
    ]
    evidence = guard._verify_min_snr_metrics(rows, method="cofitok")
    assert evidence["epsilon_le_unweighted_all_rows"] is True
    assert evidence["downweighted_row_count"] == 2

    rows[1]["epsilon"] = 1.1
    with pytest.raises(ValueError, match="weighted epsilon exceeds unweighted"):
        guard._verify_min_snr_metrics(rows, method="cofitok")


def test_min_snr_metrics_replay_rejects_nonfinite_auxiliary_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(min_snr_pilot, "PILOT_STEP", 2)
    rows = [
        {
            "step": step,
            "samples_seen": step * min_snr_pilot.EFFECTIVE_BATCH_SIZE,
            "total": 1.0,
            "epsilon": 0.5,
            "epsilon_unweighted": 1.0,
            "min_snr_weight_mean": 0.8,
            "learning_rate": 1e-4,
            "grad_norm": 0.1,
            "denoise_path_component": float("nan") if step == 2 else 0.1,
        }
        for step in (1, 2)
    ]
    with pytest.raises(ValueError, match="non-finite numeric value"):
        guard._verify_min_snr_metrics(rows, method="cofitok")


def _sampling_case(tmp_path: Path) -> dict[str, Any]:
    generated = tmp_path / "samples" / "prefix_8"
    generated.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (4, 4), color=(index, index, index)).save(
            generated / f"{index:06d}.png"
        )
    checkpoint = tmp_path / "checkpoint_step_00050000.pt"
    _write_checkpoint(checkpoint)
    integrity_path = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    checkpoint_sha = file_sha256(checkpoint)
    sample_sha = sample_set_sha256(find_images(generated))
    runtime_environment = {"schema_version": 1, "device": {"type": "cpu"}}
    runtime_sha = runtime_environment_sha256(runtime_environment)
    git = {
        "revision": REVISION,
        "branch": min_snr_pilot.PILOT_BRANCH,
        "tracked_dirty": False,
    }
    sampling = {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "start_index": 0,
        "num_samples": 2,
        "sample_steps": 1,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, 1),
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "precision": "bf16",
        "image_shape": [3, 4, 4],
        "prefix_budgets": [8],
        "seed": min_snr_pilot.SAMPLE_SEED,
        "class_schedule": "balanced_modulo",
    }
    output_dirs = {"8": generated.resolve().as_posix()}
    sample_sets = {"8": {"count": 2, "sha256": sample_sha}}
    manifest_path = generated.parent / "sampling_manifest.json"
    manifest = {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "git": git,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_sha,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_integrity_manifest": integrity_path.resolve().as_posix(),
        "checkpoint_step": min_snr_pilot.PILOT_STEP,
        "weights": "ema",
        "sampling": sampling,
        "output_dirs": output_dirs,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = file_sha256(manifest_path)
    progress_path = generated.parent / "sampling_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                # sampling_progress.json uses its own schema version (v1),
                # rather than the sampling_report schema (v6).
                "schema_version": 1,
                "status": "completed",
                "sampling_manifest_sha256": manifest_sha,
                "completed_samples": 2,
                "prefix_budgets": [8],
                "sample_sets": sample_sets,
                "invocation": 1,
                "cumulative_elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    sampling_report_path = generated.parent / "sampling_report.json"
    sampling_report_path.write_text(
        json.dumps(
            {
                **manifest,
                "schema_version": SAMPLING_REPORT_SCHEMA_VERSION,
                "status": "completed",
                "sampling_manifest_sha256": manifest_sha,
                "sampling_progress": progress_path.resolve().as_posix(),
                "sample_sets": sample_sets,
            }
        ),
        encoding="utf-8",
    )
    provenance = validate_sampling_provenance(
        sampling_report_path, generated, find_images(generated)
    )
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "git": git,
        "runtime_environment_sha256": runtime_sha,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_integrity_manifest": integrity_path.resolve().as_posix(),
        "checkpoint_step": min_snr_pilot.PILOT_STEP,
        "weights": "ema",
        "requested_weights": "ema",
        "request": {
            "batch_size": 32,
            "prefix_budget": 8,
            "precision": "bf16",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "warmup_forwards": 0,
            "measured_forwards": 1,
        },
        "result": {"output_finite": True, "output_shape": [32, 3, 256, 256]},
    }
    generation = {
        "schema_version": guard.GENERATION_METRICS_REPORT_SCHEMA_VERSION,
        "role": guard.GENERATION_METRICS_REPORT_ROLE,
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "git": git,
        "paths": {
            "generated_dir": generated.resolve().as_posix(),
            "sampling_report": sampling_report_path.resolve().as_posix(),
        },
        "sample_provenance": provenance,
        "counts": {
            "generated_image_count": 2,
            "real_image_count": guard.REAL_IMAGE_COUNT,
        },
        "metrics": {
            "frechet_inception_distance": 10.0,
            "inception_score_mean": 2.0,
            "inception_score_std": 0.1,
            "precision": 0.5,
            "recall": 0.5,
        },
        "runtime": {"elapsed_seconds": 1.0},
    }
    return {"generation": generation, "preflight": preflight, "generated": generated}


def test_sampling_replay_rejects_tampered_png_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(min_snr_pilot, "SAMPLE_COUNT", 2)
    monkeypatch.setattr(min_snr_pilot, "SAMPLE_STEPS", 1)
    case = _sampling_case(tmp_path)
    evidence = guard._verify_sampling_evidence(
        arm="pilot_gamma5_cofitok",
        generation=case["generation"],
        preflight=case["preflight"],
        expected_prefix=8,
        expected_revision=REVISION,
        expected_branch=min_snr_pilot.PILOT_BRANCH,
    )
    assert evidence["sampling"]["sample_set_sha256"]

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
        case["generated"] / "000001.png"
    )
    with pytest.raises(ValueError, match="sample-set SHA256"):
        guard._verify_sampling_evidence(
            arm="pilot_gamma5_cofitok",
            generation=case["generation"],
            preflight=case["preflight"],
            expected_prefix=8,
            expected_revision=REVISION,
            expected_branch=min_snr_pilot.PILOT_BRANCH,
        )


def test_sampling_replay_rejects_nonfinite_completed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(min_snr_pilot, "SAMPLE_COUNT", 2)
    monkeypatch.setattr(min_snr_pilot, "SAMPLE_STEPS", 1)
    case = _sampling_case(tmp_path)
    progress_path = case["generated"].parent / "sampling_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["cumulative_elapsed_seconds"] = float("nan")
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    sampling_report_path = case["generated"].parent / "sampling_report.json"
    case["generation"]["sample_provenance"] = validate_sampling_provenance(
        sampling_report_path,
        case["generated"],
        find_images(case["generated"]),
    )

    with pytest.raises(ValueError, match="non-finite numeric value"):
        guard._verify_sampling_evidence(
            arm="pilot_gamma5_cofitok",
            generation=case["generation"],
            preflight=case["preflight"],
            expected_prefix=8,
            expected_revision=REVISION,
            expected_branch=min_snr_pilot.PILOT_BRANCH,
        )

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from cofitok.generation.terminal_snr_confirmation_arm import (
    ARM_VALIDATION_BOUNDARY,
    build_terminal_snr_confirmation_arm_validation,
    validate_terminal_snr_confirmation_sampling_preflight,
    validate_terminal_snr_confirmation_arm_validation,
)
from cofitok.generation.terminal_snr_confirmation_execution import (
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES, ARM_SPECS
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256


EXECUTION = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/terminal-snr-confirmation",
    "tracked_dirty": False,
}
SCREEN_EXECUTION = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "analysis/terminal-snr-screen",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 1,
        "sha256": sha256(name.encode()).hexdigest(),
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _preflight_report(tmp_path: Path, *, source_git=None) -> tuple[Path, dict]:
    arm = "control_cofitok"
    checkpoint = {
        "path": (tmp_path / "checkpoint_step_00010000.pt").as_posix(),
        "bytes": 123,
        "sha256": "3" * 64,
        "integrity_manifest": {
            "path": (tmp_path / "checkpoint_step_00010000.pt.integrity.json").as_posix(),
            "bytes": 12,
            "sha256": "9" * 64,
        },
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "git": {
            "revision": EXECUTION["revision"],
            "branch": EXECUTION["branch"],
            "tracked_dirty": False,
        },
        "source_git": source_git,
        "runtime_environment": {"schema_version": 1},
        "runtime_environment_sha256": "4" * 64,
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_integrity_manifest": checkpoint["integrity_manifest"]["path"],
        "checkpoint_step": 10_000,
        "weights": "ema",
        "requested_weights": "ema",
        "artifact_type": "training_checkpoint",
        "source_checkpoint_sha256": None,
        "source_runtime_environment_sha256": None,
        "release_authorization": None,
        "release_authorization_required": False,
        "completion_authorization": None,
        "completion_authorization_required": False,
        "device": "cuda",
        "request": {
            "batch_size": 4,
            "effective_model_batch_size": 8,
            "forward_passes": 1,
            "image_shape": [3, 256, 256],
            "prefix_budget": 8,
            "token_count": 8,
            "precision": "bf16",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "class_conditional": True,
            "timestep": 999,
            "warmup_forwards": 0,
            "measured_forwards": 1,
        },
        "result": {
            "elapsed_seconds": 1.25,
            "output_shape": [4, 3, 256, 256],
            "output_dtype": "torch.float32",
            "output_finite": True,
            "cuda_memory_after_forward": {"peak_allocated_bytes": 1024},
            "device_total_memory_bytes": 2048,
        },
    }
    path = tmp_path / "sampling_preflight.json"
    _write(path, report)
    return path, checkpoint


def test_validates_raw_checkpoint_real_forward_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, checkpoint = _preflight_report(tmp_path)
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.runtime_environment_sha256",
        lambda _: "4" * 64,
    )
    validated = validate_terminal_snr_confirmation_sampling_preflight(
        path,
        arm="control_cofitok",
        expected_checkpoint=checkpoint,
        expected_execution_git=EXECUTION,
        expected_screen_git=SCREEN_EXECUTION,
        expected_runtime_environment_sha256="4" * 64,
    )
    assert validated["output_finite"] is True
    assert validated["source_git"] is None
    assert validated["checkpoint_screen_git"]["revision"] == SCREEN_EXECUTION["revision"]


def test_preflight_rejects_inference_source_git_for_raw_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, checkpoint = _preflight_report(
        tmp_path,
        source_git={"revision": "f" * 40, "branch": "wrong", "dirty": False},
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.runtime_environment_sha256",
        lambda _: "4" * 64,
    )
    with pytest.raises(ValueError, match="sampling preflight differs"):
        validate_terminal_snr_confirmation_sampling_preflight(
            path,
            arm="control_cofitok",
            expected_checkpoint=checkpoint,
            expected_execution_git=EXECUTION,
            expected_screen_git=SCREEN_EXECUTION,
            expected_runtime_environment_sha256="4" * 64,
        )


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, same_sample=False):
    arm = ARM_NAMES[0]
    spec = ARM_SPECS[arm]
    screen_sample_sha = "1" * 64
    confirmation_sample_sha = screen_sample_sha if same_sample else "2" * 64
    checkpoint = {
        "path": "/tmp/checkpoint_step_00010000.pt",
        "bytes": 123,
        "sha256": "3" * 64,
        "integrity_manifest": _identity("sidecar"),
    }
    screen = {
        "arm": arm,
        "condition": spec["condition"],
        "method": spec["method"],
        "endpoint_fraction": spec["endpoint_fraction"],
        "execution_git": SCREEN_EXECUTION,
        "training": {"checkpoint": checkpoint},
        "sampling": {"sample_set_sha256": screen_sample_sha},
        "checkpoint_evaluation": {"summary": {"ordered_rank_by_path_auc": 1}},
        "rollout": {"terminal_raw_x0_clipping": {"raw_x0_clip_fraction": 0.9}},
    }
    screen_path = tmp_path / "screen.json"
    _write(screen_path, screen)
    screen_id = file_identity(screen_path)
    launch = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "execution_checkout": EXECUTION,
        "screen_execution_checkout": SCREEN_EXECUTION,
        "runtime_environment_sha256": "4" * 64,
        "source_evidence": {
            "terminal_snr_screen_arm_validations": {arm: screen_id}
        },
        "frozen_arms": {
            arm: {
                "screen_arm_validation": screen_id,
                "condition": spec["condition"],
                "method": spec["method"],
                "endpoint_fraction": spec["endpoint_fraction"],
                "checkpoint": checkpoint,
            }
        },
        "output_dirs": {arm: (tmp_path / "confirmation" / arm).as_posix()},
    }
    launch_path = tmp_path / "launch.json"
    _write(launch_path, launch)
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.validate_terminal_snr_confirmation_launch_receipt_contract",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.validate_terminal_snr_screen_arm_validation",
        lambda value: value,
    )
    preflight_path = tmp_path / "preflight.json"
    _write(preflight_path, {"status": "passed"})
    preflight = {
        "report": file_identity(preflight_path),
        "status": "passed",
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_step": 10_000,
        "runtime_environment_sha256": "4" * 64,
        "execution_git": {
            "revision": EXECUTION["revision"],
            "branch": EXECUTION["branch"],
            "tracked_dirty": False,
        },
        "checkpoint_screen_git": {
            "revision": SCREEN_EXECUTION["revision"],
            "branch": SCREEN_EXECUTION["branch"],
            "tracked_dirty": False,
        },
        "source_git": None,
        "request": {"batch_size": 4},
        "output_finite": True,
        "elapsed_seconds": 1.0,
        "peak_allocated_bytes": 1,
        "device_total_memory_bytes": 2,
    }
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.validate_terminal_snr_confirmation_sampling_preflight",
        lambda *args, **kwargs: copy.deepcopy(preflight),
    )
    sample = {
        "report": _identity("sampling"),
        "manifest": _identity("manifest"),
        "progress": _identity("progress"),
        "generated_dir": (tmp_path / "confirmation" / arm / "samples").as_posix(),
        "sample_count": 10_000,
        "sample_set_sha256": confirmation_sample_sha,
        "checkpoint_sha256": checkpoint["sha256"],
        "runtime_environment_sha256": "4" * 64,
        "git": {
            "revision": EXECUTION["revision"],
            "branch": EXECUTION["branch"],
            "tracked_dirty": False,
        },
        "sampling": {"seed": 2028},
        "provenance": {"sample_set_sha256": confirmation_sample_sha},
    }
    distribution = {
        "report": _identity("metrics"),
        "git": sample["git"],
        "metrics": {"fid": 50.0, "precision": 0.2, "recall": 0.2},
        "counts": {"generated_image_count": 10_000},
        "real_set": {"sha256": "5" * 64},
        "runtime_environment_sha256": "6" * 64,
    }
    class_fidelity = {
        "report": _identity("class"),
        "git": sample["git"],
        "metrics": {"top1_accuracy": 0.2, "top5_accuracy": 0.5},
        "classifier": {"sha256": "7" * 64},
        "runtime_environment_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.shared_arm._sampling_evidence",
        lambda *args, **kwargs: copy.deepcopy(sample),
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.shared_arm._distribution_evidence",
        lambda *args, **kwargs: copy.deepcopy(distribution),
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.shared_arm._class_fidelity_evidence",
        lambda *args, **kwargs: copy.deepcopy(class_fidelity),
    )
    return {
        "arm": arm,
        "launch_receipt_path": launch_path,
        "expected_launch_receipt_sha256": file_sha256(launch_path),
        "screen_arm_validation_path": screen_path,
        "expected_screen_arm_validation_sha256": file_sha256(screen_path),
        "sampling_preflight_report_path": preflight_path,
        "sampling_report_path": tmp_path / "sampling.json",
        "metrics_report_path": tmp_path / "metrics.json",
        "class_fidelity_report_path": tmp_path / "class.json",
    }


def test_builds_replayable_distinct_confirmation_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _inputs(tmp_path, monkeypatch)
    report = build_terminal_snr_confirmation_arm_validation(**kwargs)
    assert report["confirmation_sample_set_distinct"] is True
    assert report["frozen_training"]["training_performed"] is False
    assert report["authorization_boundary"] == ARM_VALIDATION_BOUNDARY
    assert validate_terminal_snr_confirmation_arm_validation(report) == report


def test_rejects_reused_screen_sample_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _inputs(tmp_path, monkeypatch, same_sample=True)
    with pytest.raises(ValueError, match="sample stream is not distinct"):
        build_terminal_snr_confirmation_arm_validation(**kwargs)


def test_rejects_evaluator_runtime_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _inputs(tmp_path, monkeypatch)
    original = shared = __import__(
        "cofitok.generation.terminal_snr_confirmation_arm",
        fromlist=["shared_arm"],
    ).shared_arm._class_fidelity_evidence

    def mismatched(*args, **kwargs):
        row = original(*args, **kwargs)
        row["runtime_environment_sha256"] = "8" * 64
        return row

    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_arm.shared_arm._class_fidelity_evidence",
        mismatched,
    )
    with pytest.raises(ValueError, match="evaluator runtime identities differ"):
        build_terminal_snr_confirmation_arm_validation(**kwargs)

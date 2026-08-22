from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_generation_checkpoint_integrity_audit_replay.py"
)
READ_ONLY_SCOPE = {
    "read_only_checkpoint_verification": True,
    "gpu_required": False,
    "training_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
}


def load_verifier() -> ModuleType:
    name = "cofitok_generation_checkpoint_audit_replay_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def fixture(tmp_path: Path) -> dict[str, object]:
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "checkpoint_step_00000025.pt"
    checkpoint.write_bytes(b"checkpoint-25")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    dataset_sha = "d" * 64
    runtime_sha = "e" * 64
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_format_version": 1,
        "step": 25,
        "git_revision": "a" * 40,
        "git_branch": "training-branch",
        "git_dirty": False,
        "dataset_identity_sha256": dataset_sha,
        "runtime_environment_sha256": runtime_sha,
    }
    sidecar = Path(f"{checkpoint}.integrity.json")
    write_json(sidecar, integrity)
    latest = run / "latest.json"
    latest_content = {**integrity, "integrity_manifest": sidecar.name}
    write_json(latest, latest_content)
    metrics = run / "train_metrics.jsonl"
    rows = [
        {"step": step, "samples_seen": step * 8, "loss": 1.0 / step}
        for step in (5, 10, 15, 20, 25)
    ]
    metrics.write_bytes(
        "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    )
    auditor_checkout = tmp_path / "auditor"
    auditor = auditor_checkout / "scripts" / "wait_generation_checkpoint_integrity_audit.py"
    auditor.parent.mkdir(parents=True)
    auditor.write_text("# waiter source\n", encoding="utf-8")
    auditor_git = {
        "path": auditor_checkout.resolve().as_posix(),
        "revision": "c" * 40,
        "tree": "d" * 40,
        "branch": "auditor-branch",
        "tracked_dirty": False,
    }
    training = tmp_path / "training"
    training.mkdir()
    training_git = {
        "path": training.resolve().as_posix(),
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "training-branch",
        "tracked_dirty": False,
    }
    report = {
        "schema_version": 1,
        "role": "generation_checkpoint_physical_integrity_milestone_audit",
        "status": "pass",
        "scope": READ_ONLY_SCOPE,
        "training_checkout": training_git,
        "run_dir": str(run.resolve()),
        "checkpoint": {
            "step": 25,
            "payload": identity(checkpoint),
            "integrity_manifest": identity(sidecar),
            "integrity": integrity,
            "physical_sha256_verified": True,
        },
        "latest_pointer": {
            "identity": identity(latest),
            "content": latest_content,
            "exact_target_binding": True,
        },
        "metrics": {
            "identity": identity(metrics),
            "row_count": len(rows),
            "first_step": 5,
            "last_step": 25,
            "strictly_increasing": True,
            "samples_seen_binding": "samples_seen == step * 8",
            "samples_seen_binding_verified": True,
            "target_row": rows[-1],
        },
    }
    report_path = tmp_path / "audit.json"
    write_json(report_path, report)
    status = {
        "schema_version": 1,
        "role": "generation_checkpoint_physical_integrity_milestone_waiter",
        "status": "pass",
        "detail": "checkpoint_physical_integrity_verified",
        "checkpoint_step": 25,
        "run_dir": str(run.resolve()),
        "audit_output": report_path.resolve().as_posix(),
        "scope": READ_ONLY_SCOPE,
        "audit_output_identity": identity(report_path),
        "waiter_source": identity(auditor),
    }
    status_path = tmp_path / "status.json"
    write_json(status_path, status)
    return {
        "run": run,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "dataset_sha": dataset_sha,
        "runtime_sha": runtime_sha,
        "metrics": metrics,
        "latest": latest,
        "auditor": auditor,
        "auditor_checkout": auditor_checkout,
        "auditor_git": auditor_git,
        "training": training,
        "training_git": training_git,
        "report": report_path,
        "status": status_path,
    }


def replay(module: ModuleType, data: dict[str, object], monkeypatch) -> dict[str, object]:
    def fake_git_identity(*_args, **kwargs):
        if kwargs["label"] == "checkpoint integrity auditor checkout":
            return data["auditor_git"]
        return data["training_git"]

    monkeypatch.setattr(module, "require_git_identity", fake_git_identity)
    return module.verify_replay(
        audit_report_path=data["report"],
        expected_audit_report_sha256=identity(data["report"])["sha256"],
        waiter_status_path=data["status"],
        expected_waiter_status_sha256=identity(data["status"])["sha256"],
        auditor_source_path=data["auditor"],
        expected_auditor_source_sha256=identity(data["auditor"])["sha256"],
        auditor_checkout=data["auditor_checkout"],
        expected_auditor_revision="c" * 40,
        expected_auditor_tree="d" * 40,
        expected_auditor_branch="auditor-branch",
        training_checkout=data["training"],
        expected_training_revision="a" * 40,
        expected_training_tree="b" * 40,
        expected_training_branch="training-branch",
        expected_checkpoint_step=25,
        expected_checkpoint_sha256=data["checkpoint_sha"],
        expected_dataset_sha256=data["dataset_sha"],
        expected_runtime_sha256=data["runtime_sha"],
        effective_batch=8,
        verifier_git={"revision": "v" * 40},
    )


def test_replay_passes_while_latest_is_still_exact(tmp_path: Path, monkeypatch) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    report = replay(module, data, monkeypatch)
    assert report["status"] == "pass"
    assert (
        report["latest_pointer_replay"]["relation"]
        == "exact_historical_pointer_still_current"
    )
    assert report["metrics_replay"]["audit_time_prefix_byte_exact"] is True
    assert report["checkpoint"]["physical_sha256_reverified"] is True


def test_replay_survives_appended_metrics_and_advanced_latest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    with data["metrics"].open("ab") as handle:
        handle.write(
            (json.dumps({"step": 30, "samples_seen": 240, "loss": 0.01}) + "\n").encode(
                "utf-8"
            )
        )
    checkpoint = data["run"] / "checkpoint_step_00000030.pt"
    checkpoint.write_bytes(b"checkpoint-30")
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_format_version": 1,
        "step": 30,
        "git_revision": "a" * 40,
        "git_branch": "training-branch",
        "git_dirty": False,
        "dataset_identity_sha256": data["dataset_sha"],
        "runtime_environment_sha256": data["runtime_sha"],
    }
    sidecar = Path(f"{checkpoint}.integrity.json")
    write_json(sidecar, integrity)
    write_json(data["latest"], {**integrity, "integrity_manifest": sidecar.name})
    report = replay(module, data, monkeypatch)
    assert report["latest_pointer_replay"]["relation"] == "advanced_with_consistent_metadata"
    assert report["metrics_replay"]["current_observed_prefix"]["last_step"] == 30
    assert report["metrics_replay"]["audit_time_prefix"]["last_step"] == 25


def test_replay_rejects_historical_metric_mutation(tmp_path: Path, monkeypatch) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    rows = data["metrics"].read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["loss"] = 99.0
    rows[0] = json.dumps(first)
    data["metrics"].write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="byte prefix differs"):
        replay(module, data, monkeypatch)


def test_replay_rejects_checkpoint_mutation(tmp_path: Path, monkeypatch) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    data["checkpoint"].write_bytes(b"mutated checkpoint")
    with pytest.raises(ValueError, match="physical identity differs"):
        replay(module, data, monkeypatch)


def test_replay_rejects_waiter_report_binding_drift(tmp_path: Path, monkeypatch) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    status = json.loads(data["status"].read_text(encoding="utf-8"))
    status["audit_output_identity"]["sha256"] = "f" * 64
    write_json(data["status"], status)
    with pytest.raises(ValueError, match="waiter status contract"):
        replay(module, data, monkeypatch)


def test_replay_rejects_advanced_latest_metadata_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    checkpoint = data["run"] / "checkpoint_step_00000030.pt"
    checkpoint.write_bytes(b"checkpoint-30")
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_format_version": 1,
        "step": 30,
        "git_revision": "f" * 40,
        "git_branch": "training-branch",
        "git_dirty": False,
        "dataset_identity_sha256": data["dataset_sha"],
        "runtime_environment_sha256": data["runtime_sha"],
    }
    sidecar = Path(f"{checkpoint}.integrity.json")
    write_json(sidecar, integrity)
    write_json(data["latest"], {**integrity, "integrity_manifest": sidecar.name})
    with pytest.raises(ValueError, match="advanced latest pointer git_revision differs"):
        replay(module, data, monkeypatch)


def test_replay_uses_complete_prefix_during_partial_metrics_append(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    with data["metrics"].open("ab") as handle:
        handle.write(b'{"step": 30')
    report = replay(module, data, monkeypatch)
    current = report["metrics_replay"]["current_observed_prefix"]
    assert current["last_step"] == 25
    assert current["complete_line_prefix"] is True


def test_replay_rejects_integrity_sidecar_mutation(tmp_path: Path, monkeypatch) -> None:
    module = load_verifier()
    data = fixture(tmp_path)
    sidecar = Path(f"{data['checkpoint']}.integrity.json")
    integrity = json.loads(sidecar.read_text(encoding="utf-8"))
    integrity["runtime_environment_sha256"] = "0" * 64
    write_json(sidecar, integrity)
    with pytest.raises(ValueError, match="physical identity differs"):
        replay(module, data, monkeypatch)

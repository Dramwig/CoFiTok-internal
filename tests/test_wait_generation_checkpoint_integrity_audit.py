from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "wait_generation_checkpoint_integrity_audit.py"
)


def load_waiter() -> ModuleType:
    name = "cofitok_generation_checkpoint_integrity_waiter_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_checkpoint_fixture(root: Path, *, step: int = 25) -> tuple[Path, str, str]:
    checkpoint = root / f"checkpoint_step_{step:08d}.pt"
    checkpoint.write_bytes(b"physical checkpoint payload")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    dataset_sha = "d" * 64
    runtime_sha = "e" * 64
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_format_version": 1,
        "step": step,
        "git_revision": "a" * 40,
        "git_branch": "training-branch",
        "git_dirty": False,
        "dataset_identity_sha256": dataset_sha,
        "runtime_environment_sha256": runtime_sha,
    }
    integrity_path = root / f"{checkpoint.name}.integrity.json"
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({**integrity, "integrity_manifest": integrity_path.name}),
        encoding="utf-8",
    )
    rows = [
        {"step": value, "samples_seen": value * 8}
        for value in (5, 10, 15, 20, 25)
    ]
    (root / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return checkpoint, dataset_sha, runtime_sha


def test_target_ready_requires_exact_latest_pointer(tmp_path: Path) -> None:
    module = load_waiter()
    ready, detail = module.target_is_ready(tmp_path, 25)
    assert (ready, detail) == (False, "checkpoint_missing")

    write_checkpoint_fixture(tmp_path)
    assert module.target_is_ready(tmp_path, 25) == (True, "ready")

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    latest["step"] = 30
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    try:
        module.target_is_ready(tmp_path, 25)
    except ValueError as error:
        assert "advanced beyond" in str(error)
    else:
        raise AssertionError("advanced latest.json must fail closed")


def test_audit_binds_physical_checkpoint_latest_metrics_and_scope(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    module = load_waiter()
    _, dataset_sha, runtime_sha = write_checkpoint_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_checkout",
        lambda *_args, **_kwargs: {
            "path": "/training",
            "revision": "a" * 40,
            "branch": "training-branch",
            "tree": "b" * 40,
            "tracked_dirty": False,
        },
    )

    report = module.audit_checkpoint(
        run_dir=tmp_path,
        checkpoint_step=25,
        training_checkout=tmp_path,
        expected_revision="a" * 40,
        expected_branch="training-branch",
        expected_tree="b" * 40,
        expected_dataset_sha256=dataset_sha,
        expected_runtime_sha256=runtime_sha,
        effective_batch=8,
    )

    assert report["status"] == "pass"
    assert report["checkpoint"]["physical_sha256_verified"] is True
    assert report["latest_pointer"]["exact_target_binding"] is True
    assert report["metrics"]["strictly_increasing"] is True
    assert report["metrics"]["samples_seen_binding_verified"] is True
    assert report["scope"] == {
        "read_only_checkpoint_verification": True,
        "gpu_required": False,
        "training_process_signals_allowed": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
    }


def test_audit_rejects_samples_seen_drift(tmp_path: Path, monkeypatch: object) -> None:
    module = load_waiter()
    _, dataset_sha, runtime_sha = write_checkpoint_fixture(tmp_path)
    rows = (tmp_path / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    last["samples_seen"] += 1
    rows[-1] = json.dumps(last)
    (tmp_path / "train_metrics.jsonl").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "verify_checkout",
        lambda *_args, **_kwargs: {
            "path": "/training",
            "revision": "a" * 40,
            "branch": "training-branch",
            "tree": "b" * 40,
            "tracked_dirty": False,
        },
    )

    try:
        module.audit_checkpoint(
            run_dir=tmp_path,
            checkpoint_step=25,
            training_checkout=tmp_path,
            expected_revision="a" * 40,
            expected_branch="training-branch",
            expected_tree="b" * 40,
            expected_dataset_sha256=dataset_sha,
            expected_runtime_sha256=runtime_sha,
            effective_batch=8,
        )
    except ValueError as error:
        assert "samples_seen binding" in str(error)
    else:
        raise AssertionError("samples_seen drift must fail closed")

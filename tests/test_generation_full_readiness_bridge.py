from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_generation_full_readiness_bridge as bridge
from scripts.build_generation_full_readiness_bridge import (
    FULL_RUNBOOK,
    BRIDGE_ONLY_PREAMBLE_LINES,
    SOURCE_RUNTIME_VALIDATOR_START,
    TARGET_RUNTIME_VALIDATION_BLOCK,
    SOURCE_SAMPLE_RESERVE,
    TARGET_SAMPLE_RESERVE,
    TRAINING_CRITICAL_PATHS,
    _critical_manifest,
    _verify_runbook_change,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _source_runbook() -> str:
    return (
        SOURCE_RUNTIME_VALIDATOR_START.decode("utf-8")
        + '  --print-selected-runtime)"\n'
        + "storage_preflight() {\n"
        + SOURCE_SAMPLE_RESERVE.decode("utf-8")
        + "}\nmonitor_report_passes() {\n  echo stable\n}\n"
    )


def _target_runbook() -> str:
    preamble = TARGET_RUNTIME_VALIDATION_BLOCK.decode("utf-8")
    for line in BRIDGE_ONLY_PREAMBLE_LINES:
        preamble += line.decode("utf-8")
    return (
        preamble
        + "storage_preflight() {\n"
        + TARGET_SAMPLE_RESERVE.decode("utf-8")
        + "}\nmonitor_report_passes() {\n  echo stable\n}\n"
    )


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    for relative in TRAINING_CRITICAL_PATHS:
        if relative == "src/cofitok":
            _write(root, "src/cofitok/module.py", "VALUE = 1\n")
        elif "." not in Path(relative).name:
            _write(root, f"{relative}/placeholder.py", "VALUE = 1\n")
        else:
            _write(root, relative, "stable\n")
    _write(
        root,
        FULL_RUNBOOK,
        _source_runbook(),
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "source")
    source = _git(root, "rev-parse", "HEAD")
    _write(
        root,
        FULL_RUNBOOK,
        _target_runbook(),
    )
    _git(root, "add", FULL_RUNBOOK)
    _git(root, "commit", "-m", "target")
    target = _git(root, "rev-parse", "HEAD")
    return root, source, target


def test_bridge_accepts_identical_training_blobs_and_runway_only_change(
    tmp_path: Path,
) -> None:
    root, source, target = _repository(tmp_path)

    assert _critical_manifest(root, source) == _critical_manifest(root, target)
    runbook = _verify_runbook_change(
        root,
        source_revision=source,
        target_revision=target,
    )
    assert runbook["authorization_upgrade"] == {
        "readiness_bridge_required": True,
        "frozen_stability_supplemental_required": True,
        "launch_receipt_schema_version": 3,
        "sample_count": {"source": 16_384, "target": 116_640},
    }
    assert runbook["training_execution_identical"] is True
    assert runbook["controlled_preamble_upgrade"] is True
    assert runbook["source_preamble_sha256"] == runbook[
        "normalized_target_preamble_sha256"
    ]
    assert len(runbook["training_execution_sha256"]) == 64


def test_bridge_rejects_training_blob_or_extra_runbook_drift(tmp_path: Path) -> None:
    root, source, target = _repository(tmp_path)
    _write(root, "scripts/train_generation.py", "changed\n")
    _git(root, "add", "scripts/train_generation.py")
    _git(root, "commit", "-m", "training drift")
    training_drift = _git(root, "rev-parse", "HEAD")
    assert _critical_manifest(root, source) != _critical_manifest(
        root, training_drift
    )

    _git(root, "checkout", target)
    _write(
        root,
        FULL_RUNBOOK,
        _target_runbook().replace(
            "  echo stable\n", "  echo changed\n"
        ),
    )
    _git(root, "add", FULL_RUNBOOK)
    _git(root, "commit", "-m", "runbook drift")
    runbook_drift = _git(root, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="training execution changed"):
        _verify_runbook_change(
            root,
            source_revision=source,
            target_revision=runbook_drift,
        )


def test_bridge_rejects_incomplete_target_authorization_preamble(
    tmp_path: Path,
) -> None:
    root, source, target = _repository(tmp_path)
    _git(root, "checkout", target)
    _write(
        root,
        FULL_RUNBOOK,
        _target_runbook().replace(
            BRIDGE_ONLY_PREAMBLE_LINES[4].decode("utf-8"), ""
        ),
    )
    _git(root, "add", FULL_RUNBOOK)
    _git(root, "commit", "-m", "drop bridge binding")

    with pytest.raises(ValueError, match="bridge authorization line differs"):
        _verify_runbook_change(
            root,
            source_revision=source,
            target_revision=_git(root, "rev-parse", "HEAD"),
        )


def test_bridge_rejects_missing_training_execution_marker(tmp_path: Path) -> None:
    root, source, target = _repository(tmp_path)
    _git(root, "checkout", target)
    _write(
        root,
        FULL_RUNBOOK,
        _git(root, "show", f"{target}:{FULL_RUNBOOK}").replace(
            "monitor_report_passes() {", "monitor_report_changed() {"
        ),
    )
    _git(root, "add", FULL_RUNBOOK)
    _git(root, "commit", "-m", "move training marker")

    with pytest.raises(ValueError, match="execution marker"):
        _verify_runbook_change(
            root,
            source_revision=source,
            target_revision=_git(root, "rev-parse", "HEAD"),
        )
def test_bridge_rejects_extra_preamble_drift(tmp_path: Path) -> None:
    root, source, target = _repository(tmp_path)
    _git(root, "checkout", target)
    _write(
        root,
        FULL_RUNBOOK,
        _target_runbook().replace(
            "storage_preflight() {\n",
            "export TRAINING_DRIFT=1\nstorage_preflight() {\n",
        ),
    )
    _git(root, "add", FULL_RUNBOOK)
    _git(root, "commit", "-m", "add preamble drift")

    with pytest.raises(ValueError, match="controlled bridge upgrade"):
        _verify_runbook_change(
            root,
            source_revision=source,
            target_revision=_git(root, "rev-parse", "HEAD"),
        )


def _patch_bridge_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project: Path,
    readiness_path: Path,
    source_deployment: Path,
    target_deployment: Path,
    cofitok_config: Path,
) -> None:
    source_revision = "f" * 40
    target_revision = "a" * 40
    source_branch = "scale/source"
    target_branch = "scale/target"
    deployment_identity = {"path": source_deployment.as_posix()}
    readiness = {
        "source_reports": {"deployment_receipt": deployment_identity},
        "training_run_dirs": ["/runs/cofitok", "/runs/dense"],
        "benchmark_root": "/benchmarks",
    }
    storage_capacity = cofitok_config.parent / "storage_capacity.json"
    verified_readiness = {
        "runtime_selection": {
            "runtime_environment_sha256": "e" * 64,
        },
        "config_contract": {"status": "pass"},
        "storage_capacity": {"status": "pass"},
        "promotion_authorization": {"gate_sha256": "b" * 64},
        "training_run_dirs": ["/runs/cofitok", "/runs/dense"],
        "benchmark_root": "/benchmarks",
    }
    monkeypatch.setattr(
        bridge,
        "git_provenance",
        lambda root: {
            "revision": target_revision,
            "branch": target_branch,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(bridge, "file_sha256", lambda path: "1" * 64)
    monkeypatch.setattr(
        bridge,
        "verify_deployment_receipt",
        lambda report, *, receipt_path, **kwargs: (
            {
                "checkout": {
                    "path": project.as_posix(),
                    "git": {
                        "revision": target_revision,
                        "branch": target_branch,
                        "tracked_dirty": False,
                    },
                }
            }
            if receipt_path == target_deployment
            else {
                "checkout": {
                    "path": "/source-checkout",
                    "git": {
                        "revision": source_revision,
                        "branch": source_branch,
                        "tracked_dirty": False,
                    },
                }
            }
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        bridge,
        "_read_json",
        lambda path: (
            readiness
            if path == readiness_path
            else (
                {"filesystem": {"path": project.as_posix()}}
                if path == storage_capacity
                else {}
            )
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_source_paths_from_readiness",
        lambda report: {
            "cofitok_config": cofitok_config,
            "storage_capacity": storage_capacity,
        },
    )
    monkeypatch.setattr(
        bridge,
        "source_identity",
        lambda path: (
            deployment_identity
            if path == source_deployment
            else {"path": Path(path).as_posix()}
        ),
    )
    monkeypatch.setattr(
        bridge,
        "verify_readiness_report",
        lambda *args, **kwargs: verified_readiness,
    )
    monkeypatch.setattr(
        bridge,
        "_critical_manifest",
        lambda project, revision: [{"path": "train.py", "git_blob": "c" * 40}],
    )
    monkeypatch.setattr(
        bridge,
        "_verify_runbook_change",
        lambda *args, **kwargs: {"training_execution_identical": True},
    )


def test_bridge_historical_replay_does_not_require_current_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    readiness_path = tmp_path / "readiness.json"
    source_deployment = tmp_path / "source_deployment.json"
    target_deployment = tmp_path / "target_deployment.json"
    cofitok_config = tmp_path / "cofitok.json"
    for path in (
        readiness_path,
        source_deployment,
        target_deployment,
        cofitok_config,
    ):
        path.write_text("{}", encoding="ascii")
    _patch_bridge_dependencies(
        monkeypatch,
        project=project,
        readiness_path=readiness_path,
        source_deployment=source_deployment,
        target_deployment=target_deployment,
        cofitok_config=cofitok_config,
    )
    monkeypatch.setattr(
        bridge,
        "_current_runtime_environment_sha",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("historical replay read the current runtime")
        ),
    )

    report = bridge.build_readiness_bridge(
        project_root=project,
        readiness_path=readiness_path,
        expected_readiness_sha256="1" * 64,
        source_deployment_receipt=source_deployment,
        expected_source_deployment_receipt_sha256="2" * 64,
        target_deployment_receipt=target_deployment,
        expected_target_deployment_receipt_sha256="3" * 64,
        expected_source_revision="f" * 40,
        expected_source_branch="scale/source",
        expected_target_revision="a" * 40,
        expected_target_branch="scale/target",
        require_current_target_git=False,
        require_current_runtime_environment=False,
    )

    assert report["target_runtime_environment_sha256"] == "e" * 64


def test_bridge_launch_rejects_current_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    readiness_path = tmp_path / "readiness.json"
    source_deployment = tmp_path / "source_deployment.json"
    target_deployment = tmp_path / "target_deployment.json"
    cofitok_config = tmp_path / "cofitok.json"
    for path in (
        readiness_path,
        source_deployment,
        target_deployment,
        cofitok_config,
    ):
        path.write_text("{}", encoding="ascii")
    _patch_bridge_dependencies(
        monkeypatch,
        project=project,
        readiness_path=readiness_path,
        source_deployment=source_deployment,
        target_deployment=target_deployment,
        cofitok_config=cofitok_config,
    )
    monkeypatch.setattr(
        bridge,
        "_current_runtime_environment_sha",
        lambda *args, **kwargs: "d" * 64,
    )

    with pytest.raises(ValueError, match="target runtime environment differs"):
        bridge.build_readiness_bridge(
            project_root=project,
            readiness_path=readiness_path,
            expected_readiness_sha256="1" * 64,
            source_deployment_receipt=source_deployment,
            expected_source_deployment_receipt_sha256="2" * 64,
            target_deployment_receipt=target_deployment,
            expected_target_deployment_receipt_sha256="3" * 64,
            expected_source_revision="f" * 40,
            expected_source_branch="scale/source",
            expected_target_revision="a" * 40,
            expected_target_branch="scale/target",
            require_current_target_git=False,
            require_current_runtime_environment=True,
        )

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

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
        "launch_receipt_schema_version": 2,
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

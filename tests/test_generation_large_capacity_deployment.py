from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256, write_json_report
from scripts.build_generation_large_capacity_deployment_receipt import (
    build_deployment_receipt,
    verify_deployment_receipt,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> dict:
    formal = tmp_path / "formal"
    formal.mkdir()
    _git(formal, "init")
    _git(formal, "config", "user.email", "tests@example.com")
    _git(formal, "config", "user.name", "CoFiTok Tests")
    _git(formal, "checkout", "-b", "scale/generative-system")
    (formal / "base.txt").write_text("base\n", encoding="ascii")
    _git(formal, "add", "base.txt")
    _git(formal, "commit", "-m", "base")
    base = _git(formal, "rev-parse", "HEAD")

    _git(formal, "checkout", "-b", "prerequisite-two")
    prerequisite_file = formal / "prerequisite.txt"
    prerequisite_file.write_text("prerequisite\n", encoding="ascii")
    _git(formal, "add", "prerequisite.txt")
    _git(formal, "commit", "-m", "prerequisite")
    second_prerequisite = _git(formal, "rev-parse", "HEAD")

    _git(formal, "checkout", "-b", "scale/generation-large-capacity", base)
    runbook = formal / "artifacts/runbooks/example.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="ascii")
    _git(formal, "add", "artifacts/runbooks/example.sh")
    _git(formal, "commit", "-m", "target")
    _git(formal, "merge", "--no-ff", "prerequisite-two", "-m", "merge prerequisites")
    target = _git(formal, "rev-parse", "HEAD")
    bundle = tmp_path / "large-capacity.bundle"
    _git(
        formal,
        "bundle",
        "create",
        str(bundle),
        "scale/generation-large-capacity",
        f"^{base}",
        f"^{second_prerequisite}",
    )
    _git(formal, "checkout", "scale/generative-system")

    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--no-local", str(formal), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(
        checkout,
        "fetch",
        str(bundle),
        "refs/heads/scale/generation-large-capacity:"
        "refs/heads/scale/generation-large-capacity",
    )
    _git(checkout, "checkout", "scale/generation-large-capacity")

    runbook_report = tmp_path / "runbook_syntax.json"
    write_json_report(
        runbook_report,
        {
            "schema_version": 1,
            "role": "generation_runbook_syntax_check",
            "status": "pass",
            "enumeration": "git_ls_files",
            "runbook_root": (checkout / "artifacts/runbooks").as_posix(),
            "discovered_count": 1,
            "checked_count": 1,
            "failed_count": 0,
            "runbooks": ["artifacts/runbooks/example.sh"],
            "failures": [],
            "git": {
                "revision": target,
                "branch": "scale/generation-large-capacity",
                "tracked_dirty": False,
            },
        },
    )
    pytest_report = tmp_path / "pytest.xml"
    pytest_report.write_text(
        '<testsuites><testsuite tests="814" failures="0" errors="0" '
        'skipped="2" /></testsuites>\n',
        encoding="ascii",
    )
    kwargs = {
        "formal_repository": formal,
        "checkout": checkout,
        "bundle": bundle,
        "runbook_syntax_path": runbook_report,
        "pytest_report_path": pytest_report,
        "expected_bundle_sha256": file_sha256(bundle),
        "expected_bundle_bytes": bundle.stat().st_size,
        "expected_formal_revision": base,
        "expected_formal_branch": "scale/generative-system",
        "expected_target_revision": target,
        "expected_target_branch": "scale/generation-large-capacity",
        "expected_prerequisites": [base, second_prerequisite],
        "minimum_pytest_passed": 800,
        "maximum_pytest_skipped": 5,
    }
    return {
        "formal": formal,
        "checkout": checkout,
        "bundle": bundle,
        "runbook_report": runbook_report,
        "pytest_report": pytest_report,
        "kwargs": kwargs,
    }


def test_large_capacity_deployment_receipt_replays_exact_sources(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = build_deployment_receipt(**fixture["kwargs"])
    receipt = tmp_path / "deployment_receipt.json"
    write_json_report(receipt, report)

    verified = verify_deployment_receipt(
        report,
        receipt_path=receipt,
        expected_receipt_sha256=file_sha256(receipt),
    )

    assert verified == report
    assert report["validation"]["pytest"]["passed"] == 812
    assert report["validation"]["runbook_syntax"]["checked_count"] == 1
    assert report["readiness_execution_allowed"] is True
    assert report["full_training_launch_allowed"] is False


def test_large_capacity_deployment_rejects_bundle_and_test_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    kwargs = fixture["kwargs"]
    report = build_deployment_receipt(**kwargs)

    fixture["pytest_report"].write_text(
        '<testsuite tests="814" failures="0" errors="0" skipped="20" />\n',
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="pytest evidence"):
        build_deployment_receipt(**kwargs)

    fixture["bundle"].write_bytes(fixture["bundle"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bundle verification|bundle identity"):
        build_deployment_receipt(**kwargs)

    assert report["formal_generation_completion_claimed"] is False


def test_large_capacity_deployment_thresholds_cannot_be_weakened(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    kwargs = fixture["kwargs"]

    weakened_pass = dict(kwargs, minimum_pytest_passed=799)
    with pytest.raises(ValueError, match="pass floor"):
        build_deployment_receipt(**weakened_pass)

    weakened_skip = dict(kwargs, maximum_pytest_skipped=6)
    with pytest.raises(ValueError, match="skip ceiling"):
        build_deployment_receipt(**weakened_skip)


def test_deployment_receipt_can_replay_after_formal_repository_advances(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = build_deployment_receipt(**fixture["kwargs"])
    receipt = tmp_path / "deployment_receipt.json"
    write_json_report(receipt, report)

    formal = fixture["formal"]
    (formal / "later.txt").write_text("later\n", encoding="ascii")
    _git(formal, "add", "later.txt")
    _git(formal, "commit", "-m", "later")

    with pytest.raises(ValueError, match="formal repository identity"):
        verify_deployment_receipt(
            report,
            receipt_path=receipt,
            expected_receipt_sha256=file_sha256(receipt),
        )
    assert (
        verify_deployment_receipt(
            report,
            receipt_path=receipt,
            expected_receipt_sha256=file_sha256(receipt),
            require_current_formal_repository=False,
        )
        == report
    )


def test_large_capacity_deployment_runbook_is_isolated_and_readiness_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_deploy_large_capacity_readiness_checkout.sh"
    ).read_text(encoding="utf-8")

    assert "git clone --no-local" in source
    assert "git -C \"$FORMAL_REPOSITORY\" bundle verify" in source
    assert "build_generation_large_capacity_deployment_receipt.py" in source
    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "--minimum-pytest-passed 800" in source
    assert "--maximum-pytest-skipped 5" in source
    pytest_index = source.index("-m pytest -q --junitxml")
    promote_index = source.index('mv "$TEMP_CHECKOUT" "$CHECKOUT"')
    assert pytest_index < promote_index
    assert "TEMP_RUNBOOK_SYNTAX" in source
    assert "TEMP_PYTEST_REPORT" in source
    final_syntax = source.rindex("scripts/check_generation_runbook_syntax.py")
    receipt = source.index("build_generation_large_capacity_deployment_receipt.py")
    assert promote_index < final_syntax < receipt
    assert 'for candidate in "$TEMP_CHECKOUT" "$CHECKOUT"' in source
    assert 'case "$resolved_candidate" in' in source
    assert "scripts/train_generation.py" not in source
    assert "generation_stability_ema_teacher_full_readiness_after_gate.sh" not in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source
    assert "git -C \"$FORMAL_REPOSITORY\" checkout" not in source

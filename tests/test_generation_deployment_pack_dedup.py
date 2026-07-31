from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cofitok import deployment_pack_dedup as dedup
from cofitok.reporting import file_sha256, write_json_report


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> dict:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "tests@example.com")
    _git(source, "config", "user.name", "CoFiTok Tests")
    _git(source, "checkout", "-b", "scale/generation-large-capacity")
    (source / "payload.txt").write_text("payload\n", encoding="ascii")
    _git(source, "add", "payload.txt")
    _git(source, "commit", "-m", "payload")
    _git(source, "gc", "--prune=now")
    revision = _git(source, "rev-parse", "HEAD")

    root = tmp_path / "deployment" / "large_capacity"
    root.mkdir(parents=True)
    canonical = root / "checkout-caa66eb"
    target = root / "checkout-08d4850"
    shutil.copytree(source, canonical)
    shutil.copytree(source, target)
    _git(target, "config", "user.email", "tests@example.com")
    _git(target, "config", "user.name", "CoFiTok Tests")
    (target / "target.txt").write_text("target\n", encoding="ascii")
    _git(target, "add", "target.txt")
    _git(target, "commit", "-m", "target revision")
    target_revision = _git(target, "rev-parse", "HEAD")
    canonical_receipt = root / "deployments" / revision / "deployment_receipt.json"
    target_receipt = root / "deployments" / target_revision / "deployment_receipt.json"
    for receipt, bound_revision in (
        (canonical_receipt, revision),
        (target_receipt, target_revision),
    ):
        receipt.parent.mkdir(parents=True)
        write_json_report(receipt, {"revision": bound_revision, "role": "test_receipt"})
    return {
        "root": root,
        "canonical": canonical,
        "target": target,
        "revision": revision,
        "canonical_receipt": canonical_receipt,
        "target_receipt": target_receipt,
    }


def _plan(fixture: dict, *, active: set[str] | None = None) -> dict:
    return dedup.build_deployment_pack_dedup_plan(
        deployment_root=fixture["root"],
        canonical_checkout=fixture["canonical"],
        active_checkout_paths=(
            {fixture["canonical"].resolve().as_posix()} if active is None else active
        ),
    )


def test_dedup_plan_excludes_canonical_and_identifies_duplicate_pack(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = _plan(fixture)
    eligible = [item for item in report["actions"] if item["disposition"] == "eligible"]
    canonical = [
        item
        for item in report["actions"]
        if item["checkout"] == fixture["canonical"].resolve().as_posix()
    ]

    assert len(eligible) == 2
    assert {item["kind"] for item in eligible} == {"pack", "index"}
    assert all(item["checkout"] == fixture["target"].resolve().as_posix() for item in eligible)
    assert all(item["disposition"] == "required" for item in canonical)
    assert report["summary"]["potential_physical_bytes_saved"] > 0
    assert dedup.verify_deployment_pack_dedup_plan(report) == report


def test_active_or_unbound_checkout_is_never_eligible(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target_path = fixture["target"].resolve().as_posix()
    active = _plan(
        fixture,
        active={fixture["canonical"].resolve().as_posix(), target_path},
    )
    assert all(
        item["disposition"] == "required"
        for item in active["actions"]
        if item["checkout"] == target_path
    )

    fixture["target_receipt"].unlink()
    unbound = _plan(fixture)
    assert all(
        item["disposition"] == "indeterminate"
        for item in unbound["actions"]
        if item["checkout"] == target_path
    )


def test_apply_hardlinks_pack_and_preserves_git_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    report = _plan(fixture)
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    write_json_report(plan_path, report)
    replayed = []
    monkeypatch.setattr(dedup, "_active_checkout_paths", lambda: set())
    monkeypatch.setattr(
        dedup,
        "_replay_receipt",
        lambda checkout, receipt: replayed.append(checkout.resolve().as_posix()),
    )

    result = dedup.apply_deployment_pack_dedup_plan(
        plan=report,
        plan_path=plan_path,
        expected_plan_sha256=file_sha256(plan_path),
        output_path=result_path,
        allow_checkouts={fixture["target"].as_posix()},
    )

    assert result["status"] == "pass"
    assert result["destructive_deletion_performed"] is False
    assert len(result["applied"]) == 2
    assert len(replayed) == 2
    for item in result["applied"]:
        assert os.path.samefile(item["source"]["path"], item["target"]["path"])
        assert item["source"]["sha256"] == item["target"]["sha256"]
    assert _git(fixture["target"], "rev-parse", "HEAD") == next(
        item["revision"]
        for item in report["checkouts"]
        if item["path"] == fixture["target"].resolve().as_posix()
    )
    assert _git(fixture["target"], "status", "--porcelain", "--untracked-files=no") == ""
    assert dedup.verify_deployment_pack_dedup_result(
        json.loads(result_path.read_text(encoding="utf-8")),
        result_path=result_path,
        expected_result_sha256=file_sha256(result_path),
    )["status"] == "pass"


def test_apply_rejects_allowlist_or_pack_drift_before_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    report = _plan(fixture)
    plan_path = tmp_path / "plan.json"
    write_json_report(plan_path, report)
    monkeypatch.setattr(dedup, "_active_checkout_paths", lambda: set())
    monkeypatch.setattr(dedup, "_replay_receipt", lambda checkout, receipt: None)

    with pytest.raises(ValueError, match="must equal"):
        dedup.apply_deployment_pack_dedup_plan(
            plan=report,
            plan_path=plan_path,
            expected_plan_sha256=file_sha256(plan_path),
            output_path=tmp_path / "result.json",
            allow_checkouts={fixture["canonical"].as_posix()},
        )

    target_pack = next((fixture["target"] / ".git/objects/pack").glob("*.pack"))
    if os.name == "nt":
        os.chmod(target_pack, 0o600)
    target_pack.write_bytes(target_pack.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="plan replay changed|checkout changed|pack file changed"):
        dedup.apply_deployment_pack_dedup_plan(
            plan=report,
            plan_path=plan_path,
            expected_plan_sha256=file_sha256(plan_path),
            output_path=tmp_path / "result.json",
            allow_checkouts={fixture["target"].as_posix()},
        )


def test_apply_rejects_checkout_that_becomes_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    report = _plan(fixture)
    plan_path = tmp_path / "plan.json"
    write_json_report(plan_path, report)
    monkeypatch.setattr(
        dedup,
        "_active_checkout_paths",
        lambda: {fixture["target"].resolve().as_posix()},
    )

    with pytest.raises(ValueError, match="became active"):
        dedup.apply_deployment_pack_dedup_plan(
            plan=report,
            plan_path=plan_path,
            expected_plan_sha256=file_sha256(plan_path),
            output_path=tmp_path / "result.json",
            allow_checkouts={fixture["target"].as_posix()},
        )


def test_replanning_after_partial_atomic_link_is_recoverable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = _plan(fixture)
    first = next(item for item in report["actions"] if item["disposition"] == "eligible")
    dedup._replace_with_hardlink(
        Path(first["source"]["path"]),
        Path(first["target"]["path"]),
    )

    recovered = _plan(fixture)
    target_actions = [
        item
        for item in recovered["actions"]
        if item["checkout"] == fixture["target"].resolve().as_posix()
        and item["source"] is not None
    ]

    assert {item["disposition"] for item in target_actions} == {
        "already_linked",
        "eligible",
    }
    assert recovered["summary"]["potential_physical_bytes_saved"] > 0


def test_active_process_scan_includes_cwd_and_other_command_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("synthetic /proc symlinks require Unix semantics")
    process_root = tmp_path / "proc"
    cwd_checkout = tmp_path / "deployment" / "checkout-1111111"
    cmd_checkout = tmp_path / "deployment" / "checkout-2222222"
    cwd_checkout.mkdir(parents=True)
    cmd_checkout.mkdir(parents=True)
    for pid, cwd, command in (
        (101, cwd_checkout, b"python worker.py\0"),
        (202, tmp_path, f"python --project {cmd_checkout}\0".encode()),
    ):
        proc = process_root / str(pid)
        proc.mkdir(parents=True)
        os.symlink(cwd, proc / "cwd", target_is_directory=True)
        (proc / "cmdline").write_bytes(command)
    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(dedup, "_process_ancestry", lambda root: {999})

    assert dedup._active_checkout_paths(process_root) == {
        cwd_checkout.as_posix(),
        cmd_checkout.as_posix(),
    }


def test_active_process_scan_ignores_invocation_ancestor_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("synthetic /proc symlinks require Unix semantics")
    process_root = tmp_path / "proc"
    checkout = tmp_path / "deployment" / "checkout-3333333"
    checkout.mkdir(parents=True)
    proc = process_root / "202"
    proc.mkdir(parents=True)
    os.symlink(tmp_path, proc / "cwd", target_is_directory=True)
    (proc / "cmdline").write_bytes(f"bash apply --allow {checkout}\0".encode())
    monkeypatch.setattr(dedup, "_process_ancestry", lambda root: {202, 999})

    assert dedup._active_checkout_paths(process_root) == set()

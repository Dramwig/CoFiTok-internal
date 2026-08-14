from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from cofitok.generation_control_continuity import (
    AUTHORIZATION_BOUNDARY,
    build_continuity_archive,
    git_state,
    load_continuity_plan,
    restore_continuity_archive,
    verify_continuity_archive,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PLAN = (
    ROOT
    / "configs/generation/diagnostics/"
    "capacity_generation_control_plane_continuity_v1.json"
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _identity(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _commit(repo: Path, branch: str, filename: str, payload: str) -> tuple[str, str]:
    _git(repo, "switch", "-C", branch, "base")
    (repo / filename).write_text(payload, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", branch)
    return _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")


def _fixture(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "base")

    checkout_rows = []
    branch_specs = [
        ("followup", "scale/test-followup", "followup.txt"),
        ("quality", "scale/test-quality", "quality.txt"),
        ("preparation", "scale/test-preparation", "preparation.txt"),
    ]
    revisions = {}
    for name, branch, filename in branch_specs:
        revision, tree = _commit(repo, branch, filename, f"{name}\n")
        revisions[name] = revision
        checkout_rows.append(
            {
                "name": name,
                "revision": revision,
                "tree": tree,
                "branch": branch,
                "restore_relative_path": f"{name}/checkout",
            }
        )

    _git(repo, "switch", "-C", "scale/test-builder", "base")
    (repo / "builder.txt").write_text("builder\n", encoding="utf-8")
    _git(repo, "add", "builder.txt")
    _git(repo, "commit", "-m", "builder")

    bundle_rows = []
    for index, (name, branch, _) in enumerate(branch_specs):
        path = tmp_path / f"{name}.bundle"
        _git(repo, "bundle", "create", str(path), f"refs/heads/{branch}")
        size, digest = _identity(path)
        bundle_rows.append(
            {
                "name": name,
                "source_path": path.resolve().as_posix(),
                "archive_relative_path": f"bundles/{name}.bundle",
                "restore_relative_path": f"{name}.bundle",
                "expected_bytes": size,
                "expected_sha256": digest,
                "expected_revision": revisions[name],
                "expected_ref": f"refs/heads/{branch}",
                "repository_seed": index == 0,
            }
        )

    runtime_rows = []
    for name, payload in (("approval", b"approved\n"), ("waiter", b"wait\n")):
        source = tmp_path / "runtime" / f"{name}.txt"
        _write(source, payload)
        size, digest = _identity(source)
        runtime_rows.append(
            {
                "name": name,
                "source_path": source.resolve().as_posix(),
                "archive_relative_path": f"runtime/{name}.txt",
                "restore_relative_path": f"quality/{name}.txt",
                "expected_bytes": size,
                "expected_sha256": digest,
            }
        )

    plan = {
        "schema_version": 1,
        "role": "generation_control_plane_continuity_plan",
        "authorization_boundary": AUTHORIZATION_BOUNDARY,
        "bundles": bundle_rows,
        "checkouts": checkout_rows,
        "runtime_files": runtime_rows,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    state = git_state(repo)
    return {"repo": repo, "plan": plan_path, "state": state}


def _build(tmp_path: Path) -> tuple[dict, dict]:
    fixture = _fixture(tmp_path)
    state = fixture["state"]
    result = build_continuity_archive(
        project=fixture["repo"],
        plan_path=fixture["plan"],
        output_root=tmp_path / "archive",
        expected_revision=state["revision"],
        expected_tree=state["tree"],
        expected_branch=state["branch"],
    )
    return fixture, result


def test_production_continuity_plan_binds_the_three_vulnerable_checkouts() -> None:
    plan = load_continuity_plan(PRODUCTION_PLAN)
    assert [row["name"] for row in plan["bundles"]] == [
        "quality_bridge_followup_decision",
        "quality_bridge_execution",
        "capacity_probe_preparation",
    ]
    assert plan["bundles"][0]["repository_seed"] is True
    assert sum(row["repository_seed"] for row in plan["bundles"]) == 1
    assert {row["name"] for row in plan["runtime_files"]} == {
        "quality_bridge_standing_authorization",
        "quality_bridge_execution_approval",
        "quality_bridge_recovery_supervisor",
        "quality_bridge_idle_waiter",
        "quality_bridge_followup_waiter",
    }
    assert plan["authorization_boundary"] == AUTHORIZATION_BOUNDARY


def test_build_verify_and_restore_continuity_archive(tmp_path: Path) -> None:
    fixture, result = _build(tmp_path)
    manifest = result["manifest"]
    assert result["verification"]["status"] == "pass"
    assert result["verification"]["verified_file_count"] == 5

    verification = verify_continuity_archive(
        archive_root=tmp_path / "archive",
        expected_manifest_sha256=manifest["sha256"],
    )
    assert verification["status"] == "pass"
    restore = restore_continuity_archive(
        archive_root=tmp_path / "archive",
        expected_manifest_sha256=manifest["sha256"],
        destination_root=tmp_path / "restored",
    )
    assert restore["status"] == "pass"
    assert restore["effects"] == {
        "processes_launched": False,
        "processes_signaled": False,
        "gpu_queried_or_allocated": False,
        "formal_checkout_modified": False,
        "experiment_execution_authorized": False,
    }
    assert {row["name"] for row in restore["checkouts"]} == {
        "followup",
        "quality",
        "preparation",
    }
    assert (tmp_path / "restored/quality/approval.txt").read_bytes() == b"approved\n"
    assert (tmp_path / "restored/quality/waiter.txt").read_bytes() == b"wait\n"
    assert git_state(fixture["repo"]) == fixture["state"]


def test_continuity_archive_rejects_payload_tampering(tmp_path: Path) -> None:
    _, result = _build(tmp_path)
    path = tmp_path / "archive/payload/runtime/approval.txt"
    path.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="identity differs"):
        verify_continuity_archive(
            archive_root=tmp_path / "archive",
            expected_manifest_sha256=result["manifest"]["sha256"],
        )


def test_continuity_restore_refuses_existing_bounded_target(tmp_path: Path) -> None:
    _, result = _build(tmp_path)
    destination = tmp_path / "restored"
    (destination / "quality").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="restore target exists"):
        restore_continuity_archive(
            archive_root=tmp_path / "archive",
            expected_manifest_sha256=result["manifest"]["sha256"],
            destination_root=destination,
        )


def test_continuity_code_has_no_experiment_or_process_control_entrypoint() -> None:
    paths = [
        ROOT / "src/cofitok/generation_control_continuity.py",
        ROOT / "scripts/build_generation_control_plane_continuity_archive.py",
        ROOT / "scripts/verify_generation_control_plane_continuity_archive.py",
        ROOT / "scripts/restore_generation_control_plane_continuity_archive.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "subprocess.Popen" not in source
    assert "os.kill" not in source
    assert "signal." not in source
    assert "nvidia-smi" not in source
    assert "train_generation.py" not in source
    assert "generate_samples.py" not in source

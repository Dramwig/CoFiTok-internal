from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.deploy_generation_postbridge_two_stage import (
    apply_deployment,
    build_manifest,
    file_identity,
    git_state,
    load_manifest,
    preflight_deployment,
    validate_manifest,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEPLOYER = ROOT / "scripts/deploy_generation_postbridge_two_stage.py"


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        raise AssertionError(
            f"git {' '.join(arguments)} failed: {completed.stderr or completed.stdout}"
        )
    return completed.stdout.strip()


def _git_object_exists(repository: Path, revision: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repository,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _commit(repository: Path, path: str, content: str, message: str) -> str:
    destination = repository / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    _git(repository, "add", path)
    _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _initialize_repository(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-b", "scale/generative-system")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "user.name", "CoFiTok Tests")


def _formal_repository(
    path: Path,
    *,
    source: Path,
    include_secondary_prerequisite: bool,
) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "user.name", "CoFiTok Tests")
    _git(
        path,
        "fetch",
        "--no-tags",
        str(source),
        "refs/heads/scale/generative-system",
    )
    _git(path, "checkout", "-b", "scale/generative-system", "FETCH_HEAD")
    if include_secondary_prerequisite:
        _git(
            path,
            "fetch",
            "--no-tags",
            str(source),
            "refs/heads/prerequisite-two:refs/heads/prerequisite-two",
        )
    return path


def _terminal_guard_sources(root: Path, *, active: bool = False) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    execution = {
        "status": "running" if active else "completed",
        "pid": 424242,
    }
    pair = {
        "status": "running" if active else "pass",
        "gpu_contention": {
            "current": {
                "training_process_pids": [424243] if active else [],
            }
        },
    }
    (root / "reports/execution_status.json").write_text(
        json.dumps(execution),
        encoding="utf-8",
    )
    (root / "pair_monitor.json").write_text(
        json.dumps(pair),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict:
    source = tmp_path / "source"
    _initialize_repository(source)
    root_revision = _commit(source, "base.txt", "base\n", "base")
    formal_revision = _commit(source, "formal.txt", "formal\n", "formal")
    formal_tree = _git(source, "rev-parse", "HEAD^{tree}")

    _git(source, "checkout", "-b", "prerequisite-two", root_revision)
    secondary_revision = _commit(
        source,
        "secondary.txt",
        "secondary\n",
        "secondary prerequisite",
    )

    _git(
        source,
        "checkout",
        "-b",
        "bootstrap/generation-training-base",
        formal_revision,
    )
    _git(source, "merge", "--no-ff", "prerequisite-two", "-m", "training merge")
    bootstrap_revision = _commit(
        source,
        "training.txt",
        "training\n",
        "active training base",
    )
    bootstrap_ref = "refs/heads/bootstrap/generation-training-base"

    integration_branch = "integration/generation-postbridge-hardening-v1"
    integration_ref = f"refs/heads/{integration_branch}"
    _git(source, "checkout", "-b", integration_branch)
    integration_revision = _commit(
        source,
        "src/postbridge.py",
        "POSTBRIDGE = True\n",
        "postbridge integration",
    )

    bootstrap_bundle = tmp_path / "bootstrap.bundle"
    _git(
        source,
        "bundle",
        "create",
        str(bootstrap_bundle),
        bootstrap_ref,
        f"^{formal_revision}",
        f"^{secondary_revision}",
    )
    integration_bundle = tmp_path / "integration.bundle"
    _git(
        source,
        "bundle",
        "create",
        str(integration_bundle),
        integration_ref,
        f"^{bootstrap_revision}",
    )

    formal = _formal_repository(
        tmp_path / "formal",
        source=source,
        include_secondary_prerequisite=True,
    )
    missing_prerequisite_formal = _formal_repository(
        tmp_path / "formal-missing-prerequisite",
        source=source,
        include_secondary_prerequisite=False,
    )
    quality_root = tmp_path / "quality-bridge"
    _terminal_guard_sources(quality_root)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    manifest = build_manifest(
        repository=source,
        formal_repository_path=formal.resolve().as_posix(),
        formal_revision=formal_revision,
        formal_branch="scale/generative-system",
        bootstrap_bundle=bootstrap_bundle,
        bootstrap_revision=bootstrap_revision,
        bootstrap_ref=bootstrap_ref,
        bootstrap_prerequisites=[formal_revision, secondary_revision],
        integration_bundle=integration_bundle,
        integration_revision=integration_revision,
        integration_ref=integration_ref,
        quality_bridge_root=quality_root.resolve().as_posix(),
    )
    assert manifest["formal_repository"]["tree"] == formal_tree
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    loaded, identity = load_manifest(manifest_path)
    return {
        "source": source,
        "formal": formal,
        "missing_prerequisite_formal": missing_prerequisite_formal,
        "formal_revision": formal_revision,
        "secondary_revision": secondary_revision,
        "bootstrap_revision": bootstrap_revision,
        "integration_revision": integration_revision,
        "bootstrap_ref": bootstrap_ref,
        "integration_ref": integration_ref,
        "bootstrap_bundle": bootstrap_bundle,
        "integration_bundle": integration_bundle,
        "quality_root": quality_root,
        "proc_root": proc_root,
        "manifest": loaded,
        "manifest_path": manifest_path,
        "manifest_identity": identity,
    }


def _preflight(fixture: dict) -> dict:
    return preflight_deployment(
        manifest=fixture["manifest"],
        manifest_identity=fixture["manifest_identity"],
        formal_repository=fixture["formal"],
        bootstrap_bundle=fixture["bootstrap_bundle"],
        integration_bundle=fixture["integration_bundle"],
        temporary_parent=fixture["formal"].parent / "preflight-temp",
    )


def test_preflight_detects_reclone_gap_and_keeps_formal_repository_immutable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    formal = fixture["formal"]
    before = git_state(formal)
    assert not _git_object_exists(formal, fixture["bootstrap_revision"])

    report = _preflight(fixture)

    assert report["mode"] == "preflight"
    assert report["status"] == "pass"
    assert report["formal_repository"]["unchanged"] is True
    assert report["formal_repository"]["fetch_performed"] is False
    assert report["bundle_chain"]["integration"]["bootstrap_required"] is True
    assert (
        report["bundle_chain"]["integration"][
            "direct_formal_verify_possible_before_bootstrap"
        ]
        is False
    )
    assert report["isolated_preflight"]["integration_head"] == fixture[
        "integration_revision"
    ]
    assert report["untracked_target_conflicts"]["status"] == "pass"
    assert git_state(formal) == before
    assert not _git_object_exists(formal, fixture["bootstrap_revision"])


def test_preflight_rejects_missing_bootstrap_prerequisite(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    missing = fixture["missing_prerequisite_formal"]
    manifest["formal_repository"]["path"] = missing.resolve().as_posix()

    with pytest.raises(ValueError, match="missing bootstrap prerequisites"):
        preflight_deployment(
            manifest=manifest,
            manifest_identity=fixture["manifest_identity"],
            formal_repository=missing,
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=fixture["integration_bundle"],
        )
    assert _git(missing, "rev-parse", "HEAD") == fixture["formal_revision"]


def test_manifest_and_runtime_reject_ref_and_sha_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        ValueError,
        match="builder Git identity differs|advertised head differs",
    ):
        build_manifest(
            repository=fixture["source"],
            formal_repository_path=fixture["formal"].resolve().as_posix(),
            formal_revision=fixture["formal_revision"],
            formal_branch="scale/generative-system",
            bootstrap_bundle=fixture["bootstrap_bundle"],
            bootstrap_revision=fixture["bootstrap_revision"],
            bootstrap_ref=fixture["bootstrap_ref"],
            bootstrap_prerequisites=[
                fixture["formal_revision"],
                fixture["secondary_revision"],
            ],
            integration_bundle=fixture["integration_bundle"],
            integration_revision=fixture["integration_revision"],
            integration_ref="refs/heads/wrong-integration-ref",
            quality_bridge_root=fixture["quality_root"].resolve().as_posix(),
        )

    tampered_manifest = deepcopy(fixture["manifest"])
    tampered_manifest["integration_bundle"]["advertised_heads"][0]["ref"] = (
        "refs/heads/wrong-integration-ref"
    )
    tampered_manifest["target"]["source_branch"] = "wrong-integration-ref"
    with pytest.raises(ValueError, match="repository or target identity is invalid"):
        validate_manifest(tampered_manifest)
    tampered_manifest["builder_git"]["branch"] = "wrong-integration-ref"
    validate_manifest(tampered_manifest)
    with pytest.raises(ValueError, match="runtime bundle Git contract differs"):
        preflight_deployment(
            manifest=tampered_manifest,
            manifest_identity=fixture["manifest_identity"],
            formal_repository=fixture["formal"],
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=fixture["integration_bundle"],
        )

    changed_bundle = tmp_path / "tampered" / fixture["integration_bundle"].name
    changed_bundle.parent.mkdir()
    shutil.copyfile(fixture["integration_bundle"], changed_bundle)
    with changed_bundle.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="runtime bundle identity differs"):
        preflight_deployment(
            manifest=fixture["manifest"],
            manifest_identity=fixture["manifest_identity"],
            formal_repository=fixture["formal"],
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=changed_bundle,
        )


def test_apply_is_blocked_while_bound_training_or_controller_is_active(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _terminal_guard_sources(fixture["quality_root"], active=True)
    before = _git(fixture["formal"], "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="status is non-terminal"):
        apply_deployment(
            manifest=fixture["manifest"],
            manifest_identity=fixture["manifest_identity"],
            formal_repository=fixture["formal"],
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=fixture["integration_bundle"],
            confirm_target_revision=fixture["integration_revision"],
            proc_root=fixture["proc_root"],
        )

    assert _git(fixture["formal"], "rev-parse", "HEAD") == before
    assert not _git_object_exists(
        fixture["formal"],
        fixture["bootstrap_revision"],
    )


def test_apply_is_blocked_when_terminal_status_still_has_live_controller(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    controller = fixture["proc_root"] / "424242"
    controller.mkdir()
    (controller / "cmdline").write_bytes(
        b"bash\0generation_stability_full_data_quality_bridge_100k_execute.sh\0"
    )

    with pytest.raises(ValueError, match="trainer/controller processes are active"):
        apply_deployment(
            manifest=fixture["manifest"],
            manifest_identity=fixture["manifest_identity"],
            formal_repository=fixture["formal"],
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=fixture["integration_bundle"],
            confirm_target_revision=fixture["integration_revision"],
            proc_root=fixture["proc_root"],
        )

    assert _git(fixture["formal"], "rev-parse", "HEAD") == fixture[
        "formal_revision"
    ]
    assert not _git_object_exists(
        fixture["formal"],
        fixture["bootstrap_revision"],
    )


def test_apply_requires_exact_target_and_rejects_untracked_target_conflict(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="target confirmation differs"):
        apply_deployment(
            manifest=fixture["manifest"],
            manifest_identity=fixture["manifest_identity"],
            formal_repository=fixture["formal"],
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=fixture["integration_bundle"],
            confirm_target_revision="0" * 40,
            proc_root=fixture["proc_root"],
        )

    conflict = fixture["formal"] / "src/postbridge.py"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("user-owned\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target-added untracked conflicts"):
        apply_deployment(
            manifest=fixture["manifest"],
            manifest_identity=fixture["manifest_identity"],
            formal_repository=fixture["formal"],
            bootstrap_bundle=fixture["bootstrap_bundle"],
            integration_bundle=fixture["integration_bundle"],
            confirm_target_revision=fixture["integration_revision"],
            proc_root=fixture["proc_root"],
        )
    assert _git(fixture["formal"], "rev-parse", "HEAD") == fixture[
        "formal_revision"
    ]
    assert conflict.read_text(encoding="utf-8") == "user-owned\n"


def test_preflight_rejects_ignored_untracked_target_conflict(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["formal"] / ".git/info/exclude").write_text(
        "src/postbridge.py\n",
        encoding="utf-8",
    )
    conflict = fixture["formal"] / "src/postbridge.py"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("ignored user-owned content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target-added untracked conflicts"):
        _preflight(fixture)

    assert conflict.read_text(encoding="utf-8") == "ignored user-owned content\n"
    assert _git(fixture["formal"], "rev-parse", "HEAD") == fixture[
        "formal_revision"
    ]


def test_explicit_apply_fast_forwards_only_after_terminal_guard(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unrelated = fixture["formal"] / "artifacts/history/user.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("{}\n", encoding="utf-8")

    report = apply_deployment(
        manifest=fixture["manifest"],
        manifest_identity=fixture["manifest_identity"],
        formal_repository=fixture["formal"],
        bootstrap_bundle=fixture["bootstrap_bundle"],
        integration_bundle=fixture["integration_bundle"],
        confirm_target_revision=fixture["integration_revision"],
        proc_root=fixture["proc_root"],
    )

    assert report["mode"] == "apply"
    assert report["active_training_guard"]["before_fetch"]["status"] == "pass"
    assert report["active_training_guard"]["before_fast_forward"]["status"] == "pass"
    assert report["formal_repository"]["fetch_performed"] is True
    assert report["formal_repository"]["fast_forward_performed"] is True
    assert _git(fixture["formal"], "rev-parse", "HEAD") == fixture[
        "integration_revision"
    ]
    assert _git(fixture["formal"], "branch", "--show-current") == (
        "scale/generative-system"
    )
    assert _git(
        fixture["formal"],
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    ) == ""
    assert unrelated.read_text(encoding="utf-8") == "{}\n"


def test_cli_defaults_to_preflight_and_apply_requires_manifest_sha(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    forbidden_output = fixture["formal"] / "preflight-report.json"
    forbidden = subprocess.run(
        [
            sys.executable,
            str(DEPLOYER),
            "--manifest",
            str(fixture["manifest_path"]),
            "--formal-repository",
            str(fixture["formal"]),
            "--bootstrap-bundle",
            str(fixture["bootstrap_bundle"]),
            "--integration-bundle",
            str(fixture["integration_bundle"]),
            "--output",
            str(forbidden_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert forbidden.returncode != 0
    assert "output cannot be inside formal repository" in forbidden.stderr
    assert not forbidden_output.exists()

    output = tmp_path / "reports/preflight.json"
    command = [
        sys.executable,
        str(DEPLOYER),
        "--manifest",
        str(fixture["manifest_path"]),
        "--formal-repository",
        str(fixture["formal"]),
        "--bootstrap-bundle",
        str(fixture["bootstrap_bundle"]),
        "--integration-bundle",
        str(fixture["integration_bundle"]),
        "--output",
        str(output),
        "--proc-root",
        str(fixture["proc_root"]),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["mode"] == "preflight"
    assert _git(fixture["formal"], "rev-parse", "HEAD") == fixture[
        "formal_revision"
    ]

    apply_output = tmp_path / "reports/apply.json"
    completed = subprocess.run(
        [
            *command[:-4],
            "--output",
            str(apply_output),
            "--mode",
            "apply",
            "--confirm-target-revision",
            fixture["integration_revision"],
            "--proc-root",
            str(fixture["proc_root"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "apply requires --expected-manifest-sha256" in completed.stderr
    assert not apply_output.exists()
    assert _git(fixture["formal"], "rev-parse", "HEAD") == fixture[
        "formal_revision"
    ]


def test_load_manifest_rejects_manifest_sha_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA256 differs"):
        load_manifest(
            fixture["manifest_path"],
            expected_sha256="f" * 64,
        )
    assert file_identity(fixture["manifest_path"])["sha256"] == fixture[
        "manifest_identity"
    ]["sha256"]

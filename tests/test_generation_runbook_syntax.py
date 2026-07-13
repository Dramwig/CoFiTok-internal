from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import check_generation_runbook_syntax as syntax


REVISION = "a" * 40


def _project(tmp_path: Path, names: tuple[str, ...]) -> Path:
    root = tmp_path / "project"
    runbooks = root / "artifacts" / "runbooks"
    runbooks.mkdir(parents=True)
    for name in names:
        path = runbooks / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\ntrue\n", encoding="ascii")
    return root


def _patch_git(monkeypatch) -> None:
    monkeypatch.setattr(
        syntax,
        "git_provenance",
        lambda _root: {
            "revision": REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
    )


def _patch_runbooks(monkeypatch, root: Path, names: tuple[str, ...]) -> None:
    monkeypatch.setattr(
        syntax,
        "tracked_runbooks",
        lambda _root: [root / "artifacts" / "runbooks" / name for name in names],
    )


def test_tracked_runbooks_excludes_untracked_shell_history(tmp_path) -> None:
    root = _project(tmp_path, ("tracked.sh", "historical.sh"))
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "artifacts/runbooks/tracked.sh"],
        cwd=root,
        check=True,
    )

    paths = syntax.tracked_runbooks(root)

    assert [path.relative_to(root).as_posix() for path in paths] == [
        "artifacts/runbooks/tracked.sh"
    ]


def test_runbook_checker_enumerates_every_shell_script(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path, ("a.sh", "nested/b.sh", "ignored.txt"))
    checked = []

    def fake_run(arguments, **_kwargs):
        checked.append(arguments[-1])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(syntax.subprocess, "run", fake_run)
    _patch_git(monkeypatch)
    _patch_runbooks(monkeypatch, root, ("a.sh", "nested/b.sh"))

    report = syntax.check_runbook_syntax(root)

    assert report["status"] == "pass"
    assert report["enumeration"] == "git_ls_files"
    assert report["discovered_count"] == 2
    assert report["checked_count"] == 2
    assert report["runbooks"] == [
        "artifacts/runbooks/a.sh",
        "artifacts/runbooks/nested/b.sh",
    ]
    assert len(checked) == 2


def test_runbook_checker_records_each_failure(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path, ("good.sh", "bad.sh"))

    def fake_run(arguments, **_kwargs):
        failed = arguments[-1].endswith("bad.sh")
        return SimpleNamespace(
            returncode=2 if failed else 0,
            stderr="syntax error" if failed else "",
        )

    monkeypatch.setattr(syntax.subprocess, "run", fake_run)
    _patch_git(monkeypatch)
    _patch_runbooks(monkeypatch, root, ("bad.sh", "good.sh"))

    report = syntax.check_runbook_syntax(root)

    assert report["status"] == "failed"
    assert report["failed_count"] == 1
    assert report["failures"] == [
        {
            "path": "artifacts/runbooks/bad.sh",
            "exit_code": 2,
            "stderr": "syntax error",
        }
    ]

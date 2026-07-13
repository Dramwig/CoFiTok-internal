from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_generation_deployment_conflicts.py"


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path, target_path: str) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "CoFiTok Tests")
    (repository / "tracked.txt").write_text("pinned\n", encoding="ascii")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "pinned")
    current = _git(repository, "rev-parse", "HEAD")

    target_file = repository / Path(target_path)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("target\n", encoding="ascii")
    _git(repository, "add", target_path)
    _git(repository, "commit", "-m", "target")
    target = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--detach", current)
    return repository, current, target


def _check(
    repository: Path,
    current: str,
    target: str,
    *,
    output: Path | None = None,
) -> subprocess.CompletedProcess:
    arguments = [
            sys.executable,
            str(CHECKER),
            "--repository",
            str(repository),
            "--current-commit",
            current,
            "--target-commit",
            target,
        ]
    if output is not None:
        arguments.extend(("--output", str(output)))
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )


def test_checker_ignores_unrelated_untracked_artifact_trees(tmp_path) -> None:
    repository, current, target = _repository(tmp_path, "src/new_module.py")
    noise = repository / "artifacts/history"
    noise.mkdir(parents=True)
    for index in range(300):
        (noise / f"report_{index:04d}.json").write_text("{}", encoding="ascii")

    result = _check(repository, current, target)
    report = json.loads(result.stdout)

    assert result.returncode == 0
    assert report["status"] == "pass"
    assert report["target_added_path_count"] == 1
    assert report["conflicts"] == []


def test_checker_atomically_persists_pass_report(tmp_path) -> None:
    repository, current, target = _repository(tmp_path, "src/new_module.py")
    output = tmp_path / "reports" / "conflict_scan.json"

    result = _check(repository, current, target, output=output)

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(result.stdout)


@pytest.mark.parametrize(
    ("target_path", "blocking_path"),
    (
        ("src/new_module.py", "src/new_module.py"),
        ("new_package/module.py", "new_package"),
    ),
)
def test_checker_rejects_exact_and_parent_path_conflicts(
    tmp_path,
    target_path: str,
    blocking_path: str,
) -> None:
    repository, current, target = _repository(tmp_path, target_path)
    blocker = repository / blocking_path
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("untracked blocker\n", encoding="ascii")

    output = tmp_path / "conflict_scan.json"
    result = _check(repository, current, target, output=output)
    report = json.loads(result.stdout)

    assert result.returncode == 76
    assert report["status"] == "conflict"
    assert report["conflicts"] == [blocking_path]
    assert json.loads(output.read_text(encoding="utf-8")) == report

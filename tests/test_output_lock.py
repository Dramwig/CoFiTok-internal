from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.output_lock import (
    exclusive_output_lock,
    exclusive_output_locks,
    output_lock_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _contender(target: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from cofitok.output_lock import exclusive_output_lock; "
                f"target={str(target)!r}; "
                "\nwith exclusive_output_lock(target, role='contender'):\n"
                " print('acquired')"
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_output_lock_excludes_another_process_and_records_owner(tmp_path: Path) -> None:
    target = tmp_path / "samples"
    lock_path = output_lock_path(target)

    with exclusive_output_lock(target, role="holder") as owner:
        contender = _contender(target)

        assert contender.returncode != 0
        assert "output target is already locked" in contender.stderr
        assert owner["role"] == "holder"
        assert owner["pid"] == os.getpid()
        assert owner["target"] == target.resolve().as_posix()

    persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    assert owner == persisted

    released = _contender(target)
    assert released.returncode == 0, released.stderr
    assert released.stdout.strip() == "acquired"


def test_output_lock_rejects_symlink_target(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        return

    try:
        with exclusive_output_lock(alias, role="must_not_acquire"):
            raise AssertionError("symlink target unexpectedly acquired")
    except ValueError as error:
        assert "must not contain a symlink" in str(error)


def test_output_locks_acquire_targets_in_stable_order(tmp_path: Path) -> None:
    first = tmp_path / "a_samples"
    second = tmp_path / "z_samples"

    with exclusive_output_locks(
        [second, first],
        role="matched_writer",
    ) as owners:
        assert [owner["target"] for owner in owners] == [
            first.resolve().as_posix(),
            second.resolve().as_posix(),
        ]


def test_output_locks_reject_duplicate_targets(tmp_path: Path) -> None:
    target = tmp_path / "samples"

    with pytest.raises(ValueError, match="duplicate output lock target"):
        with exclusive_output_locks([target, target], role="matched_writer"):
            raise AssertionError("duplicate output targets unexpectedly acquired")

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

import scripts.run_generation_exposure_capacity_continuation as controller


def test_claimed_output_root_is_created_inside_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock(_path: Path):
        events.append("lock_enter")
        yield
        events.append("lock_exit")

    monkeypatch.setattr(controller, "_execution_lock", fake_lock)
    output_root = tmp_path / "candidate"
    with controller._claimed_output_root(output_root, tmp_path / "candidate.lock"):
        assert output_root.is_dir()
        assert events == ["lock_enter"]
    assert events == ["lock_enter", "lock_exit"]


@pytest.mark.skipif(controller.fcntl is None, reason="POSIX flock is required")
def test_execution_lock_rejects_a_competing_lock(tmp_path: Path) -> None:
    lock = tmp_path / "candidate.lock"
    with controller._execution_lock(lock):
        with pytest.raises(RuntimeError, match="another bounded continuation"):
            with controller._execution_lock(lock):
                pass

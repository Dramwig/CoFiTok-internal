from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cofitok.inference_replay import read_json_object


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix symlink creation requires elevated privileges on Windows",
)
def test_read_json_object_rejects_symlink_file_and_parent(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    file_alias = tmp_path / "file-alias.json"
    file_alias.symlink_to(target)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        read_json_object(file_alias, name="security probe")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "nested.json"
    nested.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        read_json_object(parent_alias / "nested.json", name="security probe")


def test_read_json_object_keeps_regular_json_behavior(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = {"status": "completed", "count": 2}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_json_object(path, name="report") == payload

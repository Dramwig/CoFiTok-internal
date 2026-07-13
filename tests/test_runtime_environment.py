from __future__ import annotations

import copy

import torch

from cofitok.environment import (
    capture_runtime_environment,
    runtime_environment_mismatch_paths,
    runtime_environment_sha256,
)


def test_runtime_environment_fingerprint_is_canonical() -> None:
    left = {"schema_version": 1, "nested": {"b": 2, "a": [1, 3]}}
    right = {"nested": {"a": [1, 3], "b": 2}, "schema_version": 1}

    assert runtime_environment_sha256(left) == runtime_environment_sha256(right)
    assert len(runtime_environment_sha256(left)) == 64


def test_runtime_environment_mismatch_reports_nested_path() -> None:
    expected = {"torch": {"version": "2.7.1", "capability": [12, 0]}}
    actual = copy.deepcopy(expected)
    actual["torch"]["capability"][1] = 1

    assert runtime_environment_mismatch_paths(expected, actual) == [
        "runtime_environment.torch.capability[1]"
    ]


def test_capture_runtime_environment_binds_project_files(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    environment = capture_runtime_environment(
        torch.device("cpu"),
        project_root=tmp_path,
    )

    assert environment["schema_version"] == 1
    assert environment["device"] == {"type": "cpu"}
    assert environment["project_files"]["pyproject.toml"]["bytes"] > 0
    assert len(environment["project_files"]["uv.lock"]["sha256"]) == 64
    assert environment["packages"]["torch"] is not None

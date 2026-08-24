from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.build_generation_epsilon_stability_sampling_result as result_builder
from cofitok.generation import (
    EPSILON_STABILITY_OBSERVATION_MANIFEST_SCHEMA,
    EPSILON_STABILITY_REAL_ARTIFACT_REFERENCE_SCHEMA,
)
from cofitok.generation_gate_sources import gate_source_report_identity


REPORT_KEYS = (
    "sampling_report",
    "metrics_report",
    "class_fidelity_report",
    "artifact_report",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    design_path = tmp_path / "design.json"
    execution_path = tmp_path / "execution_authorization.json"
    observation_path = tmp_path / "observation.json"
    real_artifact_report_path = tmp_path / "real_artifact_report.json"
    subset_manifest_path = tmp_path / "real_subset_manifest.json"
    report_paths = {key: tmp_path / f"{key}.json" for key in REPORT_KEYS}

    _write_json(design_path, {"kind": "design"})
    _write_json(execution_path, {"kind": "execution"})
    _write_json(observation_path, {"kind": "observation"})
    _write_json(real_artifact_report_path, {"kind": "real_artifact_report"})
    _write_json(subset_manifest_path, {"kind": "real_subset_manifest"})
    for key, path in report_paths.items():
        _write_json(path, {"kind": key})

    observation_identity = gate_source_report_identity(observation_path)
    report_identities = {
        key: gate_source_report_identity(path)
        for key, path in report_paths.items()
    }
    manifest_path = tmp_path / "observation_manifest.json"
    manifest = {
        "schema": EPSILON_STABILITY_OBSERVATION_MANIFEST_SCHEMA,
        "status": "pass",
        "observations": [
            {
                "case_id": "legacy_terminal_hard_clip",
                "method": "cofitok",
                "observation": observation_identity,
                "source_reports": report_identities,
            }
        ],
    }
    _write_json(manifest_path, manifest)

    real_reference_path = tmp_path / "real_artifact_reference.json"
    real_reference = {
        "schema": EPSILON_STABILITY_REAL_ARTIFACT_REFERENCE_SCHEMA,
        "status": "pass",
        "source_reports": {
            "artifact_report": gate_source_report_identity(
                real_artifact_report_path
            ),
            "subset_manifest": gate_source_report_identity(
                subset_manifest_path
            ),
        },
    }
    _write_json(real_reference_path, real_reference)

    return {
        "design_path": design_path,
        "execution_path": execution_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "real_reference_path": real_reference_path,
        "real_reference": real_reference,
        "observation_path": observation_path,
        "report_paths": report_paths,
        "real_artifact_report_path": real_artifact_report_path,
    }


def _build_kwargs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "design_path": fixture["design_path"],
        "expected_design_sha256": gate_source_report_identity(
            fixture["design_path"]
        )["sha256"],
        "execution_authorization_path": fixture["execution_path"],
        "expected_execution_authorization_sha256": gate_source_report_identity(
            fixture["execution_path"]
        )["sha256"],
        "observation_manifest_path": fixture["manifest_path"],
        "expected_observation_manifest_sha256": gate_source_report_identity(
            fixture["manifest_path"]
        )["sha256"],
        "real_artifact_reference_path": fixture["real_reference_path"],
        "expected_real_artifact_reference_sha256": gate_source_report_identity(
            fixture["real_reference_path"]
        )["sha256"],
    }


def test_result_builder_replays_manifest_reports_and_real_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(
        result_builder,
        "build_epsilon_stability_sampling_result",
        fake_build,
    )

    assert result_builder.build_from_paths(**_build_kwargs(fixture)) == {
        "status": "captured"
    }
    observation = captured["observation_sources"][0]
    assert observation["identity"] == gate_source_report_identity(
        fixture["observation_path"]
    )
    assert observation["payload"] == {"kind": "observation"}
    assert set(observation["reports"]) == set(REPORT_KEYS)
    for key in REPORT_KEYS:
        assert observation["reports"][key] == {
            "identity": gate_source_report_identity(
                fixture["report_paths"][key]
            ),
            "payload": {"kind": key},
        }
    assert captured["observation_manifest"] == {
        "identity": gate_source_report_identity(fixture["manifest_path"]),
        "payload": fixture["manifest"],
    }
    assert captured["real_artifact_reference"] == {
        "identity": gate_source_report_identity(
            fixture["real_reference_path"]
        ),
        "payload": fixture["real_reference"],
        "artifact_report": {
            "identity": gate_source_report_identity(
                fixture["real_artifact_report_path"]
            ),
            "payload": {"kind": "real_artifact_report"},
        },
    }


def test_result_builder_fails_closed_when_bound_report_changes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture["report_paths"]["sampling_report"],
        {"kind": "sampling_report", "changed": True},
    )

    with pytest.raises(ValueError, match="sampling_report changed after binding"):
        result_builder.build_from_paths(**_build_kwargs(fixture))

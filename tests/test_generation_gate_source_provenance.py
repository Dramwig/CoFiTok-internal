from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.generation_gate_sources import (
    GATE_SOURCE_SUFFIXES,
    build_generation_gate_source_reports,
    verify_generation_gate_source_reports,
)


ROOT = Path(__file__).resolve().parents[1]


def _sources(tmp_path: Path, stage: str = "scaling") -> tuple[dict, dict]:
    paths = {}
    for index, (name, suffix) in enumerate(GATE_SOURCE_SUFFIXES[stage].items()):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name, "index": index}), encoding="utf-8")
        paths[name] = path
    reports = build_generation_gate_source_reports(stage=stage, paths=paths)
    return paths, {"stage": stage, "source_reports": reports}


def test_generation_gate_source_reports_rehash_every_bound_file(tmp_path: Path) -> None:
    paths, gate = _sources(tmp_path)

    verified = verify_generation_gate_source_reports(gate)

    assert verified["status"] == "verified"
    assert set(verified["source_reports"]) == set(GATE_SOURCE_SUFFIXES["scaling"])

    paths["cofitok_generation"].write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="changed after binding"):
        verify_generation_gate_source_reports(gate)


def test_generation_gate_source_reports_reject_wrong_authoritative_path(
    tmp_path: Path,
) -> None:
    paths, _ = _sources(tmp_path)
    wrong = tmp_path / "wrong" / "training_report.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("{}\n", encoding="ascii")
    paths["cofitok_training"] = wrong

    with pytest.raises(ValueError, match="identity is invalid"):
        build_generation_gate_source_reports(stage="scaling", paths=paths)


def test_stability_scaling_source_profile_is_independent_from_gate_stage(
    tmp_path: Path,
) -> None:
    profile_root = (
        Path("\\\\?\\" + str(tmp_path.resolve()))
        if os.name == "nt"
        else tmp_path
    )
    paths = {}
    for index, (name, suffix) in enumerate(
        GATE_SOURCE_SUFFIXES["stability_scaling"].items()
    ):
        path = profile_root / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name, "index": index}), encoding="utf-8")
        paths[name] = path
    gate = {
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "source_reports": build_generation_gate_source_reports(
            stage="scaling",
            profile="stability_scaling",
            paths=paths,
        ),
    }

    verified = verify_generation_gate_source_reports(gate)

    assert verified["stage"] == "scaling"
    assert verified["source_profile"] == "stability_scaling"


def test_stability_full_source_profile_is_independent_from_legacy_full(
    tmp_path: Path,
) -> None:
    paths = {}
    for index, (name, suffix) in enumerate(
        GATE_SOURCE_SUFFIXES["stability_full"].items()
    ):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name, "index": index}), encoding="utf-8")
        paths[name] = path
    gate = {
        "stage": "full",
        "source_profile": "stability_full",
        "source_reports": build_generation_gate_source_reports(
            stage="full",
            profile="stability_full",
            paths=paths,
        ),
    }

    verified = verify_generation_gate_source_reports(gate)

    assert verified["stage"] == "full"
    assert verified["source_profile"] == "stability_full"


def test_generation_gate_source_profile_rejects_cross_stage_binding(
    tmp_path: Path,
) -> None:
    paths, _ = _sources(tmp_path, stage="full")

    with pytest.raises(ValueError, match="incompatible"):
        build_generation_gate_source_reports(
            stage="scaling",
            profile="full",
            paths=paths,
        )


def test_sources_only_cli_accepts_quality_hold_but_rejects_source_drift(
    tmp_path: Path,
) -> None:
    paths, gate = _sources(tmp_path)
    gate.update(schema_version=1, status="fail", decision="hold")
    gate_path = tmp_path / "promotion_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_generation_gate_report.py"),
        "--gate",
        str(gate_path),
        "--stage",
        "scaling",
        "--sources-only",
    ]

    passed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0, passed.stderr

    paths["dense_checkpoint_eval"].write_text("changed\n", encoding="ascii")
    failed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "changed after binding" in failed.stderr

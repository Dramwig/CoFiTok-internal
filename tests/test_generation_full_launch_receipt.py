from __future__ import annotations

import json
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256, write_json_report
from scripts import build_generation_full_launch_receipt as launch


REVISION = "a" * 40
BRANCH = "scale/generation-large-capacity"
READINESS_SHA = "b" * 64
SUPPLEMENTAL_SHA = "d" * 64


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in (
        "deployment_receipt",
        "promotion_gate",
        "stability_supplemental",
        "full_readiness",
        "readiness_bridge",
        "cofitok_config",
        "dense_config",
        "config_validation",
        "storage_capacity",
        "runtime_selection",
        "launch_storage_capacity",
    ):
        path = tmp_path / f"{name}.json"
        payload = {"name": name}
        if name == "readiness_bridge":
            payload = {
                "source_git": {
                    "revision": "f" * 40,
                    "branch": BRANCH,
                    "tracked_dirty": False,
                },
                "target_git": {
                    "revision": REVISION,
                    "branch": BRANCH,
                    "tracked_dirty": False,
                },
                "source_deployment_receipt": {
                    "path": (tmp_path / "source_deployment.json").as_posix(),
                    "sha256": "1" * 64,
                },
            }
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    return paths


def _bridge() -> dict:
    return {
        "source_git": {
            "revision": "f" * 40,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "target_git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "promotion_authorization": {"gate_sha256": "c" * 64},
        "training_semantics_identical": True,
        "full_training_runbook": {
            "controlled_preamble_upgrade": True,
            "source_preamble_sha256": "2" * 64,
            "normalized_target_preamble_sha256": "2" * 64,
            "training_execution_sha256": "3" * 64,
        },
        "runtime_selection": {
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 64,
            "runtime_environment_sha256": "e" * 64,
        },
    }


def _kwargs(tmp_path: Path, source_paths: dict[str, Path]) -> dict:
    return {
        "source_paths": source_paths,
        "training_run_dirs": [tmp_path / "cofitok", tmp_path / "dense"],
        "benchmark_root": tmp_path / "benchmarks",
        "storage_path": tmp_path,
        "project_root": tmp_path,
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
        "expected_readiness_sha256": READINESS_SHA,
        "expected_stability_supplemental_sha256": SUPPLEMENTAL_SHA,
        "require_current_runtime_environment": True,
        "require_current_git": True,
        "require_training_state_absent": True,
    }


def _patch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    source_paths: dict[str, Path],
) -> None:
    monkeypatch.setattr(
        launch,
        "file_sha256",
        lambda path: (
            READINESS_SHA
            if Path(path) == source_paths["full_readiness"]
            else file_sha256(Path(path))
        ),
    )
    monkeypatch.setattr(launch, "verify_readiness_bridge", lambda *a, **k: _bridge())
    monkeypatch.setattr(
        launch,
        "verify_frozen_supplemental_report",
        lambda *a, **k: {
            "report": launch.source_identities(
                {
                    "stability_supplemental": source_paths[
                        "stability_supplemental"
                    ]
                }
            )["stability_supplemental"],
            "checks": {"all_supplemental_quality_checks_passed": True},
            "supplemental_non_authorizing": True,
            "required_for_full_training_launch": True,
            "full_training_launch_allowed": False,
        },
    )
    monkeypatch.setattr(
        launch,
        "validate_full_storage_capacity",
        lambda *a, **k: (
            {
                "required_free_bytes": 100,
                "free_bytes": 200,
                "headroom_bytes": 100,
                "sample_count": k["minimum_sample_count"],
            }
        ),
    )


def test_full_launch_receipt_binds_sources_runtime_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_paths(tmp_path)
    _patch_dependencies(monkeypatch, sources)

    report = launch.build_full_launch_receipt(**_kwargs(tmp_path, sources))

    assert report["status"] == "pass"
    assert report["role"] == "stability_full_training_launch_receipt"
    assert report["readiness_sha256"] == READINESS_SHA
    assert report["runtime_selection"]["effective_batch_size"] == 64
    assert report["schema_version"] == 3
    supplemental = report["quality_prerequisites"][
        "frozen_stability_supplemental"
    ]
    assert supplemental["required_for_full_training_launch"] is True
    assert supplemental["full_training_launch_allowed"] is False
    assert report["readiness_bridge"]["training_semantics_identical"] is True
    assert report["readiness_bridge"]["controlled_preamble_upgrade"] is True
    assert report["readiness_bridge"]["source_preamble_sha256"] == report[
        "readiness_bridge"
    ]["normalized_target_preamble_sha256"]
    assert len(report["readiness_bridge"]["training_execution_sha256"]) == 64
    assert report["launch_storage_capacity"]["sample_count"] == 116_640
    assert report["training_state_absent_at_launch"] is True
    assert report["full_training_launch_authorized"] is True
    assert set(report["source_reports"]) == set(sources)


def test_full_launch_receipt_refuses_existing_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_paths(tmp_path)
    _patch_dependencies(monkeypatch, sources)
    kwargs = _kwargs(tmp_path, sources)
    run_dir = kwargs["training_run_dirs"][0]
    run_dir.mkdir()
    (run_dir / "latest.json").write_text("{}", encoding="ascii")

    with pytest.raises(ValueError, match="requires absent training state"):
        launch.build_full_launch_receipt(**kwargs)


def test_full_launch_receipt_replay_detects_receipt_and_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_paths(tmp_path)
    _patch_dependencies(monkeypatch, sources)
    kwargs = _kwargs(tmp_path, sources)
    report = launch.build_full_launch_receipt(**kwargs)
    receipt = tmp_path / "full_training_launch_receipt.json"
    write_json_report(receipt, report)
    expected_receipt_sha = file_sha256(receipt)

    verified = launch.verify_full_launch_receipt(
        report,
        receipt_path=receipt,
        expected_receipt_sha256=expected_receipt_sha,
        **{**kwargs, "require_training_state_absent": False},
    )
    assert verified == report

    receipt.write_text('{"status":"replaced"}\n', encoding="ascii")
    with pytest.raises(ValueError, match="launch receipt SHA256 differs"):
        launch.verify_full_launch_receipt(
            {"status": "replaced"},
            receipt_path=receipt,
            expected_receipt_sha256=expected_receipt_sha,
            **{**kwargs, "require_training_state_absent": False},
        )


def test_full_launch_receipt_requires_exact_source_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_paths(tmp_path)
    _patch_dependencies(monkeypatch, sources)
    del sources["deployment_receipt"]

    with pytest.raises(ValueError, match="source set differs"):
        launch.build_full_launch_receipt(**_kwargs(tmp_path, sources))


def test_full_launch_receipt_rejects_failed_stability_supplemental(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_paths(tmp_path)
    _patch_dependencies(monkeypatch, sources)
    monkeypatch.setattr(
        launch,
        "verify_frozen_supplemental_report",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("frozen supplemental quality prerequisite did not pass")
        ),
    )

    with pytest.raises(ValueError, match="quality prerequisite did not pass"):
        launch.build_full_launch_receipt(**_kwargs(tmp_path, sources))

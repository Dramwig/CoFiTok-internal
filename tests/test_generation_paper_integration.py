from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_generation_paper_integration as builder
from scripts import validate_generation_paper_integration as validator


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/paper-integration-test",
    "tracked_dirty": False,
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _matched(method: str, *, fid: float, prefix: int) -> dict:
    return {
        "method": method,
        "comparison_tier": "matched_training_direct",
        "directly_comparable_to_cofitok": True,
        "dataset": "imagenet_256",
        "resolution": 256,
        "training_steps": 300_000,
        "parameter_count": 62_837_576 if prefix == 8 else 62_824_707,
        "effective_batch_size": 64,
        "training_images_seen": 19_200_000,
        "sample_count": 50_000,
        "weights": "ema",
        "sampler": "ddim",
        "sample_steps": 250,
        "fid": fid,
        "inception_score": 21.25 if prefix == 8 else 20.75,
        "precision": 0.61 if prefix == 8 else 0.60,
        "recall": 0.57 if prefix == 8 else 0.56,
        "class_top1_accuracy": 0.23 if prefix == 8 else 0.22,
        "class_top5_accuracy": 0.47 if prefix == 8 else 0.46,
    }


def _official(alias: str, method: str, fid: float) -> dict:
    return {
        "alias": alias,
        "method": method,
        "comparison_tier": "official_pretrained_contextual",
        "directly_comparable_to_cofitok": False,
        "paper_table_role": "secondary related-method only",
        "source_status": "completed_eval_only_50k",
        "sample_count": 50_000,
        "fid": fid,
        "inception_score": 30.0,
        "precision": 0.70,
        "recall": 0.65,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stability: bool = False) -> dict:
    gate_path = tmp_path / "final_generation_gate.json"
    gate_source_path = tmp_path / "gate_source.json"
    _write(gate_source_path, {"status": "completed"})
    gate_sources = {"source": builder.file_identity(gate_source_path)}
    gate = {
        "schema_version": 5,
        "stage": "full",
        "status": "pass",
        "decision": "large_scale_generation_ready",
        "source_profile": "stability_full" if stability else "full",
        "source_reports": gate_sources,
    }
    _write(gate_path, gate)

    source_names = [
        "cofitok_training",
        "dense_training",
        "cofitok_generation",
        "dense_generation",
        "training_contention",
    ]
    if stability:
        source_names.append("class_fidelity_qualification")
    source_reports = {}
    for name in source_names:
        path = tmp_path / "sources" / f"{name}.json"
        _write(path, {"name": name})
        source_reports[name] = builder.file_identity(path)
    source_reports["final_gate"] = builder.file_identity(gate_path)

    official_path = tmp_path / "official_related_methods_table.json"
    _write(official_path, {"schema_version": 1, "rows": []})
    cofitok = _matched("CoFiTok K=8", fid=10.0, prefix=8)
    dense = _matched("Dense identity", fid=10.5, prefix=1)
    comparison = {
        "schema_version": builder.COMPARISON_REPORT_SCHEMA_VERSION,
        "status": "ready",
        "source_profile": "stability_full" if stability else "full",
        "final_gate": {
            "status": "pass",
            "decision": "large_scale_generation_ready",
        },
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "external_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
        },
        "source_reports": source_reports,
        "official_context_source": {
            "path": official_path.resolve().as_posix(),
            "sha256": builder.file_identity(official_path)["sha256"],
            "schema_version": 1,
        },
        "matched_training_rows": [cofitok, dense],
        "official_context_rows": [
            _official("d_ar", "D-AR", 2.63),
            _official("mar", "MAR", 2.34),
            _official("retok", "ReTok", 2.22),
        ],
        "matched_summary": {
            "cofitok_minus_dense_fid": -0.5,
            "cofitok_relative_fid": 10.0 / 10.5 - 1.0,
        },
    }
    comparison_path = tmp_path / "large_scale_generation_comparison.json"
    _write(comparison_path, comparison)

    profile = (
        "stability_generation_system_v1"
        if stability
        else "large_scale_generation_v1"
    )
    contract = builder._PROFILE_CONTRACTS[profile]
    comparison_evidence = {
        "matched_methods": list(builder.MATCHED_METHODS),
        "official_context_methods": ["d_ar", "mar", "retok"],
        "official_context_source_sha256": comparison["official_context_source"]["sha256"],
        "source_report_sha256": {
            name: identity["sha256"] for name, identity in source_reports.items()
        },
        "cross_tier_numeric_ranking_allowed": False,
        "class_fidelity": None,
    }
    gate_verification = {
        "status": "verified",
        "stage": "full",
        "source_profile": comparison["source_profile"],
        "source_reports": gate_sources,
    }
    gate_evidence = (
        {
            "decision": "large_scale_generation_ready",
            "gate_sha256": builder.file_identity(gate_path)["sha256"],
            "source_report_sha256": {
                name: identity["sha256"] for name, identity in gate_sources.items()
            },
        }
        if stability
        else {
            "decision": "large_scale_generation_ready",
            "source_reports": gate_sources,
        }
    )
    audit = {
        "schema_version": 1,
        "status": contract["audit_status"],
        "complete": True,
        "failed_checks": [],
        "missing_checks": [],
        "checks": [
            {
                "name": contract["final_gate_check"],
                "status": "pass",
                "evidence": gate_evidence,
            },
            {
                "name": contract["comparison_check"],
                "status": "pass",
                "evidence": comparison_evidence,
            },
            {
                "name": contract["release_check"],
                "status": "pass",
                "evidence": {"cofitok": {}, "dense_identity": {}},
            },
        ],
        "warnings": [],
    }
    if stability:
        audit["profile"] = profile
        audit["expectations"] = {"full_training_revision": "c" * 40}
    else:
        audit["expected_revisions"] = {"full_training": "c" * 40}
    audit_path = tmp_path / "completion_audit.json"
    _write(audit_path, audit)
    audit_identity = builder.file_identity(audit_path)

    artifact_paths = {
        "cofitok": tmp_path / "cofitok_ema.pt",
        "dense_identity": tmp_path / "dense_ema.pt",
    }
    release = {
        "schema_version": 1,
        "receipt_type": "cofitok_generation_release_receipt",
        "status": "completed",
        "completion_profile": profile,
        "completion_audit": audit_identity,
        "completion_expectations": audit.get("expectations", audit.get("expected_revisions")),
        "artifacts": {
            method: {"path": path.resolve().as_posix()}
            for method, path in artifact_paths.items()
        },
    }
    release_path = tmp_path / "generation_release_receipt.json"
    _write(release_path, release)

    monkeypatch.setattr(
        builder,
        "verify_comparison_source_reports",
        lambda report: {
            "status": "verified",
            "source_profile": report["source_profile"],
            "source_reports": report["source_reports"],
        },
    )
    monkeypatch.setattr(builder, "build_comparison_report", lambda **kwargs: comparison)
    monkeypatch.setattr(
        builder, "verify_generation_gate_source_reports", lambda report: gate_verification
    )
    monkeypatch.setattr(
        builder,
        "validate_generation_gate_authorization",
        lambda report, expected_stage: {"decision": "large_scale_generation_ready"},
    )

    def release_verifier(receipt: str | Path, artifact: str | Path) -> dict:
        method = next(
            name
            for name, path in artifact_paths.items()
            if Path(artifact).resolve() == path.resolve()
        )
        return {
            "method": method,
            "completion_profile": profile,
            "completion_audit": audit_identity,
        }

    return {
        "audit": audit,
        "audit_path": audit_path,
        "comparison": comparison,
        "comparison_path": comparison_path,
        "gate_path": gate_path,
        "release_path": release_path,
        "release_verifier": release_verifier,
        "output_dir": tmp_path / "paper_bundle",
    }


@pytest.mark.parametrize("stability", [False, True])
def test_build_and_replay_terminal_paper_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stability: bool
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, stability=stability)
    kwargs = {
        "completion_audit_path": fixture["audit_path"],
        "comparison_path": fixture["comparison_path"],
        "final_gate_path": fixture["gate_path"],
        "release_receipt_path": fixture["release_path"],
        "output_dir": fixture["output_dir"],
        "implementation_git": GIT,
        "release_verifier": fixture["release_verifier"],
    }
    first = builder.write_paper_integration_bundle(**kwargs)
    paths = [Path(row["path"]) for row in first["outputs"].values()]
    paths.append(fixture["output_dir"] / "generation_paper_integration_manifest.json")
    mtimes = {path: path.stat().st_mtime_ns for path in paths}
    second = builder.write_paper_integration_bundle(**kwargs)

    assert first == second
    assert all(path.stat().st_mtime_ns == mtimes[path] for path in paths)
    assert first["authorization_boundary"]["training_authorized"] is False
    assert first["paper_policy"]["cross_tier_numeric_ranking_allowed"] is False
    assert first["paper_policy"]["paper_consumer_files_mutated_by_builder"] is False
    assert first["paper_policy"]["generated_snippets_may_be_written_under_output_root"] is True
    assert "paper_source_mutated_by_builder" not in first["paper_policy"]
    assert "do not establish broad generation SOTA" in (
        fixture["output_dir"] / "generation_claims.tex"
    ).read_text(encoding="utf-8")
    assert "not numerically ranked" in (
        fixture["output_dir"] / "generation_official_context_table.tex"
    ).read_text(encoding="utf-8")


def test_validation_rebuilds_sources_and_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = builder.write_paper_integration_bundle(
        completion_audit_path=fixture["audit_path"],
        comparison_path=fixture["comparison_path"],
        final_gate_path=fixture["gate_path"],
        release_receipt_path=fixture["release_path"],
        output_dir=fixture["output_dir"],
        implementation_git=GIT,
        release_verifier=fixture["release_verifier"],
    )
    manifest_path = fixture["output_dir"] / "generation_paper_integration_manifest.json"
    receipt_path = fixture["output_dir"] / "generation_paper_integration.validation.json"
    receipt = validator.write_validation_receipt(
        manifest_path,
        receipt_path,
        implementation_git=GIT,
        release_verifier=fixture["release_verifier"],
    )
    mtime = receipt_path.stat().st_mtime_ns
    replay = validator.write_validation_receipt(
        manifest_path,
        receipt_path,
        implementation_git=GIT,
        release_verifier=fixture["release_verifier"],
    )

    assert replay == receipt
    assert receipt_path.stat().st_mtime_ns == mtime
    assert receipt["paper_application_ready"] is True
    assert receipt["manifest"] == builder.file_identity(manifest_path)
    assert receipt["outputs"] == manifest["outputs"]


def test_builder_rejects_incomplete_terminal_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = fixture["audit"]
    audit["complete"] = False
    audit["status"] = "in_progress"
    audit["missing_checks"] = ["formal_50k_generation"]
    _write(fixture["audit_path"], audit)

    with pytest.raises(ValueError, match="completion audit did not pass"):
        builder.build_paper_integration_manifest(
            completion_audit_path=fixture["audit_path"],
            comparison_path=fixture["comparison_path"],
            final_gate_path=fixture["gate_path"],
            release_receipt_path=fixture["release_path"],
            output_dir=fixture["output_dir"],
            implementation_git=GIT,
            release_verifier=fixture["release_verifier"],
        )


def test_builder_rejects_unsafe_cross_tier_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    comparison = fixture["comparison"]
    comparison["comparison_policy"]["cross_tier_numeric_ranking_allowed"] = True
    _write(fixture["comparison_path"], comparison)

    with pytest.raises(ValueError, match="tier policy is unsafe"):
        builder.build_paper_integration_manifest(
            completion_audit_path=fixture["audit_path"],
            comparison_path=fixture["comparison_path"],
            final_gate_path=fixture["gate_path"],
            release_receipt_path=fixture["release_path"],
            output_dir=fixture["output_dir"],
            implementation_git=GIT,
            release_verifier=fixture["release_verifier"],
        )


def test_validation_rejects_modified_latex_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    builder.write_paper_integration_bundle(
        completion_audit_path=fixture["audit_path"],
        comparison_path=fixture["comparison_path"],
        final_gate_path=fixture["gate_path"],
        release_receipt_path=fixture["release_path"],
        output_dir=fixture["output_dir"],
        implementation_git=GIT,
        release_verifier=fixture["release_verifier"],
    )
    output = fixture["output_dir"] / "generation_claims.tex"
    output.write_text(output.read_text(encoding="utf-8") + "% tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output changed: claims"):
        validator.validate_paper_integration_manifest(
            fixture["output_dir"] / "generation_paper_integration_manifest.json",
            implementation_git=GIT,
            release_verifier=fixture["release_verifier"],
        )


def test_builder_requires_clean_source_bound_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    dirty = {**GIT, "tracked_dirty": True}
    with pytest.raises(ValueError, match="clean named Git checkout"):
        builder.build_paper_integration_manifest(
            completion_audit_path=fixture["audit_path"],
            comparison_path=fixture["comparison_path"],
            final_gate_path=fixture["gate_path"],
            release_receipt_path=fixture["release_path"],
            output_dir=fixture["output_dir"],
            implementation_git=dirty,
            release_verifier=fixture["release_verifier"],
        )


@pytest.mark.parametrize(
    "script",
    [
        "build_generation_paper_integration.py",
        "validate_generation_paper_integration.py",
    ],
)
def test_paper_integration_cli_help(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

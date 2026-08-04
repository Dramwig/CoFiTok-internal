from __future__ import annotations

import shlex
import sys
from pathlib import Path

from cofitok.generation_paths import (
    SCALING_COFITOK_RUN_ID,
    SCALING_DENSE_RUN_ID,
    SCALING_REPORT_ID,
    generation_deployment_attestation_paths,
    generation_large_capacity_deployment_paths,
    generation_stability_workspace_paths,
    generation_workspace_paths,
)
from scripts import print_generation_workspace_paths


ROOT = Path(__file__).resolve().parents[1]


def test_scaling_workspace_uses_fresh_fixed_basis_v3_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = tmp_path / "outputs"
    paths = generation_workspace_paths(project_root=project, output_root=output)

    assert paths["SCALING_COFITOK_RUN"].name == SCALING_COFITOK_RUN_ID
    assert paths["SCALING_DENSE_RUN"].name == SCALING_DENSE_RUN_ID
    assert paths["SCALING_REPORT_ROOT"].name == SCALING_REPORT_ID
    assert paths["SCALING_GATE"] == paths["SCALING_REPORT_ROOT"] / "promotion_gate.json"
    assert "fixed_basis" in paths["SCALING_COFITOK_RUN"].name
    assert paths["SCALING_COFITOK_RUN"].name.endswith("_v3")
    assert "compressed_cofitok_k8_50k" not in paths["SCALING_COFITOK_RUN"].as_posix()


def test_shell_export_round_trips_paths_with_spaces(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = tmp_path / "project with spaces"
    output = tmp_path / "outputs with spaces"
    revision = "b" * 40
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "print_generation_workspace_paths.py",
            "--project-root",
            str(project),
            "--output-root",
            str(output),
            "--format",
            "shell",
            "--target-revision",
            revision,
        ],
    )

    print_generation_workspace_paths.main()

    assignments = {}
    for line in capsys.readouterr().out.splitlines():
        name, separator, raw = line.partition("=")
        assert separator
        assignments[name] = shlex.split(raw)[0]
    expected = {
        name: path.as_posix()
        for name, path in generation_workspace_paths(
            project_root=project,
            output_root=output,
        ).items()
    }
    expected.update(
        {
            name: path.as_posix()
            for name, path in generation_deployment_attestation_paths(
                output_root=output,
                target_revision=revision,
            ).items()
        }
    )
    assert assignments == expected


def test_deployment_attestation_paths_are_target_versioned(tmp_path: Path) -> None:
    revision = "a" * 40
    paths = generation_deployment_attestation_paths(
        output_root=tmp_path / "outputs",
        target_revision=revision,
    )

    assert paths["DEPLOYMENT_BUNDLE"].name == (
        f"cofitok-generation-upgrade-{revision}.bundle"
    )
    for name in (
        "DEPLOYMENT_RECEIPT",
        "DEPLOYMENT_CONFLICT_SCAN",
        "DEPLOYMENT_RUNBOOK_SYNTAX",
        "DEPLOYMENT_PYTEST",
    ):
        assert revision in paths[name].name
        assert paths[name].parent == paths["DEPLOYMENT_EVIDENCE_ROOT"]



def test_large_capacity_deployment_paths_preserve_prior_revisions(
    tmp_path: Path,
) -> None:
    first_revision = "a" * 40
    second_revision = "b" * 40
    first = generation_large_capacity_deployment_paths(
        output_root=tmp_path / "generation",
        target_revision=first_revision,
    )
    second = generation_large_capacity_deployment_paths(
        output_root=tmp_path / "generation",
        target_revision=second_revision,
    )

    assert first["STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT"] == second[
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT"
    ]
    assert first["STABILITY_LARGE_CAPACITY_DEPLOYMENT_EVIDENCE_ROOT"].name == (
        first_revision
    )
    assert second["STABILITY_LARGE_CAPACITY_DEPLOYMENT_EVIDENCE_ROOT"].name == (
        second_revision
    )
    assert first["STABILITY_LARGE_CAPACITY_DEPLOYMENT_RECEIPT"] != second[
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_RECEIPT"
    ]
    assert first["STABILITY_LARGE_CAPACITY_DEPLOYMENT_CHECKOUT"].name == (
        "checkout-aaaaaaa"
    )
    assert second["STABILITY_LARGE_CAPACITY_DEPLOYMENT_CHECKOUT"].name == (
        "checkout-bbbbbbb"
    )

def test_stability_workspace_paths_keep_every_stage_isolated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generation"
    paths = generation_stability_workspace_paths(output_root=output)

    assert paths["STABILITY_DECISION"].name == "scaling_decision.json"
    assert (
        paths["STABILITY_SCALING_GATE"]
        == paths["STABILITY_SCALING_REPORT_ROOT"] / "promotion_gate.json"
    )
    assert (
        paths["STABILITY_FULL_GATE"]
        == paths["STABILITY_FULL_REPORT_ROOT"] / "final_generation_gate.json"
    )
    assert (
        paths["STABILITY_QUALITY_BRIDGE_PREPARATION"]
        == paths["STABILITY_QUALITY_BRIDGE_REPORT_ROOT"] / "preparation.json"
    )
    assert (
        paths["STABILITY_QUALITY_BRIDGE_MONITOR"]
        == paths["STABILITY_QUALITY_BRIDGE_ROOT"] / "pair_monitor.json"
    )
    assert (
        paths["STABILITY_COFITOK_INFERENCE_ARTIFACT"].parent
        == paths["STABILITY_EXPORT_ROOT"]
    )
    assert (
        paths["STABILITY_LARGE_CAPACITY_DEPLOYMENT_RECEIPT"]
        == paths["STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT"]
        / "deployment_receipt.json"
    )
    assert (
        paths["STABILITY_SCALING_ROOT"]
        != paths["STABILITY_QUALITY_BRIDGE_ROOT"]
        != paths["STABILITY_FULL_ROOT"]
        != paths["STABILITY_EXPORT_ROOT"]
    )


def test_deployment_attestation_rejects_abbreviated_revision(tmp_path: Path) -> None:
    try:
        generation_deployment_attestation_paths(
            output_root=tmp_path,
            target_revision="abc123",
        )
    except ValueError as error:
        assert "full Git SHA-1" in str(error)
    else:
        raise AssertionError("abbreviated deployment revision was accepted")


def test_active_pipeline_runbooks_use_the_path_contract() -> None:
    for name in (
        "generation_10pct_matched_50k_2026-07-12.sh",
        "generation_10pct_posteval_2026-07-12.sh",
        "generation_complete_pipeline_after_10pct.sh",
        "generation_full_matched_300k_after_gate.sh",
        "generation_full_posteval_50k.sh",
        "generation_export_inference_artifacts.sh",
    ):
        source = (ROOT / "artifacts/runbooks" / name).read_text(encoding="utf-8")
        assert "scripts/print_generation_workspace_paths.py" in source
        assert '"$OUTPUT_ROOT/imagenet256_10pct_compressed_cofitok_k8_50k' not in source
        assert '"$OUTPUT_ROOT/imagenet256_10pct_compressed_dense_50k' not in source
        assert (
            "reports/generation/imagenet256_10pct_compressed_matched_50k"
            not in source
        )

    training = (
        ROOT / "artifacts/runbooks/generation_10pct_matched_50k_2026-07-12.sh"
    ).read_text(encoding="utf-8")
    assert "imagenet256_10pct_fixed_basis_cofitok_k8_50k.json" in training
    assert "imagenet256_10pct_fixed_basis_dense_50k.json" in training
    assert "rankcomplete_v2" not in training

    posteval = (
        ROOT / "artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh"
    ).read_text(encoding="utf-8")
    assert "fixed_basis_v3_gate10k_sampling" in posteval
    assert "rankcomplete_v2" not in posteval


def test_deployment_attestation_runbook_uses_versioned_evidence() -> None:
    source = (
        ROOT / "artifacts/runbooks/generation_attest_deployed_revision.sh"
    ).read_text(encoding="utf-8")

    assert "--target-revision \"$TARGET_REVISION\"" in source
    assert '[[ -e "$DEPLOYMENT_RECEIPT" ]]' in source
    assert "write_generation_deployment_receipt.py" in source
    assert "generation_upgrade_deployment_receipt.json" not in source

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from cofitok.generation.capacity_completion_result import (
    build_capacity_completion_100k_result,
    validate_capacity_completion_100k_result,
)
from cofitok.generation.capacity_completion_training import (
    validate_capacity_completion_training,
)
from cofitok.generation_class_fidelity import validate_class_fidelity_report
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from scripts.archive_generation_capacity_completion_sources import (
    build_source_archive,
)
from scripts.build_generation_milestone_report import (
    validate_milestone_report,
    verify_milestone_source_reports,
)
from scripts.build_generation_quality_bridge_result import (
    _verify_physical_generation_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain")),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-completion-execution-revision", required=True)
    parser.add_argument("--expected-completion-execution-tree", required=True)
    parser.add_argument("--expected-completion-execution-branch", required=True)
    parser.add_argument("--expected-completion-decision-revision", required=True)
    parser.add_argument("--expected-completion-decision-tree", required=True)
    parser.add_argument("--expected-completion-decision-branch", required=True)
    parser.add_argument("--expected-scaling-execution-revision", required=True)
    parser.add_argument("--expected-scaling-execution-tree", required=True)
    parser.add_argument("--expected-scaling-execution-branch", required=True)
    parser.add_argument("--expected-scaling-result-revision", required=True)
    parser.add_argument("--expected-scaling-result-tree", required=True)
    parser.add_argument("--expected-scaling-result-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)


def source_paths(output_root: Path) -> dict[str, Path]:
    root = output_root.resolve()
    reports = root / "reports"
    completion = reports / "capacity_completion_100k"
    cofitok_terminal = root / "base256_cofitok/terminal_100k"
    dense_terminal = root / "base256_dense_identity/terminal_100k"
    cofitok_samples = cofitok_terminal / "samples_10000_ddim100_cfg15"
    dense_samples = dense_terminal / "samples_10000_ddim100_cfg15"
    paths = {
        "capacity_probe_preparation": reports / "preparation.json",
        "capacity_probe_result": reports / "capacity_probe_result.json",
        "capacity_scaling_50k_result": reports / "capacity_scaling_50k_result.json",
        "capacity_completion_decision": reports
        / "capacity_completion_100k_decision.json",
        "capacity_completion_launch_receipt": completion / "launch_receipt.json",
        "capacity_completion_execution_status": completion / "execution_status.json",
        "source_checkpoint_archive": completion
        / "source_checkpoint_archive.json",
        "cofitok_training_validation": completion / "training/cofitok.json",
        "dense_identity_training_validation": completion
        / "training/dense_identity.json",
        "milestone_50000": reports
        / "capacity_scaling_50k/milestone_step_00050000.json",
        "milestone_100000": completion / "milestone_step_00100000.json",
        "cofitok_sampling_preflight": cofitok_terminal
        / "sampling_preflight.json",
        "dense_sampling_preflight": dense_terminal / "sampling_preflight.json",
        "cofitok_generation": cofitok_samples
        / "metrics/generation_metrics_report.json",
        "dense_generation": dense_samples
        / "metrics/generation_metrics_report.json",
        "cofitok_checkpoint_eval": cofitok_terminal
        / "checkpoint_eval/checkpoint_evaluation_report.json",
        "dense_checkpoint_eval": dense_terminal
        / "checkpoint_eval/checkpoint_evaluation_report.json",
        "class_fidelity_qualification": completion
        / "class_fidelity/qualification_report.json",
        "cofitok_class_fidelity": cofitok_samples
        / "class_fidelity/class_fidelity_report.json",
        "dense_identity_class_fidelity": dense_samples
        / "class_fidelity/class_fidelity_report.json",
    }
    preparation_path = paths["capacity_probe_preparation"]
    if preparation_path.is_file():
        preparation = read_json_object(
            preparation_path,
            name="capacity probe preparation",
        )
        source = preparation.get("source_evidence", {}).get(
            "quality_bridge_preparation"
        )
        if isinstance(source, dict) and isinstance(source.get("path"), str):
            paths["quality_bridge_preparation"] = Path(source["path"]).resolve()
    return paths


def _verify_bound_sources(report: dict[str, Any], *, label: str) -> None:
    sources = report.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError(f"{label} lacks bound source identities")
    for name, expected in sources.items():
        if not isinstance(expected, dict):
            raise ValueError(f"{label} source is malformed: {name}")
        if file_identity(expected.get("path", "")) != expected:
            raise ValueError(f"{label} source changed: {name}")


def _milestone_evidence(path: Path, *, step: int) -> dict[str, Any]:
    report = read_json_object(path, name=f"capacity completion milestone {step}")
    source_verification = verify_milestone_source_reports(
        report,
        source_profile="capacity_scaling",
    )
    evidence, warnings = validate_milestone_report(
        report,
        expected_step=step,
        source_verification=source_verification,
        expected_source_profile="capacity_scaling",
    )
    return {
        "status": "verified",
        "report": file_identity(path),
        "evidence": evidence,
        "warnings": warnings,
    }


def build_from_sources(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    result_git = _git_identity(PROJECT_ROOT)
    if result_git != {
        "revision": args.expected_result_revision,
        "tree": args.expected_result_tree,
        "branch": args.expected_result_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion result checkout identity differs")
    paths = source_paths(output_root)
    missing_names = sorted(
        name for name, path in paths.items() if not path.is_file()
    )
    if "quality_bridge_preparation" not in paths:
        missing_names.append("quality_bridge_preparation")
    if missing_names:
        raise FileNotFoundError(
            "capacity completion result sources are missing: "
            + ", ".join(sorted(set(missing_names)))
        )
    identities = {name: file_identity(path) for name, path in paths.items()}

    launch_receipt = read_json_object(
        paths["capacity_completion_launch_receipt"],
        name="capacity completion launch receipt",
    )
    _verify_bound_sources(launch_receipt, label="capacity completion launch receipt")
    decision = read_json_object(
        paths["capacity_completion_decision"],
        name="capacity completion decision",
    )
    archive = read_json_object(
        paths["source_checkpoint_archive"],
        name="capacity completion source checkpoint archive",
    )
    expected_archive = build_source_archive(
        decision_path=paths["capacity_completion_decision"],
        expected_decision_sha256=identities["capacity_completion_decision"][
            "sha256"
        ],
        expected_decision_revision=args.expected_completion_decision_revision,
        expected_decision_tree=args.expected_completion_decision_tree,
        expected_decision_branch=args.expected_completion_decision_branch,
        output_root=output_root,
        create_missing=False,
    )
    if archive != expected_archive:
        raise ValueError("capacity completion source archive is not reproducible")

    runtime = launch_receipt.get("selection", {}).get("runtime_selection", {})
    micro_batch = int(runtime.get("micro_batch_size", -1))
    accumulation = int(runtime.get("gradient_accumulation_steps", -1))
    training_validations = {}
    for method, run_name, parameters in (
        ("cofitok", "base256_cofitok", 250_153_763),
        ("dense_identity", "base256_dense_identity", 250_135_043),
    ):
        config_key = "cofitok_config" if method == "cofitok" else "dense_config"
        config_path = Path(
            launch_receipt["source_reports"][config_key]["path"]
        ).resolve()
        validation = validate_capacity_completion_training(
            report_path=output_root / run_name / "training_report.json",
            config_path=config_path,
            source_archive_path=paths["source_checkpoint_archive"],
            expected_source_archive_sha256=identities[
                "source_checkpoint_archive"
            ]["sha256"],
            method=method,
            expected_revision=args.expected_training_revision,
            expected_branch=args.expected_training_branch,
            expected_parameter_count=parameters,
            expected_micro_batch_size=micro_batch,
            expected_gradient_accumulation_steps=accumulation,
        )
        saved = read_json_object(
            paths[f"{method}_training_validation"],
            name=f"capacity completion {method} training validation",
        )
        if saved != validation:
            raise ValueError(
                f"capacity completion {method} training validation is not reproducible"
            )
        training_validations[method] = saved

    generation = {
        method: read_json_object(
            paths[f"{method}_generation"],
            name=f"capacity completion {method} terminal generation metrics",
        )
        for method in ("cofitok", "dense_identity")
    }
    physical = {
        method: _verify_physical_generation_evidence(
            generation[method],
            label=(
                "CoFiTok capacity completion terminal"
                if method == "cofitok"
                else "dense capacity completion terminal"
            ),
        )
        for method in ("cofitok", "dense_identity")
    }
    class_reports = {
        method: read_json_object(
            paths[f"{method}_class_fidelity"],
            name=f"capacity completion {method} class fidelity",
        )
        for method in ("cofitok", "dense_identity")
    }
    for method, report in class_reports.items():
        validate_class_fidelity_report(report)
        if report.get("sample_provenance") != generation[method].get(
            "sample_provenance"
        ):
            raise ValueError(
                f"capacity completion {method} class fidelity uses other samples"
            )
    qualification = read_json_object(
        paths["class_fidelity_qualification"],
        name="capacity completion class-fidelity qualification",
    )
    expected_class_sources = {
        "cofitok": identities["cofitok_class_fidelity"],
        "dense_identity": identities["dense_identity_class_fidelity"],
    }
    if qualification.get("sources") != expected_class_sources:
        raise ValueError("capacity completion class-fidelity sources differ")

    milestones = {
        step: read_json_object(
            paths[f"milestone_{step}"],
            name=f"capacity completion milestone {step}",
        )
        for step in (50_000, 100_000)
    }
    milestone_verifications = {
        step: _milestone_evidence(paths[f"milestone_{step}"], step=step)
        for step in (50_000, 100_000)
    }
    execution_git = {
        "revision": args.expected_completion_execution_revision,
        "tree": args.expected_completion_execution_tree,
        "branch": args.expected_completion_execution_branch,
        "tracked_dirty": False,
    }
    training_git = {
        "revision": args.expected_training_revision,
        "tree": args.expected_training_tree,
        "branch": args.expected_training_branch,
        "tracked_dirty": False,
    }
    result = build_capacity_completion_100k_result(
        quality_bridge_preparation=read_json_object(
            paths["quality_bridge_preparation"],
            name="quality bridge preparation",
        ),
        capacity_probe_preparation=read_json_object(
            paths["capacity_probe_preparation"],
            name="capacity probe preparation",
        ),
        capacity_probe_result=read_json_object(
            paths["capacity_probe_result"],
            name="capacity probe result",
        ),
        capacity_scaling_result=read_json_object(
            paths["capacity_scaling_50k_result"],
            name="capacity scaling 50K result",
        ),
        capacity_completion_decision=decision,
        capacity_completion_launch_receipt=launch_receipt,
        capacity_completion_execution_status=read_json_object(
            paths["capacity_completion_execution_status"],
            name="capacity completion execution status",
        ),
        source_checkpoint_archive=archive,
        training_validations=training_validations,
        milestone_reports=milestones,
        milestone_verifications=milestone_verifications,
        sampling_preflights={
            method: read_json_object(
                paths[f"{method}_sampling_preflight"],
                name=f"capacity completion {method} sampling preflight",
            )
            for method in ("cofitok", "dense_identity")
        },
        generation_reports=generation,
        checkpoint_evaluations={
            method: read_json_object(
                paths[f"{method}_checkpoint_eval"],
                name=f"capacity completion {method} checkpoint evaluation",
            )
            for method in ("cofitok", "dense_identity")
        },
        class_fidelity_qualification=qualification,
        physical_evidence=physical,
        source_identities=identities,
        completion_execution_git=execution_git,
        training_git=training_git,
        result_builder_git=result_git,
        expected_completion_decision_revision=(
            args.expected_completion_decision_revision
        ),
        expected_completion_decision_tree=args.expected_completion_decision_tree,
        expected_completion_decision_branch=args.expected_completion_decision_branch,
        expected_scaling_execution_revision=args.expected_scaling_execution_revision,
        expected_scaling_execution_tree=args.expected_scaling_execution_tree,
        expected_scaling_execution_branch=args.expected_scaling_execution_branch,
        expected_scaling_result_revision=args.expected_scaling_result_revision,
        expected_scaling_result_tree=args.expected_scaling_result_tree,
        expected_scaling_result_branch=args.expected_scaling_result_branch,
    )
    validate_capacity_completion_100k_result(
        result,
        expected_execution_revision=args.expected_completion_execution_revision,
        expected_execution_tree=args.expected_completion_execution_tree,
        expected_execution_branch=args.expected_completion_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay the matched 250M step-100K terminal evidence and "
            "build a non-authorizing capacity-completion result."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"capacity completion 100K result already exists: {args.output}"
        )
    report = build_from_sources(args)
    write_json_report(args.output, report)
    print(report["decision_support"]["recommended_next_stage"]["id"])


if __name__ == "__main__":
    main()

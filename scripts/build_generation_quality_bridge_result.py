from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.quality_bridge import build_quality_bridge_result
from cofitok.generation_class_fidelity import validate_class_fidelity_report
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.image_integrity import image_tree_sha256
from cofitok.reporting import write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)
from scripts.build_generation_milestone_report import (
    validate_milestone_report,
    verify_milestone_source_reports,
)
from scripts.evaluate_generation_metrics import (
    find_images,
    validate_sampling_provenance,
)
from scripts.validate_generation_training_pair import validate_training_pair


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument("--cofitok-training", type=Path, required=True)
    parser.add_argument("--dense-training", type=Path, required=True)
    parser.add_argument("--training-pair-validation", type=Path, required=True)
    parser.add_argument("--cofitok-training-audit", type=Path, required=True)
    parser.add_argument("--dense-training-audit", type=Path, required=True)
    parser.add_argument("--milestone-50000", type=Path, required=True)
    parser.add_argument("--milestone-100000", type=Path, required=True)
    parser.add_argument("--cofitok-sampling-preflight", type=Path, required=True)
    parser.add_argument("--dense-sampling-preflight", type=Path, required=True)
    parser.add_argument("--cofitok-generation", type=Path, required=True)
    parser.add_argument("--dense-generation", type=Path, required=True)
    parser.add_argument("--cofitok-checkpoint-eval", type=Path, required=True)
    parser.add_argument("--dense-checkpoint-eval", type=Path, required=True)
    parser.add_argument("--class-fidelity-qualification", type=Path, required=True)
    parser.add_argument("--cofitok-class-fidelity", type=Path, required=True)
    parser.add_argument("--dense-class-fidelity", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "preparation": args.preparation.resolve(),
        "launch_receipt": args.launch_receipt.resolve(),
        "cofitok_training": args.cofitok_training.resolve(),
        "dense_training": args.dense_training.resolve(),
        "training_pair_validation": args.training_pair_validation.resolve(),
        "cofitok_training_audit": args.cofitok_training_audit.resolve(),
        "dense_training_audit": args.dense_training_audit.resolve(),
        "milestone_50000": args.milestone_50000.resolve(),
        "milestone_100000": args.milestone_100000.resolve(),
        "cofitok_sampling_preflight": args.cofitok_sampling_preflight.resolve(),
        "dense_sampling_preflight": args.dense_sampling_preflight.resolve(),
        "cofitok_generation": args.cofitok_generation.resolve(),
        "dense_generation": args.dense_generation.resolve(),
        "cofitok_checkpoint_eval": args.cofitok_checkpoint_eval.resolve(),
        "dense_checkpoint_eval": args.dense_checkpoint_eval.resolve(),
        "class_fidelity_qualification": (
            args.class_fidelity_qualification.resolve()
        ),
        "cofitok_class_fidelity": args.cofitok_class_fidelity.resolve(),
        "dense_class_fidelity": args.dense_class_fidelity.resolve(),
    }


def _verify_bound_sources(report: dict[str, Any]) -> None:
    sources = report.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError("quality bridge launch receipt lacks bound sources")
    for name, expected in sources.items():
        if not isinstance(expected, dict):
            raise ValueError(f"quality bridge launch source is malformed: {name}")
        actual = file_identity(expected.get("path", ""))
        if actual != expected:
            raise ValueError(f"quality bridge launch source changed: {name}")


def _verify_physical_generation_evidence(
    report: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    provenance = report.get("sample_provenance")
    paths = report.get("paths")
    real_set = report.get("real_set")
    if not all(isinstance(value, dict) for value in (provenance, paths, real_set)):
        raise ValueError(f"{label} generation evidence is incomplete")

    generated_dir = reject_symlink_chain(
        paths["generated_dir"],
        name=f"{label} generated directory",
    ).resolve()
    sampling_report_path = reject_symlink_chain(
        paths["sampling_report"],
        name=f"{label} sampling report",
    ).resolve()
    generated_images = find_images(generated_dir)
    recomputed_provenance = validate_sampling_provenance(
        sampling_report_path,
        generated_dir,
        generated_images,
    )
    if recomputed_provenance != provenance:
        raise ValueError(f"{label} physical sampling provenance differs")

    real_dir = reject_symlink_chain(
        paths["real_dir"],
        name=f"{label} real directory",
    ).resolve()
    real_images = find_images(real_dir)
    real_sha256 = image_tree_sha256(real_images, root=real_dir)
    if (
        int(real_set.get("image_count", -1)) != len(real_images)
        or str(real_set.get("root", "")) != real_dir.as_posix()
        or str(real_set.get("sha256", "")) != real_sha256
    ):
        raise ValueError(f"{label} physical real-set evidence differs")

    checkpoint = reject_symlink_chain(
        provenance["checkpoint"],
        name=f"{label} checkpoint",
    ).resolve()
    integrity_path = reject_symlink_chain(
        provenance["checkpoint_integrity_manifest"],
        name=f"{label} checkpoint integrity manifest",
    ).resolve()
    if integrity_path != checkpoint_integrity_path(checkpoint).resolve():
        raise ValueError(f"{label} checkpoint integrity path differs")
    integrity = verify_training_checkpoint(checkpoint)
    if (
        str(integrity.get("checkpoint_sha256", ""))
        != str(provenance.get("checkpoint_sha256", ""))
        or int(integrity.get("step", -1))
        != int(provenance.get("checkpoint_step", -1))
    ):
        raise ValueError(f"{label} physical checkpoint evidence differs")
    return {
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "bytes": int(integrity["checkpoint_bytes"]),
            "sha256": integrity["checkpoint_sha256"],
        },
        "checkpoint_integrity_manifest": file_identity(integrity_path),
        "checkpoint_step": int(integrity["step"]),
        "sampling_report": recomputed_provenance["report_identity"],
        "sampling_manifest": recomputed_provenance["manifest_identity"],
        "sampling_progress": recomputed_provenance["sampling_progress"][
            "identity"
        ],
        "sample_set_sha256": recomputed_provenance["sample_set_sha256"],
        "sample_count": len(generated_images),
        "real_set": {
            "root": real_dir.as_posix(),
            "sha256": real_sha256,
            "image_count": len(real_images),
        },
    }


def _milestone_evidence(path: Path, *, step: int) -> dict[str, Any]:
    report = read_json_object(path, name=f"quality bridge milestone {step}")
    verification = verify_milestone_source_reports(
        report,
        source_profile="quality_bridge",
    )
    evidence, warnings = validate_milestone_report(
        report,
        expected_step=step,
        source_verification=verification,
        expected_source_profile="quality_bridge",
    )
    return {
        "status": "verified",
        "report": file_identity(path),
        "evidence": evidence,
        "warnings": warnings,
    }


def build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    identities = {name: file_identity(path) for name, path in paths.items()}
    if identities["preparation"]["sha256"] != args.expected_preparation_sha256:
        raise ValueError("quality bridge preparation SHA256 differs")
    if identities["launch_receipt"]["sha256"] != args.expected_launch_receipt_sha256:
        raise ValueError("quality bridge launch receipt SHA256 differs")

    launch_receipt = read_json_object(
        paths["launch_receipt"],
        name="quality bridge launch receipt",
    )
    _verify_bound_sources(launch_receipt)

    cofitok_training = read_json_object(
        paths["cofitok_training"],
        name="quality bridge CoFiTok training",
    )
    dense_training = read_json_object(
        paths["dense_training"],
        name="quality bridge dense training",
    )
    recomputed_pair = validate_training_pair(
        cofitok_training,
        dense_training,
        expected_steps=100_000,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_dataset="imagenet_256",
        expected_recipe_stage="stability_quality_bridge",
    )
    pair_validation = read_json_object(
        paths["training_pair_validation"],
        name="quality bridge training pair validation",
    )
    if pair_validation != recomputed_pair:
        raise ValueError("quality bridge training pair validation is not reproducible")

    class_fidelity = read_json_object(
        paths["class_fidelity_qualification"],
        name="quality bridge class-fidelity qualification",
    )
    expected_class_sources = {
        "cofitok": identities["cofitok_class_fidelity"],
        "dense_identity": identities["dense_class_fidelity"],
    }
    if class_fidelity.get("sources") != expected_class_sources:
        raise ValueError("quality bridge class-fidelity source identities differ")

    cofitok_generation = read_json_object(
        paths["cofitok_generation"],
        name="quality bridge CoFiTok terminal generation metrics",
    )
    dense_generation = read_json_object(
        paths["dense_generation"],
        name="quality bridge dense terminal generation metrics",
    )
    physical_evidence = {
        "cofitok": _verify_physical_generation_evidence(
            cofitok_generation,
            label="CoFiTok terminal",
        ),
        "dense_identity": _verify_physical_generation_evidence(
            dense_generation,
            label="dense terminal",
        ),
    }
    cofitok_class = read_json_object(
        paths["cofitok_class_fidelity"],
        name="quality bridge CoFiTok class fidelity",
    )
    dense_class = read_json_object(
        paths["dense_class_fidelity"],
        name="quality bridge dense class fidelity",
    )
    validate_class_fidelity_report(cofitok_class)
    validate_class_fidelity_report(dense_class)
    if (
        cofitok_class.get("sample_provenance")
        != cofitok_generation.get("sample_provenance")
        or dense_class.get("sample_provenance")
        != dense_generation.get("sample_provenance")
    ):
        raise ValueError("quality bridge class fidelity uses another sample provenance")

    return build_quality_bridge_result(
        preparation=read_json_object(
            paths["preparation"],
            name="quality bridge preparation",
        ),
        launch_receipt=launch_receipt,
        training_pair_validation=pair_validation,
        cofitok_training_audit=read_json_object(
            paths["cofitok_training_audit"],
            name="quality bridge CoFiTok training audit",
        ),
        dense_training_audit=read_json_object(
            paths["dense_training_audit"],
            name="quality bridge dense training audit",
        ),
        milestone_evidence={
            50_000: _milestone_evidence(paths["milestone_50000"], step=50_000),
            100_000: _milestone_evidence(
                paths["milestone_100000"],
                step=100_000,
            ),
        },
        cofitok_sampling_preflight=read_json_object(
            paths["cofitok_sampling_preflight"],
            name="quality bridge CoFiTok terminal sampling preflight",
        ),
        dense_sampling_preflight=read_json_object(
            paths["dense_sampling_preflight"],
            name="quality bridge dense terminal sampling preflight",
        ),
        cofitok_generation=cofitok_generation,
        dense_generation=dense_generation,
        cofitok_checkpoint_eval=read_json_object(
            paths["cofitok_checkpoint_eval"],
            name="quality bridge CoFiTok terminal mechanism evaluation",
        ),
        dense_checkpoint_eval=read_json_object(
            paths["dense_checkpoint_eval"],
            name="quality bridge dense terminal mechanism evaluation",
        ),
        class_fidelity_qualification=class_fidelity,
        physical_evidence=physical_evidence,
        source_identities=identities,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound, non-authorizing terminal result for the "
            "full-data 100K quality bridge."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"quality bridge result already exists: {args.output}")
    report = build_from_args(args)
    write_json_report(args.output, report)
    print(report["quality_screen"]["status"])


if __name__ == "__main__":
    main()

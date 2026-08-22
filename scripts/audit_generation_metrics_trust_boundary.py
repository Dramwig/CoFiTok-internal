from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from cofitok.generation_metrics_integrity import build_metrics_trust_receipt
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RELATIVE = (
    Path("reports") / "metrics_trust_boundary_v1" / "metrics_trust_receipt.json"
)


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_git(
    project: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
) -> dict[str, Any]:
    root = reject_symlink_chain(project, name="metrics trust project").resolve()
    observed = {**git_provenance(root), "tree": _tree(root)}
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if observed != expected:
        raise ValueError(f"metrics trust project Git identity differs: {observed}")
    return observed


def validate_runtime() -> dict[str, Any]:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    omp = os.environ.get("OMP_NUM_THREADS")
    mkl = os.environ.get("MKL_NUM_THREADS")
    nice = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None
    ionice = subprocess.run(
        ["ionice", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if cuda not in {"", "-1"} or torch.cuda.is_available():
        raise ValueError("metrics trust audit must run with CUDA hidden")
    if omp != "1" or mkl != "1":
        raise ValueError("metrics trust audit must use OMP/MKL thread count 1")
    if nice is None or nice < 10 or ionice != "idle":
        raise ValueError("metrics trust audit must run at nice>=10 and ionice=idle")
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cuda_visible_devices": cuda,
        "cuda_available": torch.cuda.is_available(),
        "omp_num_threads": omp,
        "mkl_num_threads": mkl,
        "nice": nice,
        "ionice": ionice,
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "torch_version": torch.__version__,
    }


def canonical_paths(args: argparse.Namespace) -> dict[str, Path]:
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="metrics trust quality output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="metrics trust receipt output",
    ).resolve()
    expected_output = (quality_root / OUTPUT_RELATIVE).resolve()
    if output != expected_output:
        raise ValueError("metrics trust receipt output path is not canonical")
    return {"quality_root": quality_root, "output": output}


def build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    for label, value in (
        ("expected source", args.expected_source_sha256),
        ("real set", args.expected_real_set_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"metrics trust {label} SHA256 is malformed")
    paths = canonical_paths(args)
    project = reject_symlink_chain(args.project, name="metrics trust project").resolve()
    expected_source = (
        project / "scripts" / "audit_generation_metrics_trust_boundary.py"
    ).resolve()
    actual_source = reject_symlink_chain(
        Path(__file__),
        name="loaded metrics trust source",
    ).resolve()
    if actual_source != expected_source:
        raise ValueError("loaded metrics trust source is not canonical")
    source_identity = file_identity(actual_source)
    if source_identity["sha256"] != args.expected_source_sha256:
        raise ValueError("metrics trust source SHA256 differs")
    git = validate_git(
        project,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
    )
    runtime = validate_runtime()
    real_set = {
        "digest_schema": args.expected_real_set_digest_schema,
        "root": reject_symlink_chain(
            args.real_dir,
            name="metrics trust real directory",
        )
        .resolve()
        .as_posix(),
        "image_count": args.expected_real_count,
        "sha256": args.expected_real_set_sha256,
    }
    receipt = build_metrics_trust_receipt(
        git=git,
        cache_root=args.cache_root,
        real_cache_name=args.real_cache_name,
        real_set=real_set,
        creation_log=args.creation_log,
        creation_report=args.creation_report,
        inception_repair_report=args.inception_repair_report,
        inception_weight=args.inception_weight,
        vgg_weight=args.vgg_weight,
        torch_fidelity_package_root=args.torch_fidelity_package_root,
        torch_fidelity_dist_info_root=args.torch_fidelity_dist_info_root,
    )
    receipt["auditor"] = {
        "git": git,
        "source": source_identity,
        "runtime": runtime,
    }
    receipt["scope"] = {
        "quality_output_root": paths["quality_root"].as_posix(),
        "terminal_metrics_only": True,
        "result_values_modified": False,
        "cache_payloads_regenerated": False,
    }
    receipt["limitations"] = [
        (
            "This receipt physically rehashes and semantically validates the exact "
            "torch-fidelity real-cache payloads and their recorded first-creation "
            "evidence; it does not independently regenerate every feature tensor."
        ),
        (
            "The receipt is permanently non-authorizing and cannot launch training, "
            "sampling, promotion, release, or a larger-scale run."
        ),
    ]
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically attest the torch-fidelity real cache, extractor dependencies, "
            "and first-creation evidence used by terminal generation metrics."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--expected-real-set-digest-schema", required=True)
    parser.add_argument("--expected-real-set-sha256", required=True)
    parser.add_argument("--expected-real-count", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--real-cache-name", required=True)
    parser.add_argument("--creation-log", type=Path, required=True)
    parser.add_argument("--creation-report", type=Path, required=True)
    parser.add_argument("--inception-repair-report", type=Path, required=True)
    parser.add_argument("--inception-weight", type=Path, required=True)
    parser.add_argument("--vgg-weight", type=Path, required=True)
    parser.add_argument("--torch-fidelity-package-root", type=Path, required=True)
    parser.add_argument("--torch-fidelity-dist-info-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = canonical_paths(args)
    with exclusive_output_lock(paths["output"], role="generation_metrics_trust_audit"):
        report = build_from_args(args)
        identity = prepare_manifest(
            paths["output"],
            report,
            resume=args.resume,
            overwrite=False,
        )
    print(identity)


if __name__ == "__main__":
    main()

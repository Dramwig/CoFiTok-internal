from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.min_snr_terminal_guard import (
    SOURCE_CODE_PATHS,
    TERMINAL_GUARD_DIRECTORY,
    TERMINAL_GUARD_FILENAME,
    TRAINING_REPLAY_CODE_PATHS,
    _canonical_digest,
    build_terminal_guard,
)
from cofitok.inference_replay import file_identity, reject_symlink_chain
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay the completed matched Min-SNR pilot terminal evidence. "
            "The output is permanently non-authorizing."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-guard-revision", required=True)
    parser.add_argument("--expected-guard-tree", required=True)
    parser.add_argument("--expected-guard-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _guard_git(args: argparse.Namespace) -> dict[str, Any]:
    git = {
        "revision": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "branch": _git("branch", "--show-current"),
        "tracked_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no")
        ),
    }
    expected = {
        "revision": args.expected_guard_revision,
        "tree": args.expected_guard_tree,
        "branch": args.expected_guard_branch,
        "tracked_dirty": False,
    }
    if git != expected:
        raise ValueError("Min-SNR terminal guard checkout identity differs")
    return git


def build_source_code_binding(
    args: argparse.Namespace,
    *,
    guard_git: dict[str, Any],
) -> dict[str, Any]:
    source_tree = _git("rev-parse", f"{args.expected_training_revision}^{{tree}}")
    if source_tree != args.expected_training_tree:
        raise ValueError("Min-SNR training source tree differs")
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            args.expected_training_revision,
            guard_git["revision"],
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("Min-SNR training source is not a guard ancestor")
    blobs: dict[str, str] = {}
    for path in TRAINING_REPLAY_CODE_PATHS:
        source_blob = _git("rev-parse", f"{args.expected_training_revision}:{path}")
        guard_blob = _git("rev-parse", f"{guard_git['revision']}:{path}")
        if source_blob != guard_blob:
            raise ValueError(f"Min-SNR terminal replay source changed: {path}")
        blobs[path] = source_blob
    files = {
        path: file_identity(PROJECT_ROOT / path)
        for path in SOURCE_CODE_PATHS
    }
    return {
        "training_source": {
            "revision": args.expected_training_revision,
            "tree": args.expected_training_tree,
            "branch": args.expected_training_branch,
            "tracked_dirty": False,
        },
        "guard_source": guard_git,
        "training_source_is_ancestor": True,
        "training_replay_blobs": blobs,
        "files": files,
        "aggregate_sha256": _canonical_digest(files),
    }


def main() -> None:
    args = parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("MKL_NUM_THREADS") != "1"
    ):
        raise ValueError(
            "terminal physical guard requires CUDA_VISIBLE_DEVICES empty and OMP/MKL=1"
        )
    if len(args.expected_result_sha256) != 64:
        raise ValueError("expected Min-SNR result SHA256 is malformed")
    output_root = reject_symlink_chain(
        args.output_root, name="Min-SNR terminal output root"
    ).resolve()
    result = reject_symlink_chain(args.result, name="Min-SNR terminal result").resolve()
    output = reject_symlink_chain(args.output, name="Min-SNR terminal guard output")
    canonical_output = (
        output_root
        / "reports"
        / TERMINAL_GUARD_DIRECTORY
        / TERMINAL_GUARD_FILENAME
    )
    if output.resolve() != canonical_output.resolve():
        raise ValueError("Min-SNR terminal guard output path is not canonical")
    guard_git = _guard_git(args)
    source_code_binding = build_source_code_binding(args, guard_git=guard_git)
    with exclusive_output_lock(
        output,
        role="generation_matched_min_snr_terminal_physical_guard",
    ):
        if output.exists():
            raise FileExistsError(
                f"Min-SNR terminal physical guard already exists: {output}"
            )
        report = build_terminal_guard(
            output_root=output_root,
            result_path=result,
            expected_result_sha256=args.expected_result_sha256,
            expected_revision=args.expected_training_revision,
            expected_tree=args.expected_training_tree,
            expected_branch=args.expected_training_branch,
            guard_git=guard_git,
            source_code_binding=source_code_binding,
        )
        write_json_report(output, report)
    print(f"{file_sha256(output)}  {output.resolve().as_posix()}")


if __name__ == "__main__":
    main()

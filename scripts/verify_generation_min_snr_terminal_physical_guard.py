from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from cofitok.generation.min_snr_terminal_guard import (
    SOURCE_CODE_PATHS,
    TERMINAL_GUARD_DIRECTORY,
    TERMINAL_GUARD_FILENAME,
    TRAINING_REPLAY_CODE_PATHS,
    _canonical_digest,
    _verify_source_code_files,
    physical_replay,
    validate_terminal_guard,
)
from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an existing non-authorizing Min-SNR terminal physical guard."
    )
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("MKL_NUM_THREADS") != "1"
    ):
        raise ValueError(
            "terminal physical guard verifier requires CUDA hidden and OMP/MKL=1"
        )
    guard_path = reject_symlink_chain(args.guard, name="Min-SNR terminal guard")
    output_root = reject_symlink_chain(
        args.output_root,
        name="Min-SNR output root",
    ).resolve()
    canonical_guard = (
        output_root
        / "reports"
        / TERMINAL_GUARD_DIRECTORY
        / TERMINAL_GUARD_FILENAME
    )
    if guard_path.resolve() != canonical_guard.resolve():
        raise ValueError("Min-SNR terminal guard path is not canonical")
    guard_identity = file_identity(guard_path)
    if guard_identity["sha256"] != args.expected_guard_sha256:
        raise ValueError("Min-SNR terminal guard SHA256 differs")
    guard = validate_terminal_guard(
        read_json_object(guard_path, name="Min-SNR terminal physical guard")
    )
    current_git = {
        "revision": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "branch": _git("branch", "--show-current"),
        "tracked_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no")
        ),
    }
    if current_git != guard["guard_git"]:
        raise ValueError("Min-SNR terminal guard verifier Git identity differs")
    source_binding = guard["source_code_binding"]
    for name in SOURCE_CODE_PATHS:
        claimed = source_binding["files"][name]
        if Path(claimed["path"]).resolve() != (PROJECT_ROOT / name).resolve():
            raise ValueError(f"Min-SNR terminal guard source path differs: {name}")
    source_before = _verify_source_code_files(source_binding)
    for name in TRAINING_REPLAY_CODE_PATHS:
        expected_blob = source_binding["training_replay_blobs"][name]
        training_blob = _git(
            "rev-parse",
            f"{source_binding['training_source']['revision']}:{name}",
        )
        guard_blob = _git(
            "rev-parse",
            f"{source_binding['guard_source']['revision']}:{name}",
        )
        if training_blob != expected_blob or guard_blob != expected_blob:
            raise ValueError(f"Min-SNR terminal replay Git blob differs: {name}")
    replay = physical_replay(
        output_root=output_root,
        result_path=reject_symlink_chain(
            args.result, name="Min-SNR result"
        ).resolve(),
        expected_result_sha256=args.expected_result_sha256,
        expected_revision=args.expected_training_revision,
        expected_tree=args.expected_training_tree,
        expected_branch=args.expected_training_branch,
    )
    if replay != guard["evidence"]:
        raise ValueError("Min-SNR terminal physical evidence changed after guard creation")
    if _canonical_digest(replay) != guard["double_replay"]["first_sha256"]:
        raise ValueError("Min-SNR terminal guard replay digest differs")
    source_after = _verify_source_code_files(source_binding)
    if source_after != source_before:
        raise ValueError("Min-SNR terminal guard sources changed during verification")
    print(args.expected_guard_sha256)


if __name__ == "__main__":
    main()

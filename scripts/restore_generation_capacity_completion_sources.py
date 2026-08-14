from __future__ import annotations

import argparse
import os
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from cofitok.training.checkpointing import verify_training_checkpoint


def _restore(source: Path, archive: Path, expected: dict, *, label: str) -> None:
    if not archive.is_file() or file_identity(archive) != {
        "path": archive.resolve().as_posix(),
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
    }:
        raise ValueError(f"{label} archive identity differs")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        os.link(archive, source)
    if not source.is_file() or not source.samefile(archive):
        raise ValueError(f"{label} source is not the archived hard link")
    if file_identity(source) != expected:
        raise ValueError(f"{label} restored source identity differs")


def restore_sources(*, archive_path: Path, expected_archive_sha256: str) -> None:
    identity = file_identity(archive_path)
    if identity["sha256"] != expected_archive_sha256:
        raise ValueError("capacity completion source archive SHA256 differs")
    report = read_json_object(
        archive_path,
        name="capacity completion source archive",
    )
    if (
        report.get("status") != "pass"
        or report.get("role")
        != "capacity_completion_50k_source_checkpoint_archive"
    ):
        raise ValueError("capacity completion source archive contract differs")
    for method in ("cofitok", "dense_identity"):
        row = report.get("methods", {}).get(method)
        if not isinstance(row, dict):
            raise ValueError(f"capacity completion {method} archive row is missing")
        source_checkpoint = Path(row["source_checkpoint"]["path"]).resolve()
        source_integrity = Path(row["source_integrity_manifest"]["path"]).resolve()
        archive_checkpoint = Path(row["archive_checkpoint"]["path"]).resolve()
        archive_integrity = Path(row["archive_integrity_manifest"]["path"]).resolve()
        _restore(
            source_checkpoint,
            archive_checkpoint,
            row["source_checkpoint"],
            label=f"capacity completion {method} checkpoint",
        )
        _restore(
            source_integrity,
            archive_integrity,
            row["source_integrity_manifest"],
            label=f"capacity completion {method} integrity",
        )
        verified = verify_training_checkpoint(source_checkpoint)
        if int(verified.get("step", -1)) != 50_000:
            raise ValueError(f"capacity completion {method} restored step differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore pruned step-50K source names from their exact hard-link archive."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restore_sources(
        archive_path=args.archive.resolve(),
        expected_archive_sha256=args.expected_archive_sha256,
    )
    print(args.archive)


if __name__ == "__main__":
    main()

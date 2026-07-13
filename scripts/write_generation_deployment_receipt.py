from __future__ import annotations

import argparse
import json
import socket
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


SCHEMA_VERSION = 2
EXPECTED_BRANCH = "scale/generative-system"


def validate_conflict_scan(
    report: dict[str, Any],
    *,
    expected_training_revision: str,
    target_revision: str,
) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("status") != "pass"
        or report.get("current_commit") != expected_training_revision
        or report.get("target_commit") != target_revision
        or report.get("conflict_count") != 0
        or report.get("conflicts") != []
    ):
        raise ValueError("deployment conflict scan did not prove a clean transition")


def validate_runbook_syntax(
    report: dict[str, Any], *, target_revision: str
) -> int:
    runbook_git = report.get("git", {})
    runbook_paths = report.get("runbooks", [])
    discovered_count = report.get("discovered_count")
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_runbook_syntax_check"
        or report.get("enumeration") != "git_ls_files"
        or report.get("status") != "pass"
        or report.get("checked_count") != discovered_count
        or report.get("failed_count") != 0
        or report.get("failures") != []
        or not isinstance(discovered_count, int)
        or discovered_count < 1
        or not isinstance(runbook_paths, list)
        or len(runbook_paths) != discovered_count
        or len(set(runbook_paths)) != discovered_count
        or any(
            not isinstance(path, str)
            or not path.startswith("artifacts/runbooks/")
            or not path.endswith(".sh")
            for path in runbook_paths
        )
        or runbook_git.get("revision") != target_revision
        or runbook_git.get("branch") != EXPECTED_BRANCH
        or runbook_git.get("tracked_dirty") is not False
    ):
        raise ValueError("deployment runbook syntax report is invalid")
    return discovered_count


def validate_pytest_summary(summary: dict[str, Any]) -> None:
    if (
        summary.get("status") != "pass"
        or int(summary.get("tests", 0)) < 1
        or int(summary.get("failures", -1)) != 0
        or int(summary.get("errors", -1)) != 0
    ):
        raise ValueError("deployment pytest report did not pass")


def build_deployment_receipt(
    *,
    expected_training_revision: str,
    target_revision: str,
    bundle_path: str,
    bundle_bytes: int,
    bundle_sha256: str,
    bundle_heads: list[str],
    bundle_prerequisites: list[str],
    pair_validation_path: str,
    pair_validation_sha256: str,
    pair_validation: dict[str, Any],
    conflict_scan_path: str,
    conflict_scan_bytes: int,
    conflict_scan_sha256: str,
    conflict_scan: dict[str, Any],
    runbook_syntax_path: str,
    runbook_syntax_bytes: int,
    runbook_syntax_sha256: str,
    runbook_syntax: dict[str, Any],
    pytest_report_path: str,
    pytest_report_bytes: int,
    pytest_report_sha256: str,
    pytest_summary: dict[str, Any],
    git_revision: str,
    git_branch: str,
    tracked_dirty: bool,
    hostname: str,
    deployed_at: str,
) -> dict[str, Any]:
    if len(expected_training_revision) != 40 or len(target_revision) != 40:
        raise ValueError("deployment receipt requires full 40-character revisions")
    if git_revision != target_revision:
        raise ValueError("deployed Git revision does not match target revision")
    if git_branch != EXPECTED_BRANCH:
        raise ValueError(f"deployment must remain on {EXPECTED_BRANCH}")
    if tracked_dirty:
        raise ValueError("deployment receipt refuses a dirty tracked worktree")
    if bundle_bytes < 1 or len(bundle_sha256) != 64:
        raise ValueError("deployment bundle integrity evidence is invalid")
    if target_revision not in bundle_heads:
        raise ValueError("deployment bundle does not contain the target revision")
    if bundle_prerequisites != [expected_training_revision]:
        raise ValueError("deployment bundle prerequisite differs from pinned training")
    if len(pair_validation_sha256) != 64:
        raise ValueError("training-pair validation SHA256 is invalid")
    if pair_validation.get("status") != "pass":
        raise ValueError("training-pair validation did not pass")
    if pair_validation.get("expected_revision") != expected_training_revision:
        raise ValueError("training-pair validation revision differs from the pinned queue")
    for source_name, source_bytes, source_sha256 in (
        ("conflict scan", conflict_scan_bytes, conflict_scan_sha256),
        ("runbook syntax", runbook_syntax_bytes, runbook_syntax_sha256),
        ("pytest", pytest_report_bytes, pytest_report_sha256),
    ):
        if source_bytes < 1 or len(source_sha256) != 64:
            raise ValueError(f"deployment {source_name} source integrity is invalid")
    validate_conflict_scan(
        conflict_scan,
        expected_training_revision=expected_training_revision,
        target_revision=target_revision,
    )
    discovered_count = validate_runbook_syntax(
        runbook_syntax, target_revision=target_revision
    )
    validate_pytest_summary(pytest_summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "expected_training_revision": expected_training_revision,
        "target_revision": target_revision,
        "git": {
            "revision": git_revision,
            "branch": git_branch,
            "tracked_dirty": tracked_dirty,
        },
        "bundle": {
            "path": bundle_path,
            "bytes": bundle_bytes,
            "sha256": bundle_sha256,
            "heads": bundle_heads,
            "prerequisites": bundle_prerequisites,
        },
        "training_pair_validation": {
            "path": pair_validation_path,
            "sha256": pair_validation_sha256,
            "status": pair_validation["status"],
            "expected_revision": pair_validation["expected_revision"],
        },
        "verification": {
            "pytest": {
                "path": pytest_report_path,
                "bytes": pytest_report_bytes,
                "sha256": pytest_report_sha256,
                **pytest_summary,
            },
            "runbook_syntax": {
                "path": runbook_syntax_path,
                "bytes": runbook_syntax_bytes,
                "sha256": runbook_syntax_sha256,
                "status": "pass",
                "checked_count": discovered_count,
            },
            "untracked_target_conflicts": {
                "path": conflict_scan_path,
                "bytes": conflict_scan_bytes,
                "sha256": conflict_scan_sha256,
                "status": "pass",
                "target_added_path_count": conflict_scan.get(
                    "target_added_path_count"
                ),
                "conflict_count": 0,
            },
        },
        "hostname": hostname,
        "deployed_at": deployed_at,
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bundle_heads(path: Path) -> list[str]:
    output = _git("bundle", "list-heads", path.as_posix())
    heads = sorted({line.split()[0] for line in output.splitlines() if line.split()})
    if not heads:
        raise ValueError("deployment bundle has no advertised heads")
    return heads


def bundle_prerequisites(path: str | Path) -> list[str]:
    prerequisites = []
    with Path(path).open("rb") as handle:
        first_line = handle.readline()
        if not first_line.startswith(b"# v") or b"git bundle" not in first_line:
            raise ValueError("deployment bundle header is invalid")
        for raw_line in handle:
            line = raw_line.rstrip(b"\r\n")
            if not line:
                break
            if line.startswith(b"-"):
                token = line[1:].split(b" ", 1)[0]
                try:
                    value = token.decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError("bundle prerequisite is not ASCII") from error
                if len(value) not in {40, 64} or any(
                    character not in "0123456789abcdef" for character in value.lower()
                ):
                    raise ValueError("bundle prerequisite object id is invalid")
                prerequisites.append(value.lower())
    return sorted(set(prerequisites))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def pytest_junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    suites = [
        element
        for element in root.iter()
        if local_name(element) == "testsuite"
        and not any(local_name(child) == "testsuite" for child in element)
    ]
    if not suites and local_name(root) == "testsuite":
        suites = [root]
    if not suites:
        raise ValueError("pytest JUnit report has no testsuite")
    summary = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    summary["status"] = (
        "pass"
        if summary["tests"] > 0
        and summary["failures"] == 0
        and summary["errors"] == 0
        else "failed"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an integrity-bound generation upgrade deployment receipt."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--pair-validation", required=True)
    parser.add_argument("--conflict-scan", required=True)
    parser.add_argument("--runbook-syntax", required=True)
    parser.add_argument("--pytest-report", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--target-revision", required=True)
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    pair_validation_path = Path(args.pair_validation).resolve()
    conflict_scan_path = Path(args.conflict_scan).resolve()
    runbook_syntax_path = Path(args.runbook_syntax).resolve()
    pytest_report_path = Path(args.pytest_report).resolve()
    pair_validation = _read_json(pair_validation_path)
    conflict_scan = _read_json(conflict_scan_path)
    runbook_syntax = _read_json(runbook_syntax_path)
    tracked_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    receipt = build_deployment_receipt(
        expected_training_revision=args.expected_training_revision,
        target_revision=args.target_revision,
        bundle_path=bundle.as_posix(),
        bundle_bytes=bundle.stat().st_size,
        bundle_sha256=file_sha256(bundle),
        bundle_heads=_bundle_heads(bundle),
        bundle_prerequisites=bundle_prerequisites(bundle),
        pair_validation_path=pair_validation_path.as_posix(),
        pair_validation_sha256=file_sha256(pair_validation_path),
        pair_validation=pair_validation,
        conflict_scan_path=conflict_scan_path.as_posix(),
        conflict_scan_bytes=conflict_scan_path.stat().st_size,
        conflict_scan_sha256=file_sha256(conflict_scan_path),
        conflict_scan=conflict_scan,
        runbook_syntax_path=runbook_syntax_path.as_posix(),
        runbook_syntax_bytes=runbook_syntax_path.stat().st_size,
        runbook_syntax_sha256=file_sha256(runbook_syntax_path),
        runbook_syntax=runbook_syntax,
        pytest_report_path=pytest_report_path.as_posix(),
        pytest_report_bytes=pytest_report_path.stat().st_size,
        pytest_report_sha256=file_sha256(pytest_report_path),
        pytest_summary=pytest_junit_summary(pytest_report_path),
        git_revision=_git("rev-parse", "HEAD"),
        git_branch=_git("branch", "--show-current"),
        tracked_dirty=tracked_dirty,
        hostname=socket.gethostname(),
        deployed_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_report(args.output, receipt)
    print(args.output)


if __name__ == "__main__":
    main()

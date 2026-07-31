from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, git_provenance, write_json_report


SCHEMA_VERSION = 1
ROLE = "generation_large_capacity_isolated_deployment"
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
FORMAL_BRANCH = "scale/generative-system"
TARGET_BRANCH = "scale/generation-large-capacity"
MINIMUM_PYTEST_PASSED = 800
MAXIMUM_PYTEST_SKIPPED = 5


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"deployment source is missing: {resolved}")
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _bundle_identity(bundle: Path, *, repository: Path) -> dict[str, Any]:
    resolved = bundle.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"deployment bundle is missing: {resolved}")
    verify = subprocess.run(
        ["git", "bundle", "verify", str(resolved)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if verify.returncode:
        raise ValueError(
            "deployment bundle verification failed: "
            + (verify.stderr or verify.stdout).strip()
        )
    verification_output = "\n".join(
        value for value in (verify.stdout.strip(), verify.stderr.strip()) if value
    )
    prerequisites = re.findall(r"(?m)^([0-9a-f]{40})\s*$", verification_output)
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(resolved)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    advertised = []
    for line in heads:
        revision, ref = line.split(maxsplit=1)
        advertised.append({"revision": revision, "ref": ref})
    return {
        **_source_identity(resolved),
        "advertised_heads": advertised,
        "prerequisites": prerequisites,
        "verification": "git_bundle_verify_pass",
    }


def _pytest_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("deployment pytest JUnit has no test suite")
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _validate_runbook_syntax(
    report: dict[str, Any],
    *,
    checkout: Path,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    git = report.get("git", {})
    runbooks = report.get("runbooks")
    discovered = int(report.get("discovered_count", -1))
    tracked = subprocess.run(
        ["git", "ls-files", "--", "artifacts/runbooks"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    expected_runbooks = sorted(path for path in tracked if path.endswith(".sh"))
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_runbook_syntax_check"
        or report.get("status") != "pass"
        or report.get("enumeration") != "git_ls_files"
        or int(report.get("checked_count", -1)) != discovered
        or int(report.get("failed_count", -1)) != 0
        or report.get("failures") != []
        or not isinstance(runbooks, list)
        or runbooks != expected_runbooks
        or len(expected_runbooks) != discovered
        or discovered < 1
        or Path(str(report.get("runbook_root", ""))).resolve()
        != (checkout / "artifacts/runbooks").resolve()
        or git
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("large-capacity deployment runbook syntax is invalid")
    return {"checked_count": discovered, "failed_count": 0}


def build_deployment_receipt(
    *,
    formal_repository: Path,
    checkout: Path,
    bundle: Path,
    runbook_syntax_path: Path,
    pytest_report_path: Path,
    expected_bundle_sha256: str,
    expected_bundle_bytes: int,
    expected_formal_revision: str,
    expected_formal_branch: str,
    expected_target_revision: str,
    expected_target_branch: str,
    expected_prerequisites: list[str],
    minimum_pytest_passed: int,
    maximum_pytest_skipped: int,
    require_current_formal_repository: bool = True,
) -> dict[str, Any]:
    for name, revision in (
        ("formal", expected_formal_revision),
        ("target", expected_target_revision),
        *(("prerequisite", value) for value in expected_prerequisites),
    ):
        if not FULL_REVISION.fullmatch(revision):
            raise ValueError(f"{name} revision must be a full Git SHA-1")
    if expected_formal_branch != FORMAL_BRANCH:
        raise ValueError("large-capacity deployment formal branch differs")
    if expected_target_branch != TARGET_BRANCH:
        raise ValueError("large-capacity deployment target branch differs")
    if len(expected_prerequisites) != 2 or len(set(expected_prerequisites)) != 2:
        raise ValueError("large-capacity deployment requires two distinct prerequisites")
    if minimum_pytest_passed < MINIMUM_PYTEST_PASSED:
        raise ValueError("large-capacity deployment pytest pass floor was weakened")
    if maximum_pytest_skipped < 0 or maximum_pytest_skipped > MAXIMUM_PYTEST_SKIPPED:
        raise ValueError("large-capacity deployment pytest skip ceiling was weakened")
    expected_formal_git = {
        "revision": expected_formal_revision,
        "branch": expected_formal_branch,
        "tracked_dirty": False,
    }
    formal_git = (
        git_provenance(formal_repository)
        if require_current_formal_repository
        else expected_formal_git
    )
    if require_current_formal_repository and formal_git != expected_formal_git:
        raise ValueError("formal repository identity differs")
    checkout_git = git_provenance(checkout)
    expected_checkout_git = {
        "revision": expected_target_revision,
        "branch": expected_target_branch,
        "tracked_dirty": False,
    }
    if checkout_git != expected_checkout_git:
        raise ValueError("large-capacity checkout identity differs")

    bundle_identity = _bundle_identity(bundle, repository=formal_repository)
    if (
        bundle_identity["bytes"] != expected_bundle_bytes
        or bundle_identity["sha256"] != expected_bundle_sha256
        or bundle_identity["advertised_heads"]
        != [
            {
                "revision": expected_target_revision,
                "ref": f"refs/heads/{expected_target_branch}",
            }
        ]
        or set(bundle_identity["prerequisites"]) != set(expected_prerequisites)
    ):
        raise ValueError("large-capacity deployment bundle identity differs")

    runbook_report = _read_json(runbook_syntax_path)
    runbook_evidence = _validate_runbook_syntax(
        runbook_report,
        checkout=checkout,
        expected_revision=expected_target_revision,
        expected_branch=expected_target_branch,
    )
    pytest_summary = _pytest_summary(pytest_report_path)
    pytest_summary["passed"] = (
        pytest_summary["tests"]
        - pytest_summary["failures"]
        - pytest_summary["errors"]
        - pytest_summary["skipped"]
    )
    if (
        pytest_summary["passed"] < minimum_pytest_passed
        or pytest_summary["failures"] != 0
        or pytest_summary["errors"] != 0
        or pytest_summary["skipped"] > maximum_pytest_skipped
    ):
        raise ValueError("large-capacity deployment pytest evidence did not pass")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "formal_repository": {
            "path": formal_repository.resolve().as_posix(),
            "git": formal_git,
        },
        "checkout": {
            "path": checkout.resolve().as_posix(),
            "git": checkout_git,
        },
        "bundle": bundle_identity,
        "validation": {
            "runbook_syntax": {
                **_source_identity(runbook_syntax_path),
                **runbook_evidence,
            },
            "pytest": {
                **_source_identity(pytest_report_path),
                **pytest_summary,
                "minimum_passed": minimum_pytest_passed,
                "maximum_skipped": maximum_pytest_skipped,
            },
        },
        "readiness_execution_allowed": True,
        "readiness_executed": False,
        "full_training_launch_allowed": False,
        "formal_generation_completion_claimed": False,
    }


def verify_deployment_receipt(
    report: dict[str, Any],
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    require_current_formal_repository: bool = True,
) -> dict[str, Any]:
    if file_sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("large-capacity deployment receipt SHA256 differs")
    formal = report.get("formal_repository", {})
    checkout = report.get("checkout", {})
    bundle = report.get("bundle", {})
    validation = report.get("validation", {})
    pytest = validation.get("pytest", {})
    expected = build_deployment_receipt(
        formal_repository=Path(str(formal.get("path", ""))),
        checkout=Path(str(checkout.get("path", ""))),
        bundle=Path(str(bundle.get("path", ""))),
        runbook_syntax_path=Path(
            str(validation.get("runbook_syntax", {}).get("path", ""))
        ),
        pytest_report_path=Path(str(pytest.get("path", ""))),
        expected_bundle_sha256=str(bundle.get("sha256", "")),
        expected_bundle_bytes=int(bundle.get("bytes", -1)),
        expected_formal_revision=str(formal.get("git", {}).get("revision", "")),
        expected_formal_branch=str(formal.get("git", {}).get("branch", "")),
        expected_target_revision=str(checkout.get("git", {}).get("revision", "")),
        expected_target_branch=str(checkout.get("git", {}).get("branch", "")),
        expected_prerequisites=list(bundle.get("prerequisites", [])),
        minimum_pytest_passed=int(pytest.get("minimum_passed", -1)),
        maximum_pytest_skipped=int(pytest.get("maximum_skipped", -1)),
        require_current_formal_repository=require_current_formal_repository,
    )
    if report != expected:
        raise ValueError("large-capacity deployment receipt is not reproducible")
    return expected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic isolated large-capacity deployment receipt."
    )
    parser.add_argument("--formal-repository", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runbook-syntax", type=Path, required=True)
    parser.add_argument("--pytest-report", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-bundle-bytes", type=int, required=True)
    parser.add_argument("--expected-formal-revision", required=True)
    parser.add_argument("--expected-formal-branch", required=True)
    parser.add_argument("--expected-target-revision", required=True)
    parser.add_argument("--expected-target-branch", required=True)
    parser.add_argument("--expected-prerequisite", action="append", required=True)
    parser.add_argument(
        "--minimum-pytest-passed", type=int, default=MINIMUM_PYTEST_PASSED
    )
    parser.add_argument(
        "--maximum-pytest-skipped", type=int, default=MAXIMUM_PYTEST_SKIPPED
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"deployment receipt already exists: {args.output}")
    report = build_deployment_receipt(
        formal_repository=args.formal_repository,
        checkout=args.checkout,
        bundle=args.bundle,
        runbook_syntax_path=args.runbook_syntax,
        pytest_report_path=args.pytest_report,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_bundle_bytes=args.expected_bundle_bytes,
        expected_formal_revision=args.expected_formal_revision,
        expected_formal_branch=args.expected_formal_branch,
        expected_target_revision=args.expected_target_revision,
        expected_target_branch=args.expected_target_branch,
        expected_prerequisites=args.expected_prerequisite,
        minimum_pytest_passed=args.minimum_pytest_passed,
        maximum_pytest_skipped=args.maximum_pytest_skipped,
        require_current_formal_repository=True,
    )
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

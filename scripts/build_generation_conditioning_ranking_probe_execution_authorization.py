from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from cofitok.generation.conditioning_ranking_probe import (
    build_conditioning_ranking_probe_execution_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict, dict]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _replay_embedded_identity(descriptor: object, *, label: str) -> dict:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} identity is missing")
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=label,
    ).resolve()
    identity = file_identity(source)
    if identity != descriptor:
        raise ValueError(f"{label} identity differs")
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact source-bound standing authorization receipt for the "
            "four-arm 1K class-conditioning-ranking probe."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--expected-followup-decision-sha256", required=True)
    parser.add_argument("--quality-bridge-result", type=Path, required=True)
    parser.add_argument("--expected-quality-bridge-result-sha256", required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--expected-terminal-system-guard-sha256", required=True)
    parser.add_argument("--terminal-system-guard-status", type=Path, required=True)
    parser.add_argument("--expected-terminal-system-guard-status-sha256", required=True)
    parser.add_argument(
        "--requested-class-visual-audit-status",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-requested-class-visual-audit-status-sha256",
        required=True,
    )
    parser.add_argument(
        "--requested-class-visual-audit-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-requested-class-visual-audit-report-sha256",
        required=True,
    )
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparation, preparation_identity = _bound_json(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        label="conditioning-ranking preparation",
    )
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    followup, followup_identity = _bound_json(
        args.followup_decision,
        expected_sha256=args.expected_followup_decision_sha256,
        label="quality-bridge follow-up decision",
    )
    quality, quality_identity = _bound_json(
        args.quality_bridge_result,
        expected_sha256=args.expected_quality_bridge_result_sha256,
        label="quality-bridge result",
    )
    terminal, terminal_identity = _bound_json(
        args.terminal_system_guard,
        expected_sha256=args.expected_terminal_system_guard_sha256,
        label="terminal-system claim guard",
    )
    terminal_status, terminal_status_identity = _bound_json(
        args.terminal_system_guard_status,
        expected_sha256=args.expected_terminal_system_guard_status_sha256,
        label="terminal-system claim guard waiter status",
    )
    visual_status, visual_status_identity = _bound_json(
        args.requested_class_visual_audit_status,
        expected_sha256=(
            args.expected_requested_class_visual_audit_status_sha256
        ),
        label="requested-class visual-audit waiter status",
    )
    visual_report, visual_report_identity = _bound_json(
        args.requested_class_visual_audit_report,
        expected_sha256=(
            args.expected_requested_class_visual_audit_report_sha256
        ),
        label="requested-class visual-audit report",
    )
    if _replay_embedded_identity(
        followup.get("source_reports", {}).get("quality_bridge_result"),
        label="follow-up quality-bridge result",
    ) != quality_identity:
        raise ValueError("follow-up quality-bridge result binding differs")
    if _replay_embedded_identity(
        terminal.get("sources", {}).get(
            "requested_class_visual_audit_waiter_status"
        ),
        label="terminal requested-class visual-audit waiter status",
    ) != visual_status_identity:
        raise ValueError("terminal requested-class visual-audit binding differs")
    if _replay_embedded_identity(
        visual_status.get("visual_audit"),
        label="requested-class visual-audit report",
    ) != visual_report_identity:
        raise ValueError("requested-class visual-audit report binding differs")
    report = build_conditioning_ranking_probe_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        followup_decision=followup,
        followup_decision_identity=followup_identity,
        quality_bridge_result=quality,
        quality_bridge_result_identity=quality_identity,
        terminal_system_guard=terminal,
        terminal_system_guard_identity=terminal_identity,
        terminal_system_guard_status=terminal_status,
        terminal_system_guard_status_identity=terminal_status_identity,
        requested_class_visual_audit_status=visual_status,
        requested_class_visual_audit_status_identity=visual_status_identity,
        requested_class_visual_audit_report=visual_report,
        requested_class_visual_audit_report_identity=visual_report_identity,
        authorization_git=git_provenance(PROJECT_ROOT),
        authorization_tree=subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip(),
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
        expected_output_root=args.output_root,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.inference_replay import file_identity, read_json_object


APPROVAL_SCHEMA_VERSION = 1
APPROVAL_ROLE = "generation_stability_sampling_execution_approval"
APPROVED_BY = "user"
RECOVERY_SCOPE = "stability_50k_sampling_recovery_v1_execution_only"
CONFIRMATION_SCOPE = (
    "stability_50k_sampling_confirmation_10k_v1_execution_only"
)
SCOPE_CONTRACTS = {
    RECOVERY_SCOPE: {
        "approval_text": (
            "Approve the non-authorizing matched 1000-sample sampling-recovery "
            "diagnostic only."
        ),
        "authorization_boundary": {
            "sampling_recovery_execution_allowed": True,
            "sampling_confirmation_execution_allowed": False,
            "training_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "replaces_frozen_promotion_gate": False,
            "formal_protocol_change_allowed": False,
        },
    },
    CONFIRMATION_SCOPE: {
        "approval_text": (
            "Approve the non-authorizing matched 10000-sample sampling "
            "confirmation only."
        ),
        "authorization_boundary": {
            "sampling_recovery_execution_allowed": False,
            "sampling_confirmation_execution_allowed": True,
            "training_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "replaces_frozen_promotion_gate": False,
            "formal_protocol_change_allowed": False,
        },
    },
}
APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "status",
        "scope",
        "evidence",
        "git",
        "output_root",
        "approval_record",
        "authorization_boundary",
    }
)


def _source_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    path = str(identity.get("path", ""))
    bytes_count = int(identity.get("bytes", 0))
    sha256 = str(identity.get("sha256", ""))
    path_is_absolute = PurePosixPath(path).is_absolute() or (
        len(path) >= 3 and path[1] == ":" and path[2] == "/"
    )
    if (
        not path_is_absolute
        or bytes_count < 1
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("sampling execution approval evidence identity is invalid")
    return {"path": path, "bytes": bytes_count, "sha256": sha256}


def _validate_approval_time(value: Any) -> str:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("sampling execution approval time is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("sampling execution approval time must include a timezone")
    return text


def validate_sampling_execution_approval(
    approval: Mapping[str, Any],
    *,
    evidence_identity: Mapping[str, Any],
    expected_scope: str,
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    contract = SCOPE_CONTRACTS.get(expected_scope)
    if contract is None:
        raise ValueError("sampling execution approval scope is unsupported")
    if set(approval) != APPROVAL_FIELDS:
        raise ValueError("sampling execution approval top-level fields differ")
    if (
        int(approval.get("schema_version", -1)) != APPROVAL_SCHEMA_VERSION
        or approval.get("role") != APPROVAL_ROLE
        or approval.get("status") != "approved"
        or approval.get("scope") != expected_scope
    ):
        raise ValueError("sampling execution approval contract differs")
    evidence = _source_identity(evidence_identity)
    if approval.get("evidence") != evidence:
        raise ValueError("sampling execution approval binds another evidence source")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if approval.get("git") != expected_git:
        raise ValueError("sampling execution approval Git identity differs")
    output_root = str(approval.get("output_root", ""))
    if (
        output_root != expected_output_root
        or not PurePosixPath(output_root).is_absolute()
    ):
        raise ValueError("sampling execution approval output root differs")
    record = approval.get("approval_record")
    if not isinstance(record, Mapping) or set(record) != {
        "approved_by",
        "approved_at",
        "approval_text",
    }:
        raise ValueError("sampling execution approval record is incomplete")
    approved_by = str(record.get("approved_by", "")).strip()
    if approved_by != APPROVED_BY:
        raise ValueError("sampling execution approval approver differs")
    approved_at = _validate_approval_time(record.get("approved_at"))
    if record.get("approval_text") != contract["approval_text"]:
        raise ValueError("sampling execution approval text differs")
    if approval.get("authorization_boundary") != contract[
        "authorization_boundary"
    ]:
        raise ValueError("sampling execution approval boundary differs")
    return {
        "scope": expected_scope,
        "evidence": evidence,
        "git": expected_git,
        "output_root": expected_output_root,
        "approval_record": {
            "approved_by": approved_by,
            "approved_at": approved_at,
            "approval_text": contract["approval_text"],
        },
        "authorization_boundary": dict(contract["authorization_boundary"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a user-created, execution-only approval sentinel for a "
            "frozen sampling-recovery or matched confirmation stage."
        )
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-scope", choices=tuple(SCOPE_CONTRACTS), required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approval_identity = file_identity(args.approval)
    evidence_identity = file_identity(args.evidence)
    if approval_identity["sha256"] != args.expected_approval_sha256:
        raise ValueError("sampling execution approval SHA256 differs")
    evidence = validate_sampling_execution_approval(
        read_json_object(args.approval, name="sampling execution approval"),
        evidence_identity=evidence_identity,
        expected_scope=args.expected_scope,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.expected_output_root.resolve().as_posix(),
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "approval": approval_identity,
                "evidence": evidence_identity,
                "contract": evidence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

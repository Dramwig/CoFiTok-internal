from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256


FROZEN_SUPPLEMENTAL_SCHEMA_VERSION = 1
FROZEN_SUPPLEMENTAL_ROLE = "generation_stability_frozen_supplemental_qualification"
FROZEN_POSTEVAL_ROLE = "generation_stability_frozen_posteval_verification"
STABILITY_DISTRIBUTION_ROLE = (
    "generation_stability_distribution_support_qualification"
)
EXPECTED_COMBINED_CHECKS = {
    "base_gate_passed",
    "distribution_support_passed",
    "ema_rollout_stability_passed",
    "all_supplemental_quality_checks_passed",
}
EXPECTED_DISTRIBUTION_CHECKS = {
    "base_gate_metric_binding",
    "formal_matched_metrics",
    "distribution_metric_ranges",
    "minimum_precision",
    "minimum_recall",
    "matched_precision_retention",
    "matched_recall_retention",
}
EXPECTED_ROLLOUT_GATES = {
    "tail_two_energy",
    "single_token_energy",
    "ordered_rank",
    "endpoint_regression",
    "validation_regression",
    "predicted_x0_high_frequency",
    "reconstruction_regression",
    "zero_token",
    "shuffle_mismatch",
}
DIRECT_SOURCE_NAMES = {
    "promotion_gate",
    "posteval_verification",
    "distribution_support",
    "rollout_stability",
}
ROLLOUT_SOURCE_NAMES = {
    "cofitok_training",
    "dense_training",
    "cofitok_checkpoint",
    "dense_checkpoint",
    "cofitok_rollout",
    "dense_rollout",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _revision(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _git_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and _revision(value.get("revision"))
        and bool(str(value.get("branch", "")))
        and value.get("tracked_dirty") is False
    )


def _verify_bound_identity(identity: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} source identity is malformed")
    path = Path(str(identity.get("path", "")))
    actual = _source_identity(path)
    if actual != identity:
        raise ValueError(f"{label} source changed")
    return actual


def _bound_report(identity: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = _verify_bound_identity(identity, label=label)
    path = Path(actual["path"])
    return _read_json(path), actual


def _rehash_sources(
    report: dict[str, Any],
    *,
    expected_names: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    sources = report.get("sources")
    if not isinstance(sources, dict) or set(sources) != expected_names:
        raise ValueError(f"{label} source identity set differs")
    for name, identity in sources.items():
        _verify_bound_identity(identity, label=f"{label} {name}")
    return sources


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def _verify_promotion_gate(report: dict[str, Any]) -> dict[str, str]:
    provenance = report.get("provenance_contract")
    expected_keys = {
        "training_revision",
        "training_branch",
        "evaluation_revision",
        "evaluation_branch",
    }
    if (
        int(report.get("schema_version", -1)) not in {2, 3, 4}
        or report.get("stage") != "scaling"
        or report.get("source_profile") != "stability_scaling"
        or report.get("status") != "pass"
        or report.get("decision") != "promote_to_full_imagenet256"
        or not isinstance(provenance, dict)
        or set(provenance) != expected_keys
        or not _revision(provenance.get("training_revision"))
        or not _revision(provenance.get("evaluation_revision"))
        or not str(provenance.get("training_branch", ""))
        or not str(provenance.get("evaluation_branch", ""))
    ):
        raise ValueError("frozen supplemental promotion gate did not pass exactly")
    return {name: str(provenance[name]) for name in expected_keys}


def _verify_posteval(
    report: dict[str, Any],
    *,
    provenance: dict[str, str],
    builder_git: dict[str, Any],
) -> None:
    source = report.get("source")
    _bound_report(source, label="frozen post-evaluation waiter")
    expected = {**provenance, "formal_300k_allowed": False}
    if (
        report.get("schema_version") != 1
        or report.get("role") != FROZEN_POSTEVAL_ROLE
        or report.get("status") != "verified"
        or report.get("verifier_git") != builder_git
        or not _git_identity(report.get("verifier_git"))
        or report.get("expected") != expected
        or report.get("claim_boundary")
        != {
            "supplemental_execution_allowed": True,
            "replaces_postevaluation": False,
            "full_training_launch_allowed": False,
        }
    ):
        raise ValueError("frozen post-evaluation verification contract differs")


def _verify_distribution(
    report: dict[str, Any],
    *,
    promotion_gate_source: dict[str, Any],
    builder_git: dict[str, Any],
) -> None:
    sources = _rehash_sources(
        report,
        expected_names={"promotion_gate", "cofitok_generation", "dense_generation"},
        label="distribution-support",
    )
    checks = report.get("checks")
    check_names = (
        {str(row.get("name")) for row in checks}
        if isinstance(checks, list) and all(isinstance(row, dict) for row in checks)
        else set()
    )
    thresholds = report.get("thresholds")
    decision = report.get("decision_boundary")
    boundary = report.get("claim_boundary")
    if (
        report.get("schema_version") != 1
        or report.get("role") != STABILITY_DISTRIBUTION_ROLE
        or report.get("status") != "pass"
        or report.get("decision") != "distribution_support_qualified"
        or sources["promotion_gate"] != promotion_gate_source
        or report.get("builder_git") != builder_git
        or not _git_identity(report.get("builder_git"))
        or thresholds
        != {
            "min_precision": 0.10,
            "min_recall": 0.10,
            "max_precision_regression": 0.05,
            "max_recall_regression": 0.05,
        }
        or check_names != EXPECTED_DISTRIBUTION_CHECKS
        or not all(row.get("passed") is True for row in checks)
        or not isinstance(decision, dict)
        or decision.get("base_gate_passed") is not True
        or decision.get("distribution_support_passed") is not True
        or decision.get("base_gate_and_distribution_support_passed") is not True
        or decision.get("scaling_authorization_evaluated") is not False
        or decision.get("full_training_launch_allowed") is not False
        or not isinstance(boundary, dict)
        or boundary.get("supplemental_non_authorizing") is not True
        or boundary.get("replaces_generation_gate") is not False
        or boundary.get("replaces_rollout_stability_qualification") is not False
        or boundary.get("full_training_launch_allowed") is not False
        or not _all_finite(report)
    ):
        raise ValueError("distribution-support quality prerequisite did not pass exactly")


def _verify_rollout(
    report: dict[str, Any],
    *,
    provenance: dict[str, str],
    builder_git: dict[str, Any],
) -> None:
    _rehash_sources(
        report,
        expected_names=ROLLOUT_SOURCE_NAMES,
        label="EMA rollout-stability",
    )
    protocol = report.get("protocol")
    rollout = protocol.get("rollout") if isinstance(protocol, dict) else None
    identity = report.get("identity")
    gates = report.get("gates")
    pair_contract = report.get("pair_contract")
    if (
        report.get("schema_version") != 2
        or report.get("status") != "pass"
        or not isinstance(protocol, dict)
        or protocol.get("weights") != "ema"
        or int(protocol.get("checkpoint_step", -1)) != 50_000
        or int(protocol.get("checkpoint_evaluated_images", -1)) != 1_024
        or int(protocol.get("checkpoint_timestep", -1)) != 500
        or not isinstance(rollout, dict)
        or int(rollout.get("num_images", -1)) != 64
        or int(rollout.get("batch_size", -1)) != 8
        or int(rollout.get("sample_steps", -1)) != 100
        or int(rollout.get("seed", -1)) != 2029
        or float(rollout.get("guidance_scale", -1.0)) != 1.5
        or float(rollout.get("guidance_rescale", -1.0)) != 0.0
        or float(rollout.get("teacher_guidance_scale", -1.0)) != 1.0
        or rollout.get("cfg_batch_mode") != "batched"
        or rollout.get("clip_x0") is not True
        or rollout.get("precision") != "bf16"
        or not isinstance(identity, dict)
        or identity.get("git_revision") != provenance["training_revision"]
        or identity.get("training_git_revision") != provenance["training_revision"]
        or identity.get("evaluation_git_revision") != builder_git["revision"]
        or identity.get("evaluation_git_branch") != builder_git["branch"]
        or not _sha256(identity.get("cofitok_checkpoint_sha256"))
        or not _sha256(identity.get("dense_checkpoint_sha256"))
        or not isinstance(gates, dict)
        or set(gates) != EXPECTED_ROLLOUT_GATES
        or not all(
            isinstance(gate, dict) and gate.get("passed") is True
            for gate in gates.values()
        )
        or not isinstance(pair_contract, dict)
        or pair_contract.get("valid") is not True
        or not _all_finite(report)
    ):
        raise ValueError("EMA rollout-stability quality prerequisite did not pass exactly")


def verify_frozen_supplemental_report(
    report: dict[str, Any],
    *,
    report_path: Path,
    expected_report_sha256: str,
    promotion_gate_path: Path,
) -> dict[str, Any]:
    report_path = report_path.resolve()
    promotion_gate_path = promotion_gate_path.resolve()
    if file_sha256(report_path) != expected_report_sha256:
        raise ValueError("frozen supplemental report SHA256 differs")
    if _read_json(report_path) != report:
        raise ValueError("frozen supplemental report payload differs from its path")

    sources = report.get("sources")
    if not isinstance(sources, dict) or set(sources) != DIRECT_SOURCE_NAMES:
        raise ValueError("frozen supplemental direct source set differs")
    direct_reports: dict[str, dict[str, Any]] = {}
    for name, identity in sources.items():
        direct_reports[name], _ = _bound_report(
            identity,
            label=f"frozen supplemental {name}",
        )
    physical_gate = _source_identity(promotion_gate_path)
    if sources["promotion_gate"] != physical_gate:
        raise ValueError("frozen supplemental uses a different promotion gate")

    builder_git = report.get("builder_git")
    provenance = report.get("provenance_contract")
    checks = report.get("checks")
    if (
        report.get("schema_version") != FROZEN_SUPPLEMENTAL_SCHEMA_VERSION
        or report.get("role") != FROZEN_SUPPLEMENTAL_ROLE
        or report.get("status") != "pass"
        or report.get("decision") != "supplemental_quality_complete"
        or not _git_identity(builder_git)
        or not isinstance(provenance, dict)
        or provenance.get("supplemental_revision") != builder_git["revision"]
        or provenance.get("supplemental_branch") != builder_git["branch"]
        or not isinstance(checks, dict)
        or set(checks) != EXPECTED_COMBINED_CHECKS
        or not all(value is True for value in checks.values())
        or report.get("claim_boundary")
        != {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_readiness": False,
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        }
    ):
        raise ValueError("frozen supplemental quality prerequisite did not pass")

    gate_provenance = _verify_promotion_gate(direct_reports["promotion_gate"])
    expected_provenance = {
        **gate_provenance,
        "supplemental_revision": builder_git["revision"],
        "supplemental_branch": builder_git["branch"],
    }
    if provenance != expected_provenance:
        raise ValueError("frozen supplemental provenance differs from the promotion gate")
    _verify_posteval(
        direct_reports["posteval_verification"],
        provenance=gate_provenance,
        builder_git=builder_git,
    )
    _verify_distribution(
        direct_reports["distribution_support"],
        promotion_gate_source=physical_gate,
        builder_git=builder_git,
    )
    _verify_rollout(
        direct_reports["rollout_stability"],
        provenance=gate_provenance,
        builder_git=builder_git,
    )
    if (
        file_sha256(report_path) != expected_report_sha256
        or _read_json(report_path) != report
    ):
        raise ValueError("frozen supplemental report changed while verifying")
    for name, identity in sources.items():
        replayed, _ = _bound_report(
            identity,
            label=f"frozen supplemental {name}",
        )
        if replayed != direct_reports[name]:
            raise ValueError(f"frozen supplemental {name} changed while verifying")
    _verify_bound_identity(
        direct_reports["posteval_verification"]["source"],
        label="frozen post-evaluation waiter",
    )
    _rehash_sources(
        direct_reports["distribution_support"],
        expected_names={"promotion_gate", "cofitok_generation", "dense_generation"},
        label="distribution-support",
    )
    _rehash_sources(
        direct_reports["rollout_stability"],
        expected_names=ROLLOUT_SOURCE_NAMES,
        label="EMA rollout-stability",
    )
    return {
        "report": _source_identity(report_path),
        "builder_git": builder_git,
        "provenance_contract": provenance,
        "checks": checks,
        "supplemental_non_authorizing": True,
        "required_for_full_training_launch": True,
        "full_training_launch_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a source-bound frozen stability supplemental as a mandatory "
            "but non-authorizing full-training quality prerequisite."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    args = parser.parse_args()
    evidence = verify_frozen_supplemental_report(
        _read_json(args.report),
        report_path=args.report,
        expected_report_sha256=args.expected_report_sha256,
        promotion_gate_path=args.promotion_gate,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

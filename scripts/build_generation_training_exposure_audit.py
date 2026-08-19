from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report
from cofitok.generation.quality_bridge import validate_quality_bridge_preparation
from cofitok.training_exposure import (
    TRAINING_EXPOSURE_SCHEMA_VERSION,
    compare_training_exposures,
    training_exposure_summary,
)
from scripts.build_generation_milestone_report import (
    validate_milestone_report,
    verify_milestone_source_reports,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_training_exposure_audit"


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _normalized_git_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} lacks Git identity")
    revision = str(value.get("revision", ""))
    branch = str(value.get("branch", ""))
    dirty = value.get("dirty", value.get("tracked_dirty"))
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not branch
        or dirty is not False
    ):
        raise ValueError(f"{label} Git identity is invalid or dirty")
    return {"revision": revision, "branch": branch, "tracked_dirty": False}


def parse_training_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("training input must use LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def build_report(
    training_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    quality_bridge_preparation: tuple[dict[str, Any], dict[str, Any]] | None = None,
    milestone_report: tuple[dict[str, Any], dict[str, Any]] | None = None,
    expected_milestone_step: int | None = None,
    milestone_source_profile: str | None = None,
) -> dict[str, Any]:
    if not training_reports:
        raise ValueError("at least one training report is required")
    rows: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for label, (report, identity) in training_reports.items():
        if not label or label in rows:
            raise ValueError("training report labels must be unique and non-empty")
        rows[label] = training_exposure_summary(report)
        sources[label] = dict(identity)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "exposure_schema_version": TRAINING_EXPOSURE_SCHEMA_VERSION,
        "status": "pass",
        "role": REPORT_ROLE,
        "sources": sources,
        "rows": rows,
        "comparison": compare_training_exposures(rows),
        "claim_boundary": {
            "training_scale_claim_allowed": True,
            "sample_quality_claim_allowed": False,
            "method_quality_ranking_allowed": False,
            "formal_gate_substitute": False,
        },
    }
    if quality_bridge_preparation is not None:
        preparation, identity = quality_bridge_preparation
        validated = validate_quality_bridge_preparation(preparation)
        report["quality_bridge_plan"] = {
            "source": dict(identity),
            "training_exposure": validated["training_exposure"],
        }
    if milestone_report is not None:
        if expected_milestone_step is None or milestone_source_profile is None:
            raise ValueError("milestone step and source profile are required")
        if set(training_reports) != {"cofitok", "dense_identity"}:
            raise ValueError(
                "milestone-bound exposure requires cofitok and dense_identity labels"
            )
        milestone, identity = milestone_report
        source_verification = verify_milestone_source_reports(
            milestone,
            source_profile=milestone_source_profile,
        )
        evidence, warnings = validate_milestone_report(
            milestone,
            expected_step=expected_milestone_step,
            source_verification=source_verification,
            expected_source_profile=milestone_source_profile,
        )
        checkpoint_binding: dict[str, Any] = {}
        training_git_identities: dict[str, dict[str, Any]] = {}
        target_steps: set[int] = set()
        completion_states: set[bool] = set()
        for method in ("cofitok", "dense_identity"):
            training = training_reports[method][0]
            exposure = rows[method]
            if int(exposure["completed_steps"]) != expected_milestone_step:
                raise ValueError("training exposure step differs from milestone")
            latest = training.get("latest_checkpoint")
            milestone_method = milestone.get("methods", {}).get(method)
            if not isinstance(latest, dict) or not isinstance(milestone_method, dict):
                raise ValueError("milestone checkpoint binding is incomplete")
            if (
                int(latest.get("step", -1)) != expected_milestone_step
                or int(milestone_method.get("checkpoint_step", -1))
                != expected_milestone_step
                or latest.get("checkpoint_sha256")
                != milestone_method.get("checkpoint_sha256")
            ):
                raise ValueError("training and milestone checkpoints differ")
            checkpoint_binding[method] = {
                "step": expected_milestone_step,
                "checkpoint_sha256": latest["checkpoint_sha256"],
                "training_report": dict(training_reports[method][1]),
            }
            training_git_identities[method] = _normalized_git_identity(
                exposure.get("git"),
                label=f"{method} training exposure",
            )
            target_steps.add(int(exposure["target_steps"]))
            completion_states.add(bool(exposure["training_complete"]))
        if (
            len(
                {
                    tuple(value.items())
                    for value in training_git_identities.values()
                }
            )
            != 1
        ):
            raise ValueError("milestone training Git identities differ or are dirty")
        if len(target_steps) != 1 or len(completion_states) != 1:
            raise ValueError("milestone training horizons or completion states differ")
        required_comparison = {
            "same_dataset_identity",
            "same_effective_batch_size",
            "same_completed_steps",
            "same_images_seen",
            "same_dataset_normalized_exposure",
            "step_budget_directly_comparable",
            "image_budget_directly_comparable",
            "dataset_normalized_budget_directly_comparable",
        }
        if any(report["comparison"].get(name) is not True for name in required_comparison):
            raise ValueError("milestone training exposure is not exactly matched")
        expected_git = training_git_identities["cofitok"]
        source_report_git: dict[str, Any] = {}
        for name, source in source_verification["source_reports"].items():
            source_payload = json.loads(
                Path(source["path"]).read_text(encoding="utf-8")
            )
            observed_git = _normalized_git_identity(
                source_payload.get("git"),
                label=f"milestone source report {name}",
            )
            if observed_git != expected_git:
                raise ValueError("milestone source and training Git identities differ")
            source_report_git[name] = observed_git
        report["milestone_binding"] = {
            "source": dict(identity),
            "expected_step": expected_milestone_step,
            "source_profile": milestone_source_profile,
            "verification": evidence,
            "warnings": warnings,
            "training_checkpoint_binding": checkpoint_binding,
            "training_git": expected_git,
            "source_report_git": source_report_git,
            "matched_training_exposure_verified": True,
        }
        report["claim_boundary"].update(
            {
                "milestone_quality_diagnostic_allowed": True,
                "formal_generation_claim_allowed": False,
            }
        )
    elif expected_milestone_step is not None or milestone_source_profile is not None:
        raise ValueError("milestone step/profile were provided without a milestone report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable source-bound training report",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-bridge-preparation", type=Path)
    parser.add_argument("--milestone-report", type=Path)
    parser.add_argument("--expected-milestone-step", type=int)
    parser.add_argument(
        "--milestone-source-profile",
        choices=("full", "stability_full", "quality_bridge"),
    )
    args = parser.parse_args()

    reports: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw in args.training:
        label, path = parse_training_spec(raw)
        if label in reports:
            raise ValueError(f"duplicate training label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[label] = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    preparation = None
    if args.quality_bridge_preparation is not None:
        path = args.quality_bridge_preparation
        if not path.is_file():
            raise FileNotFoundError(path)
        preparation = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    milestone = None
    if args.milestone_report is not None:
        path = args.milestone_report
        if not path.is_file():
            raise FileNotFoundError(path)
        milestone = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    write_json_report(
        args.output,
        build_report(
            reports,
            quality_bridge_preparation=preparation,
            milestone_report=milestone,
            expected_milestone_step=args.expected_milestone_step,
            milestone_source_profile=args.milestone_source_profile,
        ),
    )


if __name__ == "__main__":
    main()

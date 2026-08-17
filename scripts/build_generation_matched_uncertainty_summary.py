from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report

try:
    import audit_generation_matched_uncertainty as audit
except (
    ModuleNotFoundError
):  # Imported as scripts.<module> by tests and library callers.
    from scripts import audit_generation_matched_uncertainty as audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "non_authorizing_repeated_matched_uncertainty_summary"
CLAIM_BOUNDARY = {
    **audit.CLAIM_BOUNDARY,
    "individual_stream_reports_replaced": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two or more disjoint matched uncertainty audits without "
            "pooling their block confidence intervals."
        )
    )
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-advantage", action="store_true")
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Matched uncertainty report is not an object: {path}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _protocol_without_window(report: dict[str, Any]) -> dict[str, Any]:
    signature = dict(report["matched_sampling"]["signature"])
    signature.pop("start_index", None)
    return signature


def _validate_execution_manifest_source(
    value: Any,
    *,
    path: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Matched uncertainty execution manifest is missing: {path}")
    source = value.get("source")
    stream_id = value.get("stream_id")
    if (
        value.get("status") != "verified"
        or not isinstance(stream_id, str)
        or not stream_id
        or not isinstance(source, dict)
        or not isinstance(source.get("path"), str)
        or not Path(source["path"]).is_absolute()
        or not isinstance(source.get("bytes"), int)
        or int(source["bytes"]) < 1
        or not audit._is_sha256(source.get("sha256"))
    ):
        raise ValueError(f"Matched uncertainty execution manifest is invalid: {path}")
    return {
        "stream_id": stream_id,
        "source": source,
    }


def _validate_audit(report: dict[str, Any], path: Path) -> dict[str, Any]:
    matched = report.get("matched_sampling")
    kid = report.get("paired_block_kid")
    fid = report.get("fid_point_estimates")
    sources = report.get("sources")
    implementation = report.get("implementation")
    if (
        report.get("schema_version") != audit.REPORT_SCHEMA_VERSION
        or report.get("role") != audit.REPORT_ROLE
        or report.get("claim_boundary") != audit.CLAIM_BOUNDARY
        or report.get("status") not in {"pass", "hold"}
        or not isinstance(matched, dict)
        or not isinstance(matched.get("signature"), dict)
        or int(matched.get("start_index", -1)) < 0
        or int(matched.get("end_index_exclusive", -1))
        <= int(matched.get("start_index", -1))
        or int(matched.get("sample_count", -1))
        != int(matched["end_index_exclusive"]) - int(matched["start_index"])
        or not isinstance(kid, dict)
        or not isinstance(kid.get("rows"), list)
        or len(kid["rows"]) != int(kid.get("block_count", -1))
        or not isinstance(fid, dict)
        or not isinstance(sources, dict)
        or not isinstance(sources.get("real_set"), dict)
        or not isinstance(sources.get("sample_sets"), dict)
        or not isinstance(implementation, dict)
        or implementation.get("feature_extractor") != audit.FEATURE_EXTRACTOR
        or implementation.get("feature_layer") != audit.FEATURE_LAYER
        or implementation.get("kid_kernel") != audit.KID_KERNEL
    ):
        raise ValueError(f"Matched uncertainty report contract mismatch: {path}")
    execution_manifest = _validate_execution_manifest_source(
        sources.get("execution_manifest"),
        path=path,
    )
    sample_sets = sources["sample_sets"]
    for method in ("cofitok", "dense_identity"):
        row = sample_sets.get(method)
        if (
            not isinstance(row, dict)
            or not audit._is_sha256(row.get("sha256"))
            or not audit._is_sha256(row.get("checkpoint_sha256"))
            or int(row.get("checkpoint_step", -1)) < 1
        ):
            raise ValueError(f"Matched uncertainty sample source is invalid: {path}")
    if (
        int(sample_sets["cofitok"]["checkpoint_step"])
        != int(sample_sets["dense_identity"]["checkpoint_step"])
        or not audit._is_sha256(sources["real_set"].get("sha256"))
        or not audit._is_sha256(report.get("runtime_environment_sha256"))
    ):
        raise ValueError(f"Matched uncertainty shared provenance is invalid: {path}")
    advantage = report.get("advantage_supported") is True
    if advantage != (
        report.get("status") == "pass"
        and fid.get("direction_supports_cofitok_advantage") is True
        and kid.get("uncertainty_supports_cofitok_advantage") is True
    ):
        raise ValueError(
            f"Matched uncertainty decision is internally inconsistent: {path}"
        )
    return {
        "source": _source(path),
        "execution_manifest": execution_manifest,
        "start_index": int(matched["start_index"]),
        "end_index_exclusive": int(matched["end_index_exclusive"]),
        "sample_count": int(matched["sample_count"]),
        "protocol": _protocol_without_window(report),
        "cofitok_checkpoint_sha256": sample_sets["cofitok"]["checkpoint_sha256"],
        "dense_checkpoint_sha256": sample_sets["dense_identity"]["checkpoint_sha256"],
        "checkpoint_step": int(sample_sets["cofitok"]["checkpoint_step"]),
        "real_set_sha256": sources["real_set"]["sha256"],
        "implementation": implementation,
        "runtime_environment_sha256": report.get("runtime_environment_sha256"),
        "cofitok_fid": float(fid["cofitok"]),
        "dense_fid": float(fid["dense_identity"]),
        "cofitok_relative_to_dense_fid": float(fid["cofitok_relative_to_dense"]),
        "kid_difference_mean": float(kid["cofitok_minus_dense_block_mean"]),
        "kid_ci_low": float(kid["paired_block_bootstrap"]["ci_low"]),
        "kid_ci_high": float(kid["paired_block_bootstrap"]["ci_high"]),
        "cofitok_better_blocks": int(kid["cofitok_better_blocks"]),
        "block_count": int(kid["block_count"]),
        "sign_test_p": float(kid["one_sided_exact_sign_test_p"]),
        "advantage_supported": advantage,
    }


def _windows_are_disjoint(rows: list[dict[str, Any]]) -> bool:
    ordered = sorted(rows, key=lambda row: row["start_index"])
    return all(
        left["end_index_exclusive"] <= right["start_index"]
        for left, right in pairwise(ordered)
    )


def build_summary(
    reports: list[tuple[dict[str, Any], Path]],
    *,
    builder_git: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("Repeated matched uncertainty requires at least two audits")
    rows = [_validate_audit(report, path) for report, path in reports]
    if len({row["source"]["sha256"] for row in rows}) != len(rows):
        raise ValueError("Repeated matched uncertainty reports must be unique")
    if len({row["execution_manifest"]["stream_id"] for row in rows}) != len(
        rows
    ) or len({row["execution_manifest"]["source"]["sha256"] for row in rows}) != len(
        rows
    ):
        raise ValueError(
            "Repeated matched uncertainty execution manifests must be unique"
        )
    disjoint = _windows_are_disjoint(rows)
    same_checkpoints = (
        len({row["cofitok_checkpoint_sha256"] for row in rows}) == 1
        and len({row["dense_checkpoint_sha256"] for row in rows}) == 1
        and len({row["checkpoint_step"] for row in rows}) == 1
    )
    same_real_set = len({row["real_set_sha256"] for row in rows}) == 1
    same_implementation = (
        len(
            {
                json.dumps(row["implementation"], sort_keys=True, separators=(",", ":"))
                for row in rows
            }
        )
        == 1
    )
    same_runtime = len({row["runtime_environment_sha256"] for row in rows}) == 1
    exact_protocol_replication = (
        len(
            {
                json.dumps(row["protocol"], sort_keys=True, separators=(",", ":"))
                for row in rows
            }
        )
        == 1
    )
    every_stream_supports_advantage = all(
        row["advantage_supported"] is True for row in rows
    )
    repeated_advantage_supported = (
        disjoint
        and same_checkpoints
        and same_real_set
        and same_implementation
        and same_runtime
        and every_stream_supports_advantage
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass" if repeated_advantage_supported else "hold",
        "role": REPORT_ROLE,
        "decision": (
            (
                "repeated_exact_protocol_relative_advantage_supported"
                if exact_protocol_replication
                else "repeated_cross_protocol_relative_advantage_supported"
            )
            if repeated_advantage_supported
            else "repeated_relative_advantage_not_confirmed"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "git": builder_git or git_provenance(PROJECT_ROOT),
        "checks": {
            "source_bound_execution_manifests": True,
            "disjoint_global_index_windows": disjoint,
            "same_checkpoints": same_checkpoints,
            "same_real_set": same_real_set,
            "same_implementation": same_implementation,
            "same_runtime_environment": same_runtime,
            "exact_protocol_replication": exact_protocol_replication,
            "every_stream_supports_advantage": every_stream_supports_advantage,
        },
        "stream_count": len(rows),
        "streams": sorted(rows, key=lambda row: row["start_index"]),
        "repeated_advantage_supported": repeated_advantage_supported,
        "limitations": [
            "Each stream retains its own disjoint-block confidence interval; block rows are not pooled across streams.",
            "Cross-protocol repetition supports robustness of the relative direction, not exact-protocol replication.",
            "Repeated relative advantage does not establish acceptable absolute generation quality or SOTA performance.",
            "This summary does not replace any promotion/final gate and cannot authorize training.",
        ],
    }


def main() -> int:
    args = parse_args()
    if len(args.audit) < 2:
        raise ValueError("Pass --audit at least twice")
    paths = [path.resolve() for path in args.audit]
    with exclusive_output_lock(args.output, role=REPORT_ROLE):
        report = build_summary([(_read_object(path), path) for path in paths])
        if args.output.exists():
            if not args.resume:
                raise FileExistsError(
                    "matched uncertainty summary already exists; pass --resume to verify it"
                )
            if _read_object(args.output) != report:
                raise ValueError(
                    "existing matched uncertainty summary differs from exact replay"
                )
            print(f"reused {args.output}")
        else:
            write_json_report(args.output, report)
            print(f"wrote {args.output}")
    return (
        2
        if args.require_advantage and report["repeated_advantage_supported"] is not True
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

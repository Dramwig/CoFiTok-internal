#!/usr/bin/env python
"""Audit whether bound CoFiTok runs support token-compression wording."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from cofitok.token_layout import resolve_token_layout, token_layout_summary


SCHEMA_VERSION = 1
ROLE = "cofitok_token_compression_claim_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="CoFiTok train or evaluation report containing a resolved model config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The audit is always printed to stdout.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional concise Markdown output path.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print/write a compact evidence projection while retaining source bindings.",
    )
    return parser.parse_args()


def _file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must contain a JSON object: {path}")
    return payload


def _model_candidates(value: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def visit(current: Any) -> None:
        if isinstance(current, Mapping):
            model = current.get("model")
            if isinstance(model, Mapping) and {
                "image_size",
                "image_channels",
                "token_count",
                "token_channels",
            }.issubset(model):
                candidates.append(dict(model))
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        unique[key] = candidate
    return list(unique.values())


def _resolved_model(report: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    candidates = _model_candidates(report)
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one resolved model config in {path}, found {len(candidates)}"
        )
    return candidates[0]


def _positive_int(model: Mapping[str, Any], key: str) -> int:
    value = model.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"model.{key} must be a positive integer, got {value!r}")
    return value


def _int_list(model: Mapping[str, Any], key: str) -> list[int]:
    value = model.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"model.{key} must be a list of integers, got {value!r}")
    return list(value)


def audit_model(model: Mapping[str, Any]) -> dict[str, Any]:
    image_size = _positive_int(model, "image_size")
    image_channels = _positive_int(model, "image_channels")
    token_count = _positive_int(model, "token_count")
    token_channels = _positive_int(model, "token_channels")
    if token_count <= 1:
        raise ValueError("compression-claim audit requires a factorized token_count > 1")

    channel_schedule = _int_list(model, "token_channel_schedule")
    spatial_strides = _int_list(model, "token_spatial_strides")
    active_channels = _int_list(model, "synthesis_active_token_channels")
    synthesis_strides = _int_list(model, "synthesis_token_strides")
    if bool(channel_schedule) != bool(spatial_strides):
        raise ValueError(
            "model.token_channel_schedule and model.token_spatial_strides must either "
            "both be populated or both be empty"
        )

    explicit_variable_shape = bool(channel_schedule)
    layout = resolve_token_layout(
        image_size=image_size,
        image_channels=image_channels,
        token_count=token_count,
        token_channels=token_channels,
        token_channel_schedule=channel_schedule or None,
        token_spatial_strides=spatial_strides or None,
    )
    summary = token_layout_summary(layout)
    per_token_supported = all(
        scalar_count < layout.dense_scalar_count
        for scalar_count in layout.scalar_counts
    )
    aggregate_supported = layout.total_scalar_count < layout.dense_scalar_count

    notes: list[str] = []
    if not explicit_variable_shape:
        notes.append(
            "The predictor emits token_channels at full image resolution for every token."
        )
    if active_channels and not explicit_variable_shape:
        notes.append(
            "synthesis_active_token_channels masks S_k inputs but does not reduce the "
            "emitted token field or predictor-feedback storage."
        )
    if synthesis_strides and not explicit_variable_shape:
        notes.append(
            "synthesis_token_strides changes S_k processing but does not make the emitted "
            "predictor token spatially compressed."
        )
    if per_token_supported and not aggregate_supported:
        notes.append(
            "Every individual token is smaller than the dense field, but the concatenated "
            "token sequence is not aggregate-compressed."
        )

    return {
        "model_identity": {
            key: model.get(key)
            for key in (
                "predictor_type",
                "synthesis_mode",
                "predictor_use_feedback",
            )
            if key in model
        },
        "emitted_token_layout": (
            "explicit_variable_channel_spatial_layout"
            if explicit_variable_shape
            else "full_resolution_token_fields"
        ),
        "layout": summary,
        "synthesis_active_token_channels": active_channels,
        "synthesis_token_strides": synthesis_strides,
        "claims": {
            "per_token_compression_supported": per_token_supported,
            "aggregate_sequence_compression_supported": aggregate_supported,
        },
        "notes": notes,
    }


def audit_reports(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        report = _read_json(resolved)
        rows.append(
            {
                "source": _file_identity(resolved),
                "audit": audit_model(_resolved_model(report, path=resolved)),
            }
        )

    if not rows:
        raise ValueError("at least one report is required")
    per_token_supported = all(
        row["audit"]["claims"]["per_token_compression_supported"] for row in rows
    )
    aggregate_supported = all(
        row["audit"]["claims"]["aggregate_sequence_compression_supported"]
        for row in rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "policy": {
            "compression_is_evaluated_on_actual_emitted_token_shapes": True,
            "synthesis_only_channel_masks_count_as_token_compression": False,
            "synthesis_only_resampling_counts_as_token_compression": False,
            "per_token_and_aggregate_compression_are_separate_claims": True,
        },
        "decision": {
            "report_count": len(rows),
            "per_token_compression_supported_by_all_reports": per_token_supported,
            "aggregate_sequence_compression_supported_by_all_reports": aggregate_supported,
            "allowed_paper_wording": (
                "individually compressed denoising tokens"
                if per_token_supported
                else "ordered restricted denoising components"
            ),
            "compressed_token_wording_requires_separate_evidence": not per_token_supported,
        },
        "reports": rows,
        "claim_boundary": {
            "layout_audit_only": True,
            "quality_ordering_prefix_and_zero_shuffle_metrics_recomputed": False,
            "aggregate_sequence_compression_must_not_be_inferred_from_per_token_compression": True,
        },
    }


def markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# CoFiTok token-compression claim audit",
        "",
        f"Reports audited: **{decision['report_count']}**",
        "",
        "- Per-token compression supported by every report: "
        f"**{decision['per_token_compression_supported_by_all_reports']}**",
        "- Aggregate sequence compression supported by every report: "
        f"**{decision['aggregate_sequence_compression_supported_by_all_reports']}**",
        f"- Allowed wording: `{decision['allowed_paper_wording']}`",
        "",
        "| source | emitted layout | per-token | aggregate | dense scalars | token scalars |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["reports"]:
        source = row["source"]
        audit = row["audit"]
        layout = audit["layout"]
        claims = audit["claims"]
        lines.append(
            f"| `{source['path']}` | `{audit['emitted_token_layout']}` | "
            f"{claims['per_token_compression_supported']} | "
            f"{claims['aggregate_sequence_compression_supported']} | "
            f"{layout['dense_scalar_count']} | `{layout['scalar_counts']}` |"
        )
    lines.extend(
        [
            "",
            "This audit evaluates actual emitted token shapes. Channel masks or resampling "
            "inside `S_k` do not retroactively compress full-resolution predictor tokens.",
            "",
        ]
    )
    return "\n".join(lines)


def compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "role": payload["role"],
        "policy": payload["policy"],
        "decision": payload["decision"],
        "claim_boundary": payload["claim_boundary"],
        "reports": [
            {
                "source": row["source"],
                "emitted_token_layout": row["audit"]["emitted_token_layout"],
                "model_identity": row["audit"]["model_identity"],
                "synthesis_active_token_channels": row["audit"][
                    "synthesis_active_token_channels"
                ],
                "synthesis_token_strides": row["audit"]["synthesis_token_strides"],
                "layout": {
                    key: row["audit"]["layout"][key]
                    for key in (
                        "channels",
                        "spatial_strides",
                        "scalar_counts",
                        "dense_scalar_count",
                        "total_scalar_count",
                        "aggregate_token_to_dense_ratio",
                    )
                },
                "claims": row["audit"]["claims"],
                "notes": row["audit"]["notes"],
            }
            for row in payload["reports"]
        ],
    }


def main() -> None:
    args = parse_args()
    payload = audit_reports(args.report)
    output_payload = compact_payload(payload) if args.compact else payload
    rendered = json.dumps(output_payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(payload), encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

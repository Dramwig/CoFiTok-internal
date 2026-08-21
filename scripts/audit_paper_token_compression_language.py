#!/usr/bin/env python
"""Check paper wording against a bound token-compression layout audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 2
ROLE = "cofitok_paper_token_compression_language_audit"

PER_TOKEN_PATTERNS = (
    re.compile(r"\bcompressed\s+(?:negative-noise|denoising)(?:-token)?\s+(?:component|components|token|tokens)\b", re.I),
    re.compile(r"\b(?:sum|sequence)\s+of\s+compressed\s+(?:negative-noise\s+)?(?:component|components|token|tokens)\b", re.I),
    re.compile(r"\bordered\s+compressed\s+(?:denoising\s+)?(?:component|components|token|tokens)\b", re.I),
)
AGGREGATE_PATTERNS = (
    re.compile(r"\baggregate[- ]compressed\b", re.I),
    re.compile(r"\bcompressed\s+(?:token\s+)?sequence\b", re.I),
    re.compile(r"\bsequence[- ]level\s+compression\b", re.I),
)
FIGURE_SOURCE_PATTERNS = (re.compile(r"\bcompressed\b", re.I),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-audit", type=Path, required=True)
    parser.add_argument(
        "--paper-source",
        "--source",
        action="append",
        type=Path,
        required=True,
        help="Paper, figure-generator, or text-bearing figure source to audit.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-compatible", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"layout audit must contain an object: {path}")
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("layout audit decision is missing")
    return payload


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _matches(path: Path, patterns: Iterable[re.Pattern[str]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pattern in patterns:
            for match in pattern.finditer(line):
                violations.append(
                    {
                        "line": line_number,
                        "column": match.start() + 1,
                        "match": match.group(0),
                        "pattern": pattern.pattern,
                        "text": line.strip(),
                    }
                )
    return violations


def _is_figure_source(path: Path) -> bool:
    return path.name == "make_method_figure.py" or path.suffix.lower() == ".svg"


def audit_language(layout_audit: Mapping[str, Any], sources: Iterable[Path]) -> dict[str, Any]:
    decision = layout_audit.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("layout audit decision is missing")
    per_token_supported = (
        decision.get("per_token_compression_supported_by_all_reports") is True
    )
    aggregate_supported = (
        decision.get("aggregate_sequence_compression_supported_by_all_reports") is True
    )

    rows: list[dict[str, Any]] = []
    for source in sources:
        path = source.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        per_token = [] if per_token_supported else _matches(path, PER_TOKEN_PATTERNS)
        aggregate = [] if aggregate_supported else _matches(path, AGGREGATE_PATTERNS)
        figure = (
            []
            if per_token_supported or not _is_figure_source(path)
            else _matches(path, FIGURE_SOURCE_PATTERNS)
        )
        rows.append(
            {
                "source": _identity(path),
                "unsupported_per_token_compression_wording": per_token,
                "unsupported_aggregate_sequence_compression_wording": aggregate,
                "unsupported_figure_compression_wording": figure,
                "compatible": not per_token and not aggregate and not figure,
            }
        )

    if not rows:
        raise ValueError("at least one paper source is required")
    compatible = all(row["compatible"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "layout_evidence": {
            "per_token_compression_supported": per_token_supported,
            "aggregate_sequence_compression_supported": aggregate_supported,
        },
        "decision": {
            "compatible": compatible,
            "source_count": len(rows),
            "required_locked_evidence_wording": (
                "ordered restricted denoising components"
                if not per_token_supported
                else "individually compressed denoising tokens"
            ),
            "aggregate_sequence_compression_claim_allowed": aggregate_supported,
        },
        "sources": rows,
        "policy": {
            "layout_evidence_controls_compression_wording": True,
            "text_bearing_method_figure_sources_are_guarded": True,
            "mechanism_metrics_are_not_invalidated_by_layout_wording_changes": True,
            "locked_metric_artifacts_must_not_be_rewritten": True,
        },
    }


def main() -> None:
    args = parse_args()
    payload = audit_language(_read_json(args.layout_audit), args.paper_source)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_compatible and not payload["decision"]["compatible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MATRIX = Path("docs/experiment_conditions/paper_comparison_matrix_2026-07-09.json")
DEFAULT_SUMMARY = Path("artifacts/reports/summary_2026-07-08/experiment_summary.json")
DEFAULT_BASELINES = Path("baselines/registry.json")
DEFAULT_BASELINE_REPORTS = Path("artifacts/reports/baselines")
DEFAULT_P0_TABLE = Path("artifacts/reports/p0_horizontal_table_2026-07-10_imagenet256_p0/p0_paper_table.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current paper method x dataset comparison audit table.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--baseline-registry", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--baseline-reports-root", type=Path, default=DEFAULT_BASELINE_REPORTS)
    parser.add_argument("--p0-table", type=Path, default=DEFAULT_P0_TABLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_index(summary: dict[str, Any]) -> dict[tuple[str, str, str], int]:
    index: Counter[tuple[str, str, str]] = Counter()
    for section in ("train", "quality", "generated_quality", "official_fid"):
        for row in summary.get(section, []):
            dataset = row.get("dataset")
            variant = row.get("variant")
            if dataset and variant:
                index[(section, str(dataset), str(variant))] += 1
    return dict(index)


def _baseline_clone_status(registry: dict[str, Any]) -> dict[str, str]:
    status: dict[str, str] = {}
    for entry in registry.get("entries", []):
        if entry.get("repo_type") == "external":
            status[entry["alias"]] = entry.get("adapter_status") or (
                "repo_cloned" if entry.get("clone_on_setup") else "registered"
            )
        else:
            status[entry["alias"]] = "internal"
    return status


def _p0_table_status(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    payload = read_json(path)
    status: dict[tuple[str, str], str] = {}
    for row in payload.get("rows", []):
        dataset = row.get("dataset")
        method = row.get("method")
        row_status = row.get("status")
        if dataset and method and row_status:
            status[(str(dataset), str(method))] = str(row_status)
    return status


def _baseline_report_mode(payload: dict[str, Any]) -> str:
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    metric_notes = payload.get("metric_notes") if isinstance(payload.get("metric_notes"), dict) else {}
    protocol_note = str(metric_notes.get("protocol", ""))
    if parameters.get("eval_only") is True:
        return "eval_only"
    if parameters.get("sampler") == "tokenizer_reconstruction":
        return "eval_only_tokenizer_reconstruction"
    if "Tokenizer reconstruction" in protocol_note:
        return "eval_only_tokenizer_reconstruction"
    return "standard"


def _baseline_evidence_index(root: Path) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str], set[str]]]:
    index: Counter[tuple[str, str, str]] = Counter()
    modes: dict[tuple[str, str], set[str]] = {}
    if not root.exists():
        return {}, {}
    for path in sorted(root.rglob("baseline_*_report.json")):
        payload = read_json(path)
        section = payload.get("report_type")
        dataset = payload.get("dataset")
        baseline = payload.get("baseline")
        if not section or not dataset or not baseline or payload.get("status") != "completed":
            continue
        key = (str(section), str(dataset), str(baseline))
        index[key] += 1
        modes.setdefault((str(dataset), str(baseline)), set()).add(_baseline_report_mode(payload))
        if section in {"baseline_train", "baseline_eval"}:
            index[("baseline_train_or_eval", str(dataset), str(baseline))] += 1
    return dict(index), modes


def _evidence_counts(
    method: dict[str, Any],
    dataset: dict[str, Any],
    index: dict[tuple[str, str, str], int],
    baseline_index: dict[tuple[str, str, str], int],
) -> dict[str, int]:
    dataset_alias = dataset["alias"]
    counts: dict[str, int] = {}
    for section in method.get("required_sections", []):
        if section in {"baseline_train", "baseline_eval", "baseline_train_or_eval"}:
            counts[section] = baseline_index.get((section, dataset_alias, method["alias"]), 0)
            continue
        if section in {"train", "quality", "generated_quality", "official_fid"}:
            variant = method.get("summary_variant")
            if variant:
                counts[section] = index.get((section, dataset_alias, variant), 0)
    return counts


def _cell_status(
    method: dict[str, Any],
    counts: dict[str, int],
    baseline_status: dict[str, str],
    evidence_modes: set[str] | None = None,
) -> str:
    method_type = method["type"]
    evidence_modes = evidence_modes or set()
    if method_type == "external_baseline":
        repo_state = baseline_status.get(method["alias"])
        if repo_state in {None, "registered"}:
            return "needs_repo"
        if repo_state == "repo_cloned" and not any(count > 0 for count in counts.values()):
            return "needs_adapter"
        if repo_state in {
            "dependency_ready_weight_blocked",
            "feasibility_only",
            "official_50k_eval_completed",
            "official_hf_safetensors_smoke_passed_50k_not_run",
            "official_hf_safetensors_50k_running",
            "official_pth_ema_50k_running",
            "official_pth_ema_50k_completed",
            "community_non_ema_audit_completed_official_ema_pending",
            "official_gpt_vq_generation_smoke_passed_50k_not_run",
            "official_gpt_vq_pilot128_metrics_passed_50k_running",
            "protocol_blocked",
        }:
            return "protocol_blocked"
        if repo_state == "smoke_passed_eval_only" and not any(count > 0 for count in counts.values()):
            return "needs_eval"
    if not counts:
        return "missing"
    completed = [section for section, count in counts.items() if count > 0]
    if len(completed) == len(counts):
        if evidence_modes and evidence_modes <= {"eval_only", "eval_only_tokenizer_reconstruction"}:
            return "completed_eval_only"
        return "completed"
    if completed:
        return "partial"
    return "missing"


def build_rows(
    matrix: dict[str, Any],
    summary: dict[str, Any],
    baseline_registry: dict[str, Any],
    baseline_reports_root: Path = DEFAULT_BASELINE_REPORTS,
    p0_status: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    index = _summary_index(summary)
    baseline_status = _baseline_clone_status(baseline_registry)
    baseline_index, baseline_modes = _baseline_evidence_index(baseline_reports_root)
    p0_status = p0_status or {}
    rows: list[dict[str, Any]] = []
    for dataset in matrix["datasets"]:
        for method in matrix["methods"]:
            p0_override = p0_status.get((dataset["alias"], method["alias"]))
            if p0_override == "completed":
                counts = {"p0_paper_table": 1}
                evidence_modes: set[str] = set()
                missing: list[str] = []
                present = ["p0_paper_table"]
                status = "completed"
            elif p0_override == "protocol_blocked":
                counts = {"p0_paper_table": 0}
                evidence_modes = set()
                missing = list(method.get("required_sections", []))
                present = []
                status = "protocol_blocked"
            else:
                counts = _evidence_counts(method, dataset, index, baseline_index)
                evidence_modes = baseline_modes.get((dataset["alias"], method["alias"]), set())
                missing = [section for section, count in counts.items() if count == 0]
                present = [section for section, count in counts.items() if count > 0]
                status = _cell_status(method, counts, baseline_status, evidence_modes)
            rows.append(
                {
                    "dataset": dataset["alias"],
                    "dataset_tier": dataset["tier"],
                    "resolution": dataset["resolution"],
                    "method": method["alias"],
                    "display_name": method["display_name"],
                    "method_type": method["type"],
                    "priority": method["priority"],
                    "summary_variant": method.get("summary_variant") or "",
                    "status": status,
                    "present_sections": ",".join(present),
                    "missing_sections": ",".join(missing),
                    "required_sections": ",".join(method.get("required_sections", [])),
                    "evidence_counts": json.dumps(counts, sort_keys=True),
                    "baseline_repo_status": baseline_status.get(method["alias"], ""),
                    "baseline_evidence_modes": ",".join(sorted(evidence_modes)),
                    "p0_table_status": p0_override or "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_tier",
        "resolution",
        "method",
        "display_name",
        "method_type",
        "priority",
        "status",
        "present_sections",
        "missing_sections",
        "required_sections",
        "summary_variant",
        "baseline_repo_status",
        "baseline_evidence_modes",
        "p0_table_status",
        "evidence_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], values: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in values)
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    status_counts = Counter(row["status"] for row in rows)
    dataset_counts = Counter((row["dataset"], row["status"]) for row in rows)
    priority_counts = Counter((row["priority"], row["status"]) for row in rows)

    status_rows = [[status, status_counts[status]] for status in sorted(status_counts)]
    dataset_rows = [
        [dataset, status, count]
        for (dataset, status), count in sorted(dataset_counts.items())
    ]
    priority_rows = [
        [priority, status, count]
        for (priority, status), count in sorted(priority_counts.items())
    ]
    gap_rows = [
        [
            row["dataset"],
            row["method"],
            row["priority"],
            row["status"],
            row["missing_sections"],
        ]
        for row in rows
        if row["status"] != "completed"
    ]

    body = "\n\n".join(
        [
            "# Paper Comparison Matrix Audit",
            "This file audits current evidence only. It is not a completed result table.",
            "## Status Counts",
            markdown_table(["status", "count"], status_rows),
            "## Priority Counts",
            markdown_table(["priority", "status", "count"], priority_rows),
            "## Dataset Counts",
            markdown_table(["dataset", "status", "count"], dataset_rows),
            "## Open Cells",
            markdown_table(["dataset", "method", "priority", "status", "missing evidence"], gap_rows),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix = read_json(args.matrix)
    summary = read_json(args.summary)
    baseline_registry = read_json(args.baseline_registry)
    p0_status = _p0_table_status(args.p0_table)
    rows = build_rows(matrix, summary, baseline_registry, args.baseline_reports_root, p0_status)
    payload = {
        "matrix": str(args.matrix),
        "summary": str(args.summary),
        "baseline_registry": str(args.baseline_registry),
        "baseline_reports_root": str(args.baseline_reports_root),
        "p0_table": str(args.p0_table),
        "row_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paper_comparison_matrix_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(args.output_dir / "paper_comparison_matrix_audit.csv", rows)
    write_markdown(args.output_dir / "paper_comparison_matrix_audit.md", rows)
    print(f"wrote {args.output_dir / 'paper_comparison_matrix_audit.md'}")


if __name__ == "__main__":
    main()

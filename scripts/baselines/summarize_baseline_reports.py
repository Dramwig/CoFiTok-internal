from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize external baseline train/eval reports.")
    parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports/baselines"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _parameter_index(parameter_audit: dict[str, Any] | None) -> dict[tuple[str, str], int]:
    if not parameter_audit:
        return {}
    return {
        (str(row["baseline"]), str(row["run_name"])): int(row["parameter_count"])
        for row in parameter_audit.get("rows", [])
        if row.get("status") == "completed"
    }


def collect_rows(
    reports_root: Path,
    parameter_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parameter_counts = _parameter_index(parameter_audit)
    for eval_path in sorted(reports_root.rglob("baseline_eval_report.json")):
        eval_report = _read_json(eval_path)
        train_path = eval_path.parent / "baseline_train_report.json"
        train_report = _read_json(train_path) if train_path.is_file() else {}
        baseline = eval_report.get("baseline")
        sample_steps = _get(eval_report, "parameters.sample_steps")
        sampler_nfe = _get(eval_report, "parameters.sampler_nfe")
        sampler = _get(eval_report, "parameters.sampler")
        if baseline == "improved_diffusion" and sampler_nfe is None and sample_steps is not None:
            sampler_nfe = sample_steps
        if baseline == "improved_diffusion" and sampler is None:
            sampler = "ddim"
        rows.append(
            {
                "baseline": baseline,
                "dataset": eval_report.get("dataset"),
                "run_name": eval_report.get("run_name"),
                "status": eval_report.get("status"),
                "repo_commit": train_report.get("repo_commit", ""),
                "eval_only": _get(train_report, "parameters.eval_only", False),
                "train_steps": _get(train_report, "parameters.train_steps"),
                "batch_size": _get(train_report, "parameters.batch_size"),
                "microbatch": _get(train_report, "parameters.microbatch"),
                "learning_rate": _get(train_report, "parameters.learning_rate"),
                "diffusion_steps": _get(train_report, "parameters.diffusion_steps"),
                "num_channels": _get(train_report, "parameters.num_channels"),
                "num_res_blocks": _get(train_report, "parameters.num_res_blocks"),
                "parameter_count": parameter_counts.get(
                    (str(baseline), str(eval_report.get("run_name")))
                ),
                "sample_steps": sample_steps,
                "sampler_nfe": sampler_nfe,
                "sampler": sampler,
                "sample_image_count": _get(eval_report, "evaluation.sample_image_count"),
                "real_image_count": _get(eval_report, "evaluation.real_image_count"),
                "eval_image_size": _get(eval_report, "evaluation.eval_image_size"),
                "source_resolution": _get(eval_report, "evaluation.source_resolution"),
                "lowres_frechet_proxy": _get(eval_report, "metrics.lowres_frechet_proxy"),
                "inception_frechet": _get(eval_report, "metrics.inception_frechet"),
                "reconstruction_mse": _get(eval_report, "metrics.reconstruction_mse"),
                "reconstruction_psnr_db": _get(eval_report, "metrics.reconstruction_psnr_db"),
                "eval_protocol": _get(eval_report, "metric_notes.protocol"),
                "actual_device": _get(eval_report, "runtime.actual_device"),
                "checkpoint": eval_report.get("checkpoint"),
                "samples_dir": eval_report.get("samples_dir"),
                "eval_report": eval_path.as_posix(),
                "train_report": train_path.as_posix() if train_path.is_file() else "",
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "baseline",
        "dataset",
        "train steps",
        "batch",
        "params",
        "samples",
        "real",
        "DDIM steps",
        "NFE",
        "lowres Frechet",
        "Inception Frechet",
        "recon PSNR",
        "mode",
    ]
    lines = [
        "# External Baseline Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("baseline")),
                    _fmt(row.get("dataset")),
                    _fmt(row.get("train_steps")),
                    _fmt(row.get("batch_size")),
                    _fmt(row.get("parameter_count")),
                    _fmt(row.get("sample_image_count")),
                    _fmt(row.get("real_image_count")),
                    _fmt(row.get("sample_steps")),
                    _fmt(row.get("sampler_nfe")),
                    _fmt(row.get("lowres_frechet_proxy")),
                    _fmt(row.get("inception_frechet")),
                    _fmt(row.get("reconstruction_psnr_db")),
                    "eval-only" if row.get("eval_only") else "train+eval",
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    parameter_audit = (
        _read_json(args.parameter_audit)
        if args.parameter_audit is not None and args.parameter_audit.is_file()
        else None
    )
    rows = collect_rows(args.reports_root, parameter_audit)
    payload = {
        "reports_root": args.reports_root.as_posix(),
        "row_count": len(rows),
        "rows": rows,
    }
    _write_json(args.output_dir / "baseline_summary.json", payload)
    _write_csv(args.output_dir / "baseline_summary.csv", rows)
    _write_markdown(args.output_dir / "baseline_summary.md", rows)
    print(f"wrote {args.output_dir / 'baseline_summary.md'}")


if __name__ == "__main__":
    main()

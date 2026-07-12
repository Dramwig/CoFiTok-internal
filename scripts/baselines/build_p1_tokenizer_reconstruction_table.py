from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TOKENIZER_BASELINES = {"ml_flextok", "titok_1d_tokenizer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the P1 tokenizer reconstruction report table.")
    parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports/baselines"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-filter", default="")
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


def collect_rows(reports_root: Path, run_filter: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(reports_root.rglob("baseline_eval_report.json")):
        if run_filter and run_filter not in eval_path.as_posix():
            continue
        report = _read_json(eval_path)
        baseline = str(report.get("baseline", ""))
        if baseline not in TOKENIZER_BASELINES:
            continue
        if report.get("status") != "completed":
            continue
        if _get(report, "parameters.sampler") != "tokenizer_reconstruction":
            continue
        rows.append(
            {
                "dataset": report.get("dataset"),
                "baseline": baseline,
                "run_name": report.get("run_name"),
                "image_count": _get(report, "evaluation.image_count"),
                "source_resolution": _get(report, "evaluation.source_resolution"),
                "eval_image_size": _get(report, "evaluation.eval_image_size"),
                "nfe": _get(report, "parameters.sampler_nfe"),
                "reconstruction_psnr_db": _get(report, "metrics.reconstruction_psnr_db"),
                "reconstruction_mse": _get(report, "metrics.reconstruction_mse"),
                "lowres_frechet_proxy": _get(report, "metrics.lowres_frechet_proxy"),
                "actual_device": _get(report, "runtime.actual_device"),
                "model_id": _get(report, "parameters.model_id"),
                "eval_report": eval_path.as_posix(),
            }
        )
    return sorted(rows, key=lambda row: (str(row["dataset"]), str(row["baseline"])))


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


def _write_markdown(path: Path, rows: list[dict[str, Any]], run_filter: str) -> None:
    headers = [
        "dataset",
        "baseline",
        "images",
        "src res",
        "eval res",
        "NFE",
        "PSNR",
        "MSE",
        "lowres Frechet",
    ]
    lines = [
        "# P1 Tokenizer Reconstruction Table",
        "",
        "Eval-only official pretrained tokenizer reconstruction. This is not a P0 diffusion generation table.",
        "",
        f"Run filter: `{run_filter or '<none>'}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["dataset"]),
                    _fmt(row["baseline"]),
                    _fmt(row["image_count"]),
                    _fmt(row["source_resolution"]),
                    _fmt(row["eval_image_size"]),
                    _fmt(row["nfe"]),
                    _fmt(row["reconstruction_psnr_db"]),
                    _fmt(row["reconstruction_mse"]),
                    _fmt(row["lowres_frechet_proxy"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.reports_root, args.run_filter)
    payload = {
        "reports_root": args.reports_root.as_posix(),
        "run_filter": args.run_filter,
        "row_count": len(rows),
        "rows": rows,
    }
    _write_json(args.output_dir / "p1_tokenizer_reconstruction_table.json", payload)
    _write_csv(args.output_dir / "p1_tokenizer_reconstruction_table.csv", rows)
    _write_markdown(args.output_dir / "p1_tokenizer_reconstruction_table.md", rows, args.run_filter)
    print(f"wrote {args.output_dir / 'p1_tokenizer_reconstruction_table.md'}")


if __name__ == "__main__":
    main()
